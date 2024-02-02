"""
Private module containing helper functions for public module
:py:mod:`correlate_density` that compute Clebsch-gordan tensor products on
metatensor :py:class:`TensorMap` objects.
"""

import itertools
from typing import List, Optional, Tuple, Union

from metatensor import Labels, TensorBlock, TensorMap

from . import _cg_cache, _dispatch


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


def _standardize_keys(tensor: TensorMap) -> TensorMap:
    """
    Takes a nu=1 tensor and standardizes its metadata. This involves: 1) moving
    the "species_neighbor" key to properties, if present as a dimension in the
    keys, and 2) adding dimensions in the keys for tracking the body order
    ("order_nu") and parity ("inversion_sigma") of the blocks.

    Checking for the presence of the "species_neighbor" key in the keys allows
    the option of the user pre-moving this key to the properties before calling
    `n_body_iteration_single_center`, allowing sparsity in a set of global
    neighbors to be created if desired.

    Assumes that the input `tensor` is nu=1, and has only even parity blocks.
    """
    if "species_neighbor" in tensor.keys.names:
        tensor = tensor.keys_to_properties(keys_to_move="species_neighbor")
    keys = tensor.keys.insert(
        name="order_nu",
        values=_dispatch.int_array_like([1], like=tensor.keys.values),
        index=0,
    )
    keys = keys.insert(
        name="inversion_sigma",
        values=_dispatch.int_array_like([1], like=tensor.keys.values),
        index=1,
    )
    return TensorMap(keys=keys, blocks=[b.copy() for b in tensor.blocks()])


def _parse_selected_keys(
    n_iterations: int,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    like=None,
) -> List[Union[None, Labels]]:
    """
    Parses the `selected_keys` argument passed to public functions. Checks the
    values and returns a :py:class:`list` of :py:class:`Labels` objects, one for
    each iteration of CG combination.

    `like` is required if a new :py:class:`Labels` object is to be created by
    :py:mod:`_dispatch`.
    """
    # Check angular_cutoff arg
    if angular_cutoff is not None:
        if not isinstance(angular_cutoff, int):
            raise TypeError("`angular_cutoff` must be passed as an int")
        if angular_cutoff < 1:
            raise ValueError("`angular_cutoff` must be >= 1")

    if selected_keys is None:
        if angular_cutoff is None:  # no selections at all
            selected_keys = [None] * n_iterations
        else:
            # Create a key selection with all angular channels <= the specified
            # angular cutoff
            selected_keys = [
                Labels(
                    names=["spherical_harmonics_l"],
                    values=_dispatch.int_range_like(
                        0, angular_cutoff, like=like
                    ).reshape(-1, 1),
                )
            ] * n_iterations

    if isinstance(selected_keys, Labels):
        # Create a list, but only apply a key selection at the final iteration
        selected_keys = [None] * (n_iterations - 1) + [selected_keys]

    # Check the selected_keys
    if not isinstance(selected_keys, List):
        raise TypeError(
            "`selected_keys` must be a `Labels` or List[Union[None, `Labels`]]"
        )
    if not len(selected_keys) == n_iterations:
        raise ValueError(
            "`selected_keys` must be a List[Union[None, Labels]] of length"
            " `correlation_order` - 1"
        )
    if not _dispatch.all(
        [isinstance(val, (Labels, type(None))) for val in selected_keys]
    ):
        raise TypeError("`selected_keys` must be a Labels or List[Union[None, Labels]]")

    # Now iterate over each of the Labels (or None) in the list and check
    for slct in selected_keys:
        if slct is None:
            continue
        assert isinstance(slct, Labels)
        if not _dispatch.all(
            [
                name in ["spherical_harmonics_l", "inversion_sigma"]
                for name in slct.names
            ]
        ):
            raise ValueError(
                "specified key names in `selected_keys` must be either"
                " 'spherical_harmonics_l' or 'inversion_sigma'"
            )
        if "spherical_harmonics_l" in slct.names:
            if angular_cutoff is not None:
                if not _dispatch.all(
                    slct.column("spherical_harmonics_l") <= angular_cutoff
                ):
                    raise ValueError(
                        "specified angular channels in `selected_keys` must be <= the"
                        " specified `angular_cutoff`"
                    )
            if not _dispatch.all(
                [angular_l >= 0 for angular_l in slct.column("spherical_harmonics_l")]
            ):
                raise ValueError(
                    "specified angular channels in `selected_keys` must be >= 0"
                )
        if "inversion_sigma" in slct.names:
            if not _dispatch.all(
                [parity_s in [-1, +1] for parity_s in slct.column("inversion_sigma")]
            ):
                raise ValueError(
                    "specified parities in `selected_keys` must be -1 or +1"
                )

    return selected_keys


def _parse_bool_iteration_filters(
    n_iterations: int,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> List[List[bool]]:
    """
    Parses the `skip_redundant` and `output_selection` arguments passed to
    public functions.
    """
    if isinstance(skip_redundant, bool):
        skip_redundant = [skip_redundant] * n_iterations
    if not _dispatch.all([isinstance(val, bool) for val in skip_redundant]):
        raise TypeError("`skip_redundant` must be a `bool` or `list` of `bool`")
    if not len(skip_redundant) == n_iterations:
        raise ValueError(
            "`skip_redundant` must be a bool or `list` of `bool` of length"
            " `correlation_order` - 1"
        )
    if output_selection is None:
        output_selection = [False] * (n_iterations - 1) + [True]
    else:
        if isinstance(output_selection, bool):
            output_selection = [output_selection] * n_iterations
        if not isinstance(output_selection, List):
            raise TypeError("`output_selection` must be passed as `list` of `bool`")

    if not len(output_selection) == n_iterations:
        raise ValueError(
            "`output_selection` must be a ``list`` of ``bool`` of length"
            " corresponding to the number of CG iterations"
        )
    if not _dispatch.all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a `list` of `bool`")
    if not _dispatch.all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a `list` of `bool`")

    return skip_redundant, output_selection


def _precompute_keys(
    keys_1: Labels,
    keys_2: Labels,
    n_iterations: int,
    selected_keys: List[Union[None, Labels]],
    skip_redundant: List[bool],
) -> List[Tuple[Labels, List[List[int]]]]:
    """
    Computes all the keys metadata needed to perform `n_iterations` of CG
    combination steps.

    At each iteration, a full product of the keys of two tensors, i.e. `keys_1`
    and `keys_2` is computed. Then, key selections are applied according to the
    user-defined settings: the maximum angular channel cutoff
    (`angular_cutoff`), and angular and/or parity selections specified in
    `selected_keys`.

    If `skip_redundant` is True, then keys that represent redundant CG
    operations are not included in the output keys at each step.
    """
    keys_metadata = []
    keys_out = keys_1
    for iteration in range(n_iterations):
        # Get the keys metadata for the combination of the 2 tensors
        keys_1_entries, keys_2_entries, keys_out = _precompute_keys_full_product(
            keys_1=keys_out,
            keys_2=keys_2,
        )
        if selected_keys[iteration] is not None:
            keys_1_entries, keys_2_entries, keys_out = _apply_key_selection(
                keys_1_entries,
                keys_2_entries,
                keys_out,
                selected_keys=selected_keys[iteration],
            )

        if skip_redundant[iteration]:
            keys_1_entries, keys_2_entries, keys_out = _remove_redundant_keys(
                keys_1_entries, keys_2_entries, keys_out
            )

        # Check that some keys are produced as a result of the combination
        if len(keys_out) == 0:
            raise ValueError(
                f"invalid selections: iteration {iteration + 1} produces no"
                " valid combinations. Check the `angular_cutoff` and"
                " `selected_keys` args and try again."
            )

        keys_metadata.append((keys_1_entries, keys_2_entries, keys_out))

    return keys_metadata


def _precompute_keys_full_product(
    keys_1: Labels, keys_2: Labels
) -> Tuple[List, List, Labels]:
    """
    Given the keys of 2 TensorMaps, returns the keys that would be present after
    a full CG product of these TensorMaps.

    Assumes that `keys_1` corresponds to a TensorMap with arbitrary body order,
    while `keys_2` corresponds to a TensorMap with body order 1. `keys_1`  must
    follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", ..., f"l{`nu`}", "k2", ..., f"k{`nu`-1}"]. The "lx" columns
    track the l values of the nu=1 blocks that were previously combined. The
    "kx" columns tracks the intermediate lambda values of nu > 1 blocks that
    have been combined.

    For instance, a TensorMap of body order nu=4 will have key names
    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
    "l1", "l2", "l3", "l4", "k2", "k3"]. Two nu=1 TensorMaps with blocks of
    order "l1" and "l2" were combined to form a nu=2 TensorMap with blocks of
    order "k2". This was combined with a nu=1 TensorMap with blocks of order
    "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally, this was
    combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

    .. math ::

        \\bra{
            n_1 l_1 ; n_2 l_2 k_2 ; ... ;
            n_{\nu-1} l_{\\nu-1} k_{\\nu-1} ;
            n_{\\nu} l_{\\nu} k_{\\nu}; \\lambda
        }
        \\ket{ \\rho^{\\otimes \\nu}; \\lambda M }

    `keys_2` must follow the key name convention: ["order_nu",
    "inversion_sigma", "spherical_harmonics_l", "species_center"]

    Returned is Tuple[List, List, Labels]. The first two lists correspond to the
    LabelsEntry objects of the keys being combined. The third element is a
    Labels object corresponding to the keys of the output TensorMap. Each entry
    in this Labels object corresponds to the keys is formed by combination of
    the pair of blocks indexed by correspoding key pairs in the first two lists.
    """
    # Get the correlation order of the first TensorMap.
    unique_nu = _dispatch.unique(keys_1.column("order_nu"))
    if len(unique_nu) > 1:
        raise ValueError(
            "keys_1 must correspond to a tensor of a single correlation order."
            f" Found {len(unique_nu)} body orders: {unique_nu}"
        )
    nu1 = unique_nu[0]

    # Define new correlation order of output TensorMap
    nu = nu1 + 1

    # The correlation order of the second TensorMap should be nu = 1.
    assert _dispatch.all(keys_2.column("order_nu") == 1)

    # If nu1 = 1, the key names don't yet have any "lx" columns
    if nu1 == 1:
        l_list_names = []
        new_l_list_names = ["l1", "l2"]
    else:
        l_list_names = [f"l{angular_l}" for angular_l in range(1, nu1 + 1)]
        new_l_list_names = l_list_names + [f"l{nu}"]

    # Check key names
    assert _dispatch.all(
        keys_1.names
        == ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + l_list_names
        + [f"k{k}" for k in range(2, nu1)]
    )
    assert _dispatch.all(
        keys_2.names
        == ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
    )

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + new_l_list_names
        + [f"k{k}" for k in range(2, nu)]
    )

    new_key_values = []
    keys_1_entries = []
    keys_2_entries = []
    for key_1, key_2 in itertools.product(keys_1, keys_2):
        # Unpack relevant key values
        sig1, lam1, a = key_1.values[1:4]
        sig2, lam2, a2 = key_2.values[1:4]

        # Only combine blocks of the same chemical species
        if a != a2:
            continue

        # First calculate the possible non-zero angular channels that can be
        # formed from combination of blocks of order `lam1` and `lam2`. This
        # corresponds to values in the inclusive range { |lam1 - lam2|, ...,
        # |lam1 + lam2| }
        nonzero_lams = _dispatch.int_range_like(
            abs(lam1 - lam2), abs(lam1 + lam2) + 1, like=key_1.values
        )

        # Now iterate over the non-zero angular channels and apply the custom
        # selections
        for lambda_ in nonzero_lams:
            # Calculate new sigma
            sig = sig1 * sig2 * (-1) ** (lam1 + lam2 + lambda_)

            # Extract the l and k lists from keys_1
            l_list = key_1.values[4 : 4 + nu1].tolist()
            k_list = key_1.values[4 + nu1 :].tolist()

            # Build the new keys values. l{nu} is `lam2`` (i.e.
            # "spherical_harmonics_l" of the key from `keys_2`. k{nu-1} is
            # `lam1` (i.e. "spherical_harmonics_l" of the key from `keys_1`).
            new_vals = [nu, sig, lambda_, a] + l_list + [lam2] + k_list + [lam1]
            new_key_values.append(new_vals)
            keys_1_entries.append(key_1)
            keys_2_entries.append(key_2)

    # Define new keys as the full product of keys_1 and keys_2
    keys_out = Labels(
        names=new_names,
        values=_dispatch.int_array_like(new_key_values, like=keys_1.values),
    )

    return keys_1_entries, keys_2_entries, keys_out


def _apply_key_selection(
    keys_1_entries: List, keys_2_entries: List, keys_out: Labels, selected_keys: Labels
) -> Tuple[List, List, Labels]:
    """
    Applies a selection according to `selected_keys` to the keys of an output
    TensorMap `keys_out` produced by combination of blocks indexed by keys
    entries in `keys_1_entries` and `keys_2_entries` lists.

    After application of the selections, returned is a reduced set of keys and
    set of corresponding parents key entries.

    If a selection in `selected_keys` is not valid based on the keys in
    `keys_out`, an error is raised.
    """
    # Extract the relevant columns from `selected_keys` that the selection will
    # be performed on
    keys_out_vals = [[k[name] for name in selected_keys.names] for k in keys_out]

    # First check that all of the selected keys exist in the output keys
    for slct in selected_keys.values:
        if not _dispatch.any([_dispatch.all(slct == k) for k in keys_out_vals]):
            raise ValueError(
                f"selected key {selected_keys.names} = {slct} not found"
                " in the output keys. Check the `selected_keys` argument."
            )

    # Build a mask of the selected keys
    mask = [
        _dispatch.any([_dispatch.all(i == j) for j in selected_keys.values])
        for i in keys_out_vals
    ]

    # Apply the mask to key entries and keys and return
    keys_1_entries = [k for k, isin in zip(keys_1_entries, mask) if isin]
    keys_2_entries = [k for k, isin in zip(keys_2_entries, mask) if isin]
    keys_out = Labels(names=keys_out.names, values=keys_out.values[mask])

    return keys_1_entries, keys_2_entries, keys_out


def _remove_redundant_keys(
    keys_1_entries: List, keys_2_entries: List, keys_out: Labels
) -> Tuple[List, List, Labels]:
    """
    For a Labels object `keys_out` that corresponds to the keys of a TensorMap
    formed by combined of the blocks described by the entries in the lists
    `keys_1_entries` and `keys_2_entries`, removes redundant keys.

    These are the keys that correspond to blocks that have the same sorted l
    list. The block where the l values are already sorted (i.e. l1 <= l2 <= ...
    <= ln) is kept.
    """
    # Get and check the correlation order of the input keys
    nu1 = keys_1_entries[0]["order_nu"]
    nu2 = keys_2_entries[0]["order_nu"]
    assert nu2 == 1

    # Get the correlation order of the output TensorMap
    nu = nu1 + 1

    # Identify keys of redundant blocks and remove them
    key_idxs_to_keep = []
    for key_idx, key in enumerate(keys_out):
        # Get the important key values. This is all of the keys, excpet the k
        # list
        key_vals_slice = key.values[: 4 + (nu + 1)].tolist()
        first_part, l_list = key_vals_slice[:4], key_vals_slice[4:]

        # Sort the l list
        l_list_sorted = sorted(l_list)

        # Compare the sliced key with the one recreated when the l list is
        # sorted. If they are identical, this is the key of the block that we
        # want to compute a CG combination for.
        key_slice_tuple = tuple(first_part + l_list)
        key_slice_sorted_tuple = tuple(first_part + l_list_sorted)
        if _dispatch.all(key_slice_tuple == key_slice_sorted_tuple):
            key_idxs_to_keep.append(key_idx)

    # Build a reduced Labels object for the combined keys, with redundancies removed
    keys_out_red = Labels(
        names=keys_out.names,
        values=_dispatch.int_array_like(
            [keys_out[idx].values for idx in key_idxs_to_keep],
            like=keys_1_entries[0].values,
        ),
    )

    # Store the list of reduced entries that combine to form the reduced output keys
    keys_1_entries_red = [keys_1_entries[idx] for idx in key_idxs_to_keep]
    keys_2_entries_red = [keys_2_entries[idx] for idx in key_idxs_to_keep]

    return keys_1_entries_red, keys_2_entries_red, keys_out_red


# ==================================================================
# ===== Functions to perform the CG combinations of blocks
# ==================================================================


def _combine_blocks_same_samples(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lambda_: int,
    cg_cache,
    compute_metadata_only: bool = False,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired angular channel, combines the
    values arrays and returns a new TensorBlock.
    """

    # Do the CG combination - single center so no shape pre-processing required
    if compute_metadata_only:
        combined_values = _cg_cache.combine_arrays(
            block_1.values, block_2.values, lambda_, cg_cache, return_empty_array=True
        )
    else:
        combined_values = _cg_cache.combine_arrays(
            block_1.values, block_2.values, lambda_, cg_cache, return_empty_array=False
        )

    # Infer the new nu value: block 1's properties are nu pairs of
    # "species_neighbor_x" and "nx".
    combined_nu = int((len(block_1.properties.names) / 2) + 1)

    # Define the new property names for "nx" and "species_neighbor_x"
    n_names = [f"n{i}" for i in range(1, combined_nu + 1)]
    neighbor_names = [f"species_neighbor_{i}" for i in range(1, combined_nu + 1)]
    prop_names = [item for i in zip(neighbor_names, n_names) for item in i]

    # Create a TensorBlock
    combined_block = TensorBlock(
        values=combined_values,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=_dispatch.int_range_like(
                    min_val=-lambda_, max_val=lambda_ + 1, like=block_1.values
                ).reshape(-1, 1),
            ),
        ],
        properties=Labels(
            names=prop_names,
            values=_dispatch.int_array_like(
                [
                    _dispatch.concatenate((b2, b1))
                    for b2 in block_2.properties.values
                    for b1 in block_1.properties.values
                ],
                like=block_1.values,
            ),
        ),
    )

    return combined_block
