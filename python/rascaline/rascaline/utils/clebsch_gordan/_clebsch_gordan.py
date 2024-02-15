"""
Private module containing helper functions for public module
:py:mod:`correlate_density` that compute Clebsch-gordan tensor products on
metatensor :py:class:`TensorMap` objects.
"""

from typing import List, Optional, Tuple, Union

from . import _cg_cache, _dispatch
from ._classes import (
    Array,
    Labels,
    LabelsEntry,
    TensorBlock,
    TensorMap,
    is_labels,
    torch_jit_annotate,
    torch_jit_is_scripting,
)


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
        index=0,
        name="order_nu",
        values=_dispatch.int_array_like(
            len(tensor.keys.values) * [1], like=tensor.keys.values
        ),
    )
    keys = keys.insert(
        index=1,
        name="inversion_sigma",
        values=_dispatch.int_array_like(
            len(tensor.keys.values) * [1], like=tensor.keys.values
        ),
    )
    return TensorMap(keys=keys, blocks=[b.copy() for b in tensor.blocks()])


def _parse_selected_keys(
    n_iterations: int,
    array_like: Array,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Union[Labels, None]]]] = None,
) -> List[Union[None, Labels]]:
    """
    Parses the `selected_keys` argument passed to public functions. Checks the
    values and returns a :py:class:`list` of :py:class:`Labels` objects, one for
    each iteration of CG combination.The `:param array_like:` determines the
    array backend of the Labels created
    """
    # Check the selected_keys
    if (
        (selected_keys is not None)
        and (not isinstance(selected_keys, list))
        and (not is_labels(selected_keys))
    ):
        raise TypeError(
            "`selected_keys` must be `None`, `Labels` or List[Union[None, `Labels`]]"
        )

    if isinstance(selected_keys, list):
        # Both if conditions check the same thing, the second is for metetensor-core and
        # metatensor-torch, the first one for torch-scripted metatensor-torch
        if torch_jit_is_scripting():
            if not all(
                [
                    isinstance(selected_keys[i], Labels) or (selected_keys[i] is None)
                    for i in range(len(selected_keys))
                ]
            ):
                raise TypeError(
                    "`selected_keys` must be a Labels or List[Union[None, Labels]]"
                )
        elif not all(
            [
                is_labels(selected_keys[i]) or (selected_keys[i] is None)
                for i in range(len(selected_keys))
            ]
        ):
            raise TypeError(
                "`selected_keys` must be a Labels or List[Union[None, Labels]]"
            )

    # Check angular_cutoff arg
    if angular_cutoff is not None:
        if not isinstance(angular_cutoff, int):
            raise TypeError("`angular_cutoff` must be passed as an int")
        if angular_cutoff < 1:
            raise ValueError("`angular_cutoff` must be >= 1")

    # we use a new variable for selected_keys so TorchScript can infer correct type
    selected_keys_: List[Union[None, Labels]] = []

    if selected_keys is None:
        if angular_cutoff is None:  # no selections at all
            selected_keys_ = [
                torch_jit_annotate(Union[None, Labels], None)
            ] * n_iterations
        else:
            # Create a key selection with all angular channels <= the specified
            # angular cutoff
            label: Union[None, Labels] = torch_jit_annotate(
                Union[None, Labels],
                Labels(
                    names=["spherical_harmonics_l"],
                    values=_dispatch.int_array_like(
                        list(range(0, angular_cutoff)), like=array_like
                    ).reshape(-1, 1),
                ),
            )
            selected_keys_ = [label] * n_iterations

    # Both if conditions check the same thing, we cannot write them out into one
    # condition, because otherwise the TorchScript compiler cannot infer that
    # selected_keys is Labels. We need both because isinstance(selected, Labels) works
    # with metatensor-torch only when scripted
    if torch_jit_is_scripting():
        if isinstance(selected_keys, Labels):
            # Create a list, but only apply a key selection at the final iteration
            selected_keys_ = [torch_jit_annotate(Union[None, Labels], None)] * (
                n_iterations - 1
            )
            selected_keys_.append(torch_jit_annotate(Labels, selected_keys))
    elif is_labels(selected_keys):
        # Create a list, but only apply a key selection at the final iteration
        selected_keys_ = [torch_jit_annotate(Union[None, Labels], None)] * (
            n_iterations - 1
        )
        selected_keys_.append(torch_jit_annotate(Labels, selected_keys))
    elif isinstance(selected_keys, list):
        selected_keys_ = selected_keys

    if not len(selected_keys_) == n_iterations:
        raise ValueError(
            "`selected_keys` must be a List[Union[None, Labels]] of length"
            " `correlation_order` - 1"
        )

    # Now iterate over each of the Labels (or None) in the list and check
    for slct in selected_keys_:
        if slct is None:
            continue
        if torch_jit_is_scripting():
            if not (isinstance(slct, Labels)):
                raise ValueError("Asserted that elements in `slct` are Labels")
        else:
            if not (is_labels(slct)):
                raise ValueError("Asserted that elements in `slct` are Labels")

        if not all(
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
                below_cutoff: Array = (
                    slct.column("spherical_harmonics_l") <= angular_cutoff
                )
                if not _dispatch.all(below_cutoff):
                    raise ValueError(
                        "specified angular channels in `selected_keys` must be <= the"
                        " specified `angular_cutoff`"
                    )
            above_zero = _dispatch.bool_array_like(
                [
                    bool(angular_l >= 0)
                    for angular_l in slct.column("spherical_harmonics_l")
                ],
                like=array_like,
            )
            if not _dispatch.all(above_zero):
                raise ValueError(
                    "specified angular channels in `selected_keys` must be >= 0"
                )
        if "inversion_sigma" in slct.names:
            if not _dispatch.all(
                _dispatch.bool_array_like(
                    [
                        bool(parity_s in [-1, 1])
                        for parity_s in slct.column("inversion_sigma")
                    ],
                    array_like,
                )
            ):
                raise ValueError(
                    "specified parities in `selected_keys` must be -1 or +1"
                )

    return selected_keys_


def _parse_bool_iteration_filters(
    n_iterations: int,
    skip_redundant: Union[bool, List[bool]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> Tuple[List[bool], List[bool]]:
    """
    Parses the `skip_redundant` and `output_selection` arguments passed to
    public functions.
    """
    if isinstance(skip_redundant, bool):
        skip_redundant_ = [skip_redundant] * n_iterations
    else:
        skip_redundant_ = skip_redundant

    if not all([isinstance(val, bool) for val in skip_redundant_]):
        raise TypeError("`skip_redundant` must be a `bool` or `list` of `bool`")
    if not len(skip_redundant_) == n_iterations:
        raise ValueError(
            "`skip_redundant` must be a bool or `list` of `bool` of length"
            " `correlation_order` - 1"
        )
    if output_selection is None:
        output_selection = [False] * (n_iterations - 1) + [True]
    else:
        if isinstance(output_selection, bool):
            output_selection = [output_selection] * n_iterations
        if not isinstance(output_selection, list):
            raise TypeError("`output_selection` must be passed as `list` of `bool`")

    if not len(output_selection) == n_iterations:
        raise ValueError(
            "`output_selection` must be a ``list`` of ``bool`` of length"
            " corresponding to the number of CG iterations"
        )
    if not all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a `list` of `bool`")
    if not all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a `list` of `bool`")

    return skip_redundant_, output_selection


def _precompute_keys(
    keys_1: Labels,
    keys_2: Labels,
    n_iterations: int,
    selected_keys: List[Union[None, Labels]],
    skip_redundant: List[bool],
) -> List[Tuple[List[LabelsEntry], List[LabelsEntry], Labels]]:
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
    keys_metadata: List[Tuple[List[LabelsEntry], List[LabelsEntry], Labels]] = []
    keys_out = keys_1
    for iteration in range(n_iterations):
        # Get the keys metadata for the combination of the 2 tensors
        keys_1_entries, keys_2_entries, keys_out = _precompute_keys_full_product(
            keys_1=keys_out,
            keys_2=keys_2,
        )
        selected_keys_i = selected_keys[iteration]
        if selected_keys_i is not None:
            keys_1_entries, keys_2_entries, keys_out = _apply_key_selection(
                keys_1_entries,
                keys_2_entries,
                keys_out,
                selected_keys=selected_keys_i,
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
) -> Tuple[List[LabelsEntry], List[LabelsEntry], Labels]:
    # Due to TorchScript we cannot use List[LabelsEntry]
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

    The first two lists of the returned value correspond to the LabelsEntry objects of
    the keys being combined. The third element is a Labels object corresponding to the
    keys of the output TensorMap. Each entry in this Labels object corresponds to the
    keys is formed by combination of the pair of blocks indexed by correspoding key
    pairs in the first two lists.
    """
    # Get the correlation order of the first TensorMap.
    unique_nu = _dispatch.unique(keys_1.column("order_nu"))
    if len(unique_nu) > 1:
        raise ValueError(
            "keys_1 must correspond to a tensor of a single correlation order."
            f" Found {len(unique_nu)} body orders: {unique_nu}"
        )
    nu1 = int(unique_nu[0])

    # Define new correlation order of output TensorMap
    nu = nu1 + 1

    # The correlation order of the second TensorMap should be nu = 1.
    assert _dispatch.all(keys_2.column("order_nu") == 1)

    # If nu1 = 1, the key names don't yet have any "lx" columns
    if nu1 == 1:
        l_list_names: List[str] = []
        new_l_list_names = ["l1", "l2"]
    else:
        l_list_names = [f"l{angular_l}" for angular_l in range(1, nu1 + 1)]
        new_l_list_names = l_list_names + [f"l{nu}"]

    # Check key names
    assert keys_1.names == [
        "order_nu",
        "inversion_sigma",
        "spherical_harmonics_l",
        "species_center",
    ] + l_list_names + [f"k{k}" for k in range(2, nu1)]
    assert keys_2.names == [
        "order_nu",
        "inversion_sigma",
        "spherical_harmonics_l",
        "species_center",
    ]

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + new_l_list_names
        + [f"k{k}" for k in range(2, nu)]
    )

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + new_l_list_names
        + [f"k{k}" for k in range(2, nu)]
    )

    new_key_values: List[List[int]] = []
    # Types are actually LabelsEntry, but TorchScript does not understand this.
    keys_1_entries: List[LabelsEntry] = []
    keys_2_entries: List[LabelsEntry] = []
    for i in range(len(keys_1)):
        for j in range(len(keys_2)):
            key_1 = keys_1.entry(i)
            key_2 = keys_2.entry(j)
            # Unpack relevant key values
            sig1 = int(keys_1.values[i, 1])
            lam1 = int(keys_1.values[i, 2])
            a = int(keys_1.values[i, 3])
            sig2 = int(keys_2.values[j, 1])
            lam2 = int(keys_2.values[j, 2])
            a2 = int(keys_2.values[j, 3])

            # Only combine blocks of the same chemical species
            if a != a2:
                continue

            # First calculate the possible non-zero angular channels that can be
            # formed from combination of blocks of order `lam1` and `lam2`. This
            # corresponds to values in the inclusive range { |lam1 - lam2|, ...,
            # |lam1 + lam2| }
            min_lam: int = abs(lam1 - lam2)
            max_lam: int = abs(lam1 + lam2) + 1
            nonzero_lams = list(range(min_lam, max_lam))

            # Now iterate over the non-zero angular channels and apply the custom
            # selections
            for lambda_ in nonzero_lams:
                # Calculate new sigma
                sig = int(sig1 * sig2 * (-1) ** (lam1 + lam2 + lambda_))

                # Extract the l and k lists from keys_1
                # We have to convert to int64 because of
                # https://github.com/pytorch/pytorch/issues/76295
                l_list: List[int] = _dispatch.to_int_list(keys_1.values[i, 4 : 4 + nu1])
                k_list: List[int] = _dispatch.to_int_list(keys_1.values[i, 4 + nu1 :])

                # Build the new keys values. l{nu} is `lam2`` (i.e.
                # "spherical_harmonics_l" of the key from `keys_2`. k{nu-1} is
                # `lam1` (i.e. "spherical_harmonics_l" of the key from `keys_1`).
                new_vals: List[int] = (
                    torch_jit_annotate(List[int], [nu, sig, lambda_, a])
                    + l_list
                    + [lam2]
                    + k_list
                    + [lam1]
                )
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
    keys_1_entries: List[LabelsEntry],
    keys_2_entries: List[LabelsEntry],
    keys_out: Labels,
    selected_keys: Labels,
) -> Tuple[List[LabelsEntry], List[LabelsEntry], Labels]:
    """
    Applies a selection according to `selected_keys` to the keys of an output
    TensorMap `keys_out` produced by combination of blocks indexed by keys
    entries in `keys_1_entries` and `keys_2_entries` lists.

    After application of the selections, returned is a reduced set of keys and
    set of corresponding parents key entries.

    If a selection in `selected_keys` is not valid based on the keys in
    `keys_out`, an error is raised.
    """
    # Extract the relevant columns from `selected_keys` that the selection will be
    # performed on
    col_idx = _dispatch.int_array_like(
        [keys_out.names.index(name) for name in selected_keys.names], keys_out.values
    )
    keys_out_vals = keys_out.values[:, col_idx]

    # First check that all of the selected keys exist in the output keys
    for slct in selected_keys.values:
        if not any(
            [bool(all(slct == keys_out_vals[i])) for i in range(len(keys_out_vals))]
        ):
            raise ValueError(
                f"selected key {selected_keys.names} = {slct} not found"
                " in the output keys. Check the `selected_keys` argument."
            )

    # Build a mask of the selected keys
    mask = _dispatch.bool_array_like(
        [any([bool(all(i == j)) for j in selected_keys.values]) for i in keys_out_vals],
        like=selected_keys.values,
    )

    mask_indices = _dispatch.int_array_like(
        list(range(len(keys_1_entries))), like=selected_keys.values
    )[mask]
    # Apply the mask to key entries and keys and return
    keys_1_entries = [keys_1_entries[i] for i in mask_indices]
    keys_2_entries = [keys_2_entries[i] for i in mask_indices]
    keys_out = Labels(names=keys_out.names, values=keys_out.values[mask])

    return keys_1_entries, keys_2_entries, keys_out


def _remove_redundant_keys(
    keys_1_entries: List[LabelsEntry],
    keys_2_entries: List[LabelsEntry],
    keys_out: Labels,
) -> Tuple[List[LabelsEntry], List[LabelsEntry], Labels]:
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
    key_idxs_to_keep: List[int] = []
    for key_idx in range(len(keys_out)):
        key = keys_out.entry(key_idx)
        # Get the important key values. This is all of the keys, excpet the k
        # list. We have to convert to int64 because of
        # https://github.com/pytorch/pytorch/issues/76295
        key_vals_slice: List[int] = _dispatch.to_int_list(key.values[: 4 + (nu + 1)])
        first_part, l_list = key_vals_slice[:4], key_vals_slice[4:]

        # Sort the l list
        l_list_sorted = sorted(l_list)

        # Compare the sliced key with the one recreated when the l list is
        # sorted. If they are identical, this is the key of the block that we
        # want to compute a CG combination for.
        key_slice_tuple = _dispatch.int_array_like(first_part + l_list, like=key.values)
        key_slice_sorted_tuple = _dispatch.int_array_like(
            first_part + l_list_sorted, like=key.values
        )
        if all(key_slice_tuple == key_slice_sorted_tuple):
            key_idxs_to_keep.append(key_idx)

    # Build a reduced Labels object for the combined keys, with redundancies removed
    keys_out_red = Labels(
        names=keys_out.names,
        values=keys_out.values[
            _dispatch.int_array_like(key_idxs_to_keep, like=keys_out.values)
        ],
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
    cg_coeffs: TensorMap,
    cg_backend: str,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired angular channel, combines the
    values arrays and returns a new TensorBlock.

    If cg_coeffs are None, tensor blocks with empty arrays are returned that only
    contain the metadata.
    """

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _cg_cache.combine_arrays(
        block_1.values, block_2.values, lambda_, cg_coeffs, cg_backend
    )

    # Infer the new nu value: block 1's properties are nu pairs of
    # "species_neighbor_x" and "nx".
    combined_nu = int((len(block_1.properties.names) / 2) + 1)

    # Define the new property names for "nx" and "species_neighbor_x"
    n_names = [f"n{i}" for i in range(1, combined_nu + 1)]
    neighbor_names = [f"species_neighbor_{i}" for i in range(1, combined_nu + 1)]
    prop_names_zip = [
        [neighbor_names[i], n_names[i]] for i in range(len(neighbor_names))
    ]
    prop_names: List[str] = []
    for i in range(len(prop_names_zip)):
        prop_names.extend(prop_names_zip[i])

    # create cross product list of indices in a torch-scriptable way of
    # [[i, j] for i in range(len(block_1.properties.values)) for j in
    #             range(len(block_2.properties.values))]
    # [0, 1, 2], [0, 1] -> [[0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1]]
    block_1_block_2_product_idx = _dispatch.cartesian_prod(
        _dispatch.int_range_like(
            0, len(block_2.properties.values), like=block_2.values
        ),
        _dispatch.int_range_like(
            0, len(block_1.properties.values), like=block_1.values
        ),
    )

    # Create a TensorBlock
    combined_block = TensorBlock(
        values=combined_values,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=_dispatch.int_range_like(
                    min_val=-lambda_, max_val=lambda_ + 1, like=block_1.samples.values
                ).reshape(-1, 1),
            ),
        ],
        properties=Labels(
            names=prop_names,
            values=_dispatch.int_array_like(
                [
                    _dispatch.to_int_list(block_2.properties.values[indices[0]])
                    + _dispatch.to_int_list(block_1.properties.values[indices[1]])
                    for indices in block_1_block_2_product_idx
                ],
                block_1.samples.values,
            ),
        ),
    )

    return combined_block
