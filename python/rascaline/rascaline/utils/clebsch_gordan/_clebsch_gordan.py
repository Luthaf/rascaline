"""
Private module containing helper functions for public module
:py:mod:`correlate_density` that compute Clebsch-gordan tensor products on
metatensor :py:class:`TensorMap` objects.
"""

from typing import List, Optional, Tuple, Union

from .. import _dispatch
from .._backend import Array, Labels, LabelsEntry, TensorBlock, TensorMap, is_labels
from . import _cg_cache


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


def _standardize_keys(tensor: TensorMap) -> TensorMap:
    """
    Takes a nu=1 tensor and standardizes its metadata. This involves: 1) moving
    the "neighbor_type" key to properties, if present as a dimension in the
    keys, and 2) adding dimensions in the keys for tracking the body order
    ("order_nu").

    Checking for the presence of the "neighbor_type" key in the keys allows
    the option of the user pre-moving this key to the properties before calling
    `n_body_iteration_single_center`, allowing sparsity in a set of global
    neighbors to be created if desired.

    Assumes that the input `tensor` is nu=1.
    """

    if "neighbor_type" in tensor.keys.names:
        tensor = tensor.keys_to_properties(keys_to_move="neighbor_type")

    keys = tensor.keys.insert(
        index=0,
        name="order_nu",
        values=_dispatch.int_array_like(
            len(tensor.keys.values) * [1], like=tensor.keys.values
        ),
    )
    return TensorMap(keys=keys, blocks=[b.copy() for b in tensor.blocks()])


def _parse_selected_keys(
    n_iterations: int,
    array_like: Array,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
) -> List[Labels]:
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
        raise TypeError("`selected_keys` must be `None`, `Labels` or List[`Labels`]")

    if isinstance(selected_keys, list):
        if not all(
            [
                is_labels(selected_keys[i]) or (selected_keys[i] is None)
                for i in range(len(selected_keys))
            ]
        ):
            raise TypeError("`selected_keys` must be a Labels or List[Labels]")

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
            selected_keys_ = [None] * n_iterations
        else:
            # Create a key selection with all angular channels <= the specified
            # angular cutoff
            label = Labels(
                names=["o3_lambda"],
                values=_dispatch.int_array_like(
                    list(range(0, angular_cutoff)), like=array_like
                ).reshape(-1, 1),
            )
            selected_keys_ = [label] * n_iterations

    if is_labels(selected_keys):
        # Create a list, but only apply a key selection at the final iteration
        selected_keys_ = [None] * (n_iterations - 1) + [selected_keys]
    elif isinstance(selected_keys, list):
        selected_keys_ = selected_keys

    if not len(selected_keys_) == n_iterations:
        raise ValueError(
            "`selected_keys` must be a List[Labels] of length"
            " `correlation_order` - 1"
        )

    # Now iterate over each of the Labels (or None) in the list and check
    for selected in selected_keys_:
        if selected is None:
            continue
        if not (is_labels(selected)):
            raise ValueError("Asserted that elements in `selected` are Labels")

        if not all([name in ["o3_lambda", "o3_sigma"] for name in selected.names]):
            raise ValueError(
                "specified key names in `selected_keys` must be either"
                " 'o3_lambda' or 'o3_sigma'"
            )
        if "o3_lambda" in selected.names:
            if angular_cutoff is not None:
                below_cutoff: Array = selected.column("o3_lambda") <= angular_cutoff
                if not _dispatch.all(below_cutoff):
                    raise ValueError(
                        "o3_lambda in `selected_keys` must be smaller or equal to "
                        "`angular_cutoff`"
                    )
            above_zero = _dispatch.bool_array_like(
                [bool(angular_l >= 0) for angular_l in selected.column("o3_lambda")],
                like=array_like,
            )
            if not _dispatch.all(above_zero):
                raise ValueError("o3_lambda in `selected_keys` must be >= 0")
        if "o3_sigma" in selected.names:
            if not _dispatch.all(
                _dispatch.bool_array_like(
                    [bool(sigma in [-1, 1]) for sigma in selected.column("o3_sigma")],
                    array_like,
                )
            ):
                raise ValueError("o3_sigma in `selected_keys` must be -1 or +1")

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
    selected_keys: List[Union[Labels, None]],
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
        # For TorchScript to determine the type correctly so we can subscript it
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
    Given the keys of 2 TensorMaps, returns the keys that would be present after a full
    CG product of these TensorMaps.

    Assumes that `keys_1` corresponds to a TensorMap with arbitrary body order, while
    `keys_2` corresponds to a TensorMap with body order 1. `keys_1`  must follow the key
    name convention:

    ["order_nu", "o3_sigma", "o3_lambda", "center_type", "l1", "l2", ..., f"l{`nu`}",
    "k2", ..., f"k{`nu`-1}"]. The "lx" columns track the l values of the nu=1 blocks
    that were previously combined. The "kx" columns tracks the intermediate lambda
    values of nu > 1 blocks that have been combined.

    For instance, a TensorMap of body order nu=4 will have key names ["order_nu",
    "o3_sigma", "o3_lambda", "center_type", "l1", "l2", "l3", "l4", "k2", "k3"]. Two
    nu=1 TensorMaps with blocks of order "l1" and "l2" were combined to form a nu=2
    TensorMap with blocks of order "k2". This was combined with a nu=1 TensorMap with
    blocks of order "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally,
    this was combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

    .. math ::

        \\bra{
            n_1 l_1 ; n_2 l_2 k_2 ; ... ; n_{\nu-1} l_{\\nu-1} k_{\\nu-1} ; n_{\\nu}
            l_{\\nu} k_{\\nu}; \\lambda
        } \\ket{ \\rho^{\\otimes \\nu}; \\lambda M }

    `keys_2` must follow the key name convention: ["order_nu", "o3_sigma", "o3_lambda",
    "center_type"]

    The first two lists of the returned value correspond to the LabelsEntry objects of
    the keys being combined. The third element is a Labels object corresponding to the
    keys of the output TensorMap. Each entry in this Labels object corresponds to the
    keys is formed by combination of the pair of blocks indexed by corresponding key
    pairs in the first two lists.
    """
    # Get the correlation order of the first TensorMap.
    unique_nu = _dispatch.unique(keys_1.column("order_nu"))
    if len(unique_nu) > 1:
        raise ValueError(
            "keys_1 must correspond to a tensor of a single correlation order."
            f" Found {len(unique_nu)} body orders: {unique_nu}"
        )
    nu_1 = int(unique_nu[0])

    # Define new correlation order of output TensorMap
    nu = nu_1 + 1

    # The correlation order of the second TensorMap should be nu = 1.
    assert _dispatch.all(keys_2.column("order_nu") == 1)

    # If nu_1 = 1, the key names don't yet have any "lx" columns
    if nu_1 == 1:
        l_list_names: List[str] = []
        new_l_list_names = ["l_1", "l_2"]
    else:
        l_list_names = [f"l_{angular_l}" for angular_l in range(1, nu_1 + 1)]
        new_l_list_names = l_list_names + [f"l_{nu}"]

    # Check key names
    assert keys_1.names == [
        "order_nu",
        "o3_lambda",
        "o3_sigma",
        "center_type",
    ] + l_list_names + [f"k_{k}" for k in range(2, nu_1)]

    assert keys_2.names == [
        "order_nu",
        "o3_lambda",
        "o3_sigma",
        "center_type",
    ]

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "o3_lambda", "o3_sigma", "center_type"]
        + new_l_list_names
        + [f"k_{k}" for k in range(2, nu)]
    )

    # Define key names of output Labels (i.e. for combined TensorMap)
    new_names = (
        ["order_nu", "o3_lambda", "o3_sigma", "center_type"]
        + new_l_list_names
        + [f"k_{k}" for k in range(2, nu)]
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
            sigma_1 = int(key_1["o3_sigma"])
            l_1 = int(key_1["o3_lambda"])
            type_1 = int(key_1["center_type"])
            sigma_2 = int(key_2["o3_sigma"])
            l_2 = int(key_2["o3_lambda"])
            type_2 = int(key_2["center_type"])

            # Only combine blocks of the same atomic types
            if type_1 != type_2:
                continue

            # First calculate the possible non-zero angular channels that can be formed
            # from combination of blocks of order `lambda_1` and `lambda_2`. This
            # corresponds to values in the inclusive range { |lambda_1 - lambda_2|, ...,
            # |lambda_1 + lambda_2| }
            min_lam = abs(l_1 - l_2)
            max_lam = abs(l_1 + l_2) + 1
            nonzero_lams = list(range(min_lam, max_lam))

            # Now iterate over the non-zero angular channels and apply the custom
            # selections
            for o3_lambda in nonzero_lams:
                # Calculate new sigma
                o3_sigma = int(sigma_1 * sigma_2 * (-1) ** (l_1 + l_2 + o3_lambda))

                # Extract the l and k lists from keys_1
                l_list = _dispatch.to_int_list(keys_1.values[i, 4 : 4 + nu_1])
                k_list = _dispatch.to_int_list(keys_1.values[i, 4 + nu_1 :])

                # Build the new keys values. l{nu} is `lambda_2`` (i.e.
                # "o3_lambda" of the key from `keys_2`. k{nu-1} is
                # `lambda_1` (i.e. "o3_lambda" of the key from `keys_1`).
                new_vals: List[int] = (
                    [nu, o3_lambda, o3_sigma, type_1] + l_list + [l_2] + k_list + [l_1]
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
    for selected in selected_keys.values:
        if not any(
            [bool(all(selected == keys_out_vals[i])) for i in range(len(keys_out_vals))]
        ):
            raise ValueError(
                f"selected key {selected_keys.names} = {selected} not found"
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
        # Get the important key values. This is all of the keys, except the k
        # list.
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
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired angular channel, combines the
    values arrays and returns a new TensorBlock.

    If cg_coefficients are None, tensor blocks with empty arrays are returned that only
    contain the metadata.
    """

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _cg_cache.combine_arrays(
        block_1.values, block_2.values, lambda_, cg_coefficients, cg_backend
    )

    # Infer the new nu value: block 1's properties are nu pairs of
    # "neighbor_type_x" and "nx".
    combined_nu = int((len(block_1.properties.names) / 2) + 1)

    # Define the new property names for "n_<i>" and "neighbor_<i>_type"
    n_names = [f"n_{i}" for i in range(1, combined_nu + 1)]
    neighbor_names = [f"neighbor_{i}_type" for i in range(1, combined_nu + 1)]
    property_names_zip = [
        [neighbor_names[i], n_names[i]] for i in range(len(neighbor_names))
    ]
    property_names: List[str] = []
    for i in range(len(property_names_zip)):
        property_names.extend(property_names_zip[i])

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
                names=["o3_mu"],
                values=_dispatch.int_range_like(
                    min_val=-lambda_, max_val=lambda_ + 1, like=block_1.samples.values
                ).reshape(-1, 1),
            ),
        ],
        properties=Labels(
            names=property_names,
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
