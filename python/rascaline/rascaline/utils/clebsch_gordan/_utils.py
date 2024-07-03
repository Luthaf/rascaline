"""
Private module containing helper functions for public module
:py:mod:`_correlate_density` that compute Clebsch-gordan tensor products of
:py:class:`TensorMap`.
"""

from typing import List, Optional, Tuple, Union

from .. import _dispatch
from .._backend import Array, Labels, TensorBlock, TensorMap, is_labels
from . import _coefficients


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


def standardize_keys(tensor: TensorMap) -> TensorMap:
    """
    Takes a ``nu=1`` tensor and standardizes its metadata. This involves: (1) moving the
    ``"neighbor_type"`` key to properties, if present as a dimension in the keys, and
    (2) adding dimensions in the keys for tracking the body order (``"order_nu"``).

    Checking for the presence of the ``"neighbor_type"`` key in the keys allows the
    option of the user pre-moving this key to the properties before calling us, allowing
    sparsity in a set of global neighbors to be created if desired.

    Assumes that the input ``tensor`` is nu=1.
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


def parse_selected_keys(
    n_iterations: int,
    array_like: Array,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
) -> List[Labels]:
    """
    Parses the ``selected_keys`` argument passed to public functions. Checks the values
    and returns a list of :py:class:`Labels` objects, one for each iteration of CG
    iteration. ``array_like`` determines the array backend of the created Labels.
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
            "`selected_keys` must be a List[Labels] of length" " `body_order` - 2"
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


def parse_bool_iteration_filter(
    n_iterations: int, bool_filter: Union[bool, List[bool]], filter_name: str
) -> List[bool]:
    """
    Parses the ``bool_filter`` argument passed to public functions.
    """
    if bool_filter is None:
        bool_filter = [False] * (n_iterations - 1) + [True]
    if isinstance(bool_filter, bool):
        bool_filter_ = [bool_filter] * n_iterations
    else:
        bool_filter_ = bool_filter

    if not all([isinstance(val, bool) for val in bool_filter_]):
        raise TypeError(f"`{filter_name}` must be a `bool` or `list` of `bool`")
    if not len(bool_filter_) == n_iterations:
        raise ValueError(
            f"`{filter_name}` must be a bool or `list` of `bool` of length"
            " corresponding to the number of iterations, i.e. `body_order` - 2"
        )

    return bool_filter_


class Combination:
    """
    Small class to store together the index of two blocks to combine and the
    set of ``o3_lambdas`` created by this combination.
    """

    def __init__(self, first: int, second: int, o3_lambdas: List[int]):
        self.first = first
        self.second = second
        self.o3_lambdas = o3_lambdas


def precompute_keys(
    keys_1: Labels,
    keys_2: Labels,
    selected_keys: Optional[Labels],
    skip_redundant: bool,
) -> Tuple[Labels, List[Combination]]:
    """
    Pre-compute the output keys after a single CG iteration combining ``keys_1`` and
    ``keys_2``.

    This function returns the computed keys, and a list of tuple of integers, indicating
    for each entries in the output keys which entry of the two set of input keys should
    be combined together.

    The output keys are generated from a full product of ``keys_1`` and ``keys_2``.
    Then, key are selected according to the angular and/or parity selections specified
    in ``selected_keys``.

    If ``skip_redundant`` is True, then keys that represent redundant CG operations are
    not included in the output keys at each step.
    """

    # Get the keys metadata for the combination of the 2 tensors
    output_keys, combinations = _precompute_keys_full_product(keys_1, keys_2)

    if selected_keys is not None:
        output_keys, combinations = _apply_key_selection(
            output_keys,
            combinations,
            selected_keys,
        )

    if skip_redundant:
        output_keys, combinations = _remove_redundant_keys(
            output_keys,
            combinations,
            nu_1=keys_1.column("order_nu")[0],
            nu_2=keys_2.column("order_nu")[0],
        )

    output_keys, combinations = _group_combinations_of_same_blocks(
        output_keys, combinations
    )

    return output_keys, combinations


def _group_combinations_of_same_blocks(
    output_keys: Labels,
    combinations: List[Tuple[int, int]],
) -> Tuple[Labels, List[Combination]]:
    """
    Re-order output keys in such a way that we have all the different ``o3_lambda``
    created by the same pair of block consecutive in ``output_keys``.

    This function simultaneously returns the new ``output_keys``, and a list of
    combinations in the same order as the keys. Iterating over the returned combinations
    and then over the ``o3_lambdas`` for each combination will produce blocks in the
    same order as the output keys.
    """
    # (block_1, block_2) => list of index in the output keys
    # this is emulating a dict by separating keys & values since TorchScript does not
    # support Tuple[int, int] as Dict keys. `groups` is the dict keys and `groups_keys`
    # the dict values.
    groups: List[Tuple[int, int]] = []
    groups_keys: List[List[int]] = []
    for key_i, blocks in enumerate(combinations):
        group_id = -1
        for existing_id, existing in enumerate(groups):
            if existing == blocks:
                group_id = existing_id
                break

        if group_id == -1:
            group_id = len(groups)
            groups.append(blocks)
            keys_idx: List[int] = []
            groups_keys.append(keys_idx)

        groups_keys[group_id].append(key_i)

    all_o3_lambdas = output_keys.column("o3_lambda")

    keys_values: List[List[int]] = []
    combinations: List[Combination] = []
    for (block_1, block_2), keys_idx in zip(groups, groups_keys):
        o3_lambdas: List[int] = []
        for key_i in keys_idx:
            keys_values.append(_dispatch.to_int_list(output_keys.values[key_i]))
            o3_lambdas.append(int(all_o3_lambdas[key_i]))

        combinations.append(Combination(block_1, block_2, o3_lambdas))

    output_keys = Labels(
        output_keys.names,
        _dispatch.int_array_like(keys_values, like=output_keys.values),
    )

    return output_keys, combinations


def _precompute_keys_full_product(
    keys_1: Labels,
    keys_2: Labels,
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Given the keys of 2 TensorMaps, returns the keys that would be present after a full
    CG product of these TensorMaps.

    Assumes that ``keys_1`` corresponds to a TensorMap with arbitrary body order, while
    `keys_2` corresponds to a TensorMap with body order 1. ``keys_1``  must follow the
    key name convention:

    ["order_nu", "o3_sigma", "o3_lambda", "center_type", "l1", "l2", ..., "l<nu>", "k2",
    ..., "k<nu-1>"]. The "lx" columns track the l values of the nu=1 blocks that were
    previously combined. The "kx" columns tracks the intermediate lambda values of nu >
    1 blocks that have been combined.

    For instance, a TensorMap of body order ``nu=4`` will have key names ``["order_nu",
    "o3_sigma", "o3_lambda", "center_type", "l1", "l2", "l3", "l4", "k2", "k3"]``. Two
    ``nu=1`` TensorMaps with blocks of order "l1" and "l2" were combined to form a nu=2
    TensorMap with blocks of order "k2". This was combined with a nu=1 TensorMap with
    blocks of order "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally,
    this was combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

    .. math ::

        \\bra{
            n_1 l_1 ; n_2 l_2 k_2 ; ... ; n_{\nu-1} l_{\\nu-1} k_{\\nu-1} ; n_{\\nu}
            l_{\\nu} k_{\\nu}; \\lambda
        } \\ket{ \\rho^{\\otimes \\nu}; \\lambda M }

    ``keys_2`` must follow the key name convention: ``["order_nu", "o3_sigma",
    "o3_lambda", "center_type"]``

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
    combinations: List[Tuple[int, int]] = []
    for key_1_i in range(len(keys_1)):
        for key_2_i in range(len(keys_2)):
            key_1 = keys_1.entry(key_1_i)
            key_2 = keys_2.entry(key_2_i)
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
                l_list = _dispatch.to_int_list(keys_1.values[key_1_i, 4 : 4 + nu_1])
                k_list = _dispatch.to_int_list(keys_1.values[key_1_i, 4 + nu_1 :])

                # Build the new keys values. l{nu} is `lambda_2`` (i.e.
                # "o3_lambda" of the key from `keys_2`. k{nu-1} is
                # `lambda_1` (i.e. "o3_lambda" of the key from `keys_1`).
                new_vals: List[int] = (
                    [nu, o3_lambda, o3_sigma, type_1] + l_list + [l_2] + k_list + [l_1]
                )
                new_key_values.append(new_vals)
                combinations.append((key_1_i, key_2_i))

    # Define new keys as the full product of keys_1 and keys_2
    output_keys = Labels(
        names=new_names,
        values=_dispatch.int_array_like(new_key_values, like=keys_1.values),
    )

    return output_keys, combinations


def _apply_key_selection(
    output_keys: Labels,
    combinations: List[Tuple[int, int]],
    selected_keys: Labels,
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Applies a selection according to ``selected_keys`` to the keys of an output
    TensorMap ``output_keys`` produced by the provided ``combinations`` of blocks.

    After application of the selections, returned is a reduced set of keys and set of
    corresponding parents key entries.

    If a selection in ``selected_keys`` is not valid based on the keys in
    ``output_keys``, we raise an error.
    """
    # Extract the relevant columns from `selected_keys` that the selection will be
    # performed on
    col_idx = _dispatch.int_array_like(
        [output_keys.names.index(name) for name in selected_keys.names],
        output_keys.values,
    )
    output_keys_values = output_keys.values[:, col_idx]

    # First check that all of the selected keys exist in the output keys
    for selected in selected_keys.values:
        if not any(
            [
                bool(all(selected == output_keys_values[i]))
                for i in range(len(output_keys_values))
            ]
        ):
            raise ValueError(
                f"selected key {selected_keys.names} = {selected} not found "
                "in the output keys"
            )

    # Build a mask of the selected keys
    mask = _dispatch.bool_array_like(
        [
            any([bool(all(i == j)) for j in selected_keys.values])
            for i in output_keys_values
        ],
        like=selected_keys.values,
    )

    mask_indices = _dispatch.int_array_like(
        list(range(len(combinations))), like=selected_keys.values
    )[mask]
    # Apply the mask to combinations and keys
    combinations = [combinations[i] for i in mask_indices]
    output_keys = Labels(names=output_keys.names, values=output_keys.values[mask])

    return output_keys, combinations


def _remove_redundant_keys(
    output_keys: Labels,
    combinations: List[Tuple[int, int]],
    nu_1: int,
    nu_2: int,
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Remove redundant keys from the ``output_keys`` produced by the provided
    ``combinations`` of blocks.

    These are the keys that correspond to blocks that have the same sorted l list. The
    block where the l values are already sorted (i.e. l1 <= l2 <= ... <= ln) is kept.
    """
    # Get and check the correlation order of the input keys
    assert nu_2 == 1

    # Get the correlation order of the output TensorMap
    nu = nu_1 + 1

    # Identify keys of redundant blocks and remove them
    key_idxs_to_keep: List[int] = []
    for key_idx in range(len(output_keys)):
        key = output_keys.entry(key_idx)
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
    output_keys = Labels(
        names=output_keys.names,
        values=output_keys.values[
            _dispatch.int_array_like(key_idxs_to_keep, like=output_keys.values)
        ],
    )

    # Store the list of reduced entries that combine to form the reduced output keys
    combinations = [combinations[i] for i in key_idxs_to_keep]

    return output_keys, combinations


# ======================================================================= #
# ======== Functions to perform the CG tensor products of blocks ======== #
# ======================================================================= #


def cg_tensor_product_blocks_same_samples(
    block_1: TensorBlock,
    block_2: TensorBlock,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> List[TensorBlock]:
    """
    For a given pair of TensorBlocks and desired angular channels, combines the values
    arrays and returns a new TensorBlock.
    """

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _coefficients.cg_tensor_product(
        block_1.values, block_2.values, o3_lambdas, cg_coefficients, cg_backend
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
    properties = Labels(
        names=property_names,
        values=_dispatch.int_array_like(
            [
                _dispatch.to_int_list(block_2.properties.values[indices[0]])
                + _dispatch.to_int_list(block_1.properties.values[indices[1]])
                for indices in block_1_block_2_product_idx
            ],
            block_1.samples.values,
        ),
    )

    # Create the TensorBlocks
    results: List[TensorBlock] = []
    for values, o3_lambda in zip(combined_values, o3_lambdas):
        block = TensorBlock(
            values=values,
            samples=block_1.samples,
            components=[
                Labels(
                    names=["o3_mu"],
                    values=_dispatch.int_range_like(
                        min_val=-o3_lambda,
                        max_val=o3_lambda + 1,
                        like=block_1.samples.values,
                    ).reshape(-1, 1),
                ),
            ],
            properties=properties,
        )
        results.append(block)

    return results
