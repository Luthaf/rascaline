"""
Private module containing helper functions for public module
:py:mod:`_correlate_density` that compute Clebsch-gordan tensor products of
:py:class:`TensorMap`.
"""

from typing import List, Optional, Tuple, Union

from .. import _dispatch
from .._backend import Array, Labels, TensorBlock, TensorMap, is_labels, operations
from . import _coefficients


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


class Combination:
    """
    Small class to store together the index of two blocks to combine and the
    set of ``o3_lambdas`` created by this combination.
    """

    def __init__(self, first: int, second: int, o3_lambdas: List[int]):
        self.first = first
        self.second = second
        self.o3_lambdas = o3_lambdas


def _compute_output_keys(
    keys_1: Labels,
    keys_2: Labels,
    o3_lambda_1_name: str,
    o3_lambda_2_name: str,
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Computes the output keys from the full CG tensor product of ``keys_1`` and
    ``keys_2``.

    The output keys are the full product of the keys in ``keys_1`` and ``keys_2``, where
    Clebsch-Gordan combination rules are used to determine the new angular order and
    parity of output blocks. Key dimensions "o3_lambda" and "o3_sigma" must therefore be
    present in both input keys.

    ``o3_lambda_1_name`` and ``o3_lambda_2_name`` are the names of output key dimensions
    which store the "o3_lambda" values of the blocks in ``keys_1`` and ``keys_2``
    respectively.

    The full tensor product of any key dimensions that are not named "o3_lambda" or
    "o3_sigma" is taken. The exception to this are those dimensions that have the same
    name in ``keys_1`` and ``keys_2``, which instead are only combined if they match in
    the values of these dimensions.

    Returned is a :py:class:`Labels` object for the output keys, and a list of the pairs
    of indices in ``keys_1`` and ``keys_2`` that combine to form each key in the output
    :py:class:`Labels`.

    :param keys_1: :py:class:`Labels`, the keys of the first tensor.
    :param keys_2: :py:class:`Labels`, the keys of the second tensor.
    :param o3_lambda_1_name: str, the name of the output key dimension storing the
        "o3_lambda" values of the blocks in ``keys_1``.
    :param o3_lambda_2_name: str, the name of the output key dimension storing the
        "o3_lambda" values of the blocks in ``keys_2``.
    """

    # Define the names and column idxs of the 'standard' (required) CG key names
    cg_names = ["o3_lambda", "o3_sigma"]
    cg_idxs_1 = [keys_1.names.index(name) for name in cg_names]
    cg_idxs_2 = [keys_2.names.index(name) for name in cg_names]

    # Define the names and column idxs of the 'match' keys, if any
    match_names = []
    for name in keys_1.names:
        if name in cg_names:
            continue
        if name in keys_2.names:
            match_names.append(name)
    match_idxs_1 = [keys_1.names.index(name) for name in match_names]
    match_idxs_2 = [keys_2.names.index(name) for name in match_names]

    # keys_1: define the names and column idxs of the 'other' keys, if any
    other_names_1 = []
    for name in keys_1.names:
        if name in cg_names or name in match_names:
            continue
        other_names_1.append(name)
    other_idxs_1 = [keys_1.names.index(name) for name in other_names_1]

    # keys_2: define the names and column idxs of the 'other' keys, if any
    other_names_2 = []
    for name in keys_2.names:
        if name in cg_names or name in match_names:
            continue
        other_names_2.append(name)
    other_idxs_2 = [keys_2.names.index(name) for name in other_names_2]

    # Build the new names
    output_names = (
        cg_names
        + match_names
        + other_names_1
        + [o3_lambda_1_name]
        + other_names_2
        + [o3_lambda_2_name]
    )

    # Build the new keys by combination
    output_key_values: List[List[int]] = []
    combinations: List[Tuple[int, int]] = []
    for key_1_i in range(len(keys_1)):
        for key_2_i in range(len(keys_2)):

            # Get the keys
            key_1 = keys_1.entry(key_1_i)
            key_2 = keys_2.entry(key_2_i)

            # If they don't match in the match dimensions, skip combination
            if not _dispatch.all(
                key_1.values[match_idxs_1] == key_2.values[match_idxs_2]
            ):
                continue

            # Unpack standard key values
            o3_lambda_1, o3_sigma_1 = int(key_1["o3_lambda"]), int(key_1["o3_sigma"])
            o3_lambda_2, o3_sigma_2 = int(key_2["o3_lambda"]), int(key_2["o3_sigma"])

            # Create all non-zero angular channel combinations
            for o3_lambda in range(
                abs(o3_lambda_1 - o3_lambda_2), abs(o3_lambda_1 + o3_lambda_2) + 1
            ):

                # Calculate new sigma
                o3_sigma = int(
                    o3_sigma_1
                    * o3_sigma_2
                    * (-1) ** (o3_lambda_1 + o3_lambda_2 + o3_lambda)
                )

                # Build the new keys values
                new_vals: List[int] = (
                    [o3_lambda, o3_sigma]  # standard CG dims
                    + _dispatch.to_int_list(key_1.values[match_idxs_1])  # match dims
                    + _dispatch.to_int_list(key_1.values[other_idxs_1])  # other dims 1
                    + [o3_lambda_1]  # lambda from block_1
                    + _dispatch.to_int_list(key_2.values[other_idxs_2])  # other dims 2
                    + [o3_lambda_2]  # lambda from block_2
                )
                output_key_values.append(new_vals)
                combinations.append((key_1_i, key_2_i))

    # Define new keys as the full product of keys_1 and keys_2
    output_keys = Labels(
        names=output_names,
        values=_dispatch.int_array_like(output_key_values, like=keys_1.values),
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


def _apply_angular_cutoff_selection(
    output_keys: Labels,
    combinations: List[Tuple[int, int]],
    angular_cutoff: int,
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Applies an angular cutoff selection to the keys of an output TensorMap
    ``output_keys`` produced by the provided ``combinations`` of blocks.

    After application of the selection, returned is a reduced set of keys and set of
    corresponding parents key entries.
    """
    # Apply the angular cutoff selection
    mask = _dispatch.bool_array_like(
        output_keys.column("o3_lambda") <= angular_cutoff, like=output_keys.values
    )

    mask_indices = _dispatch.int_array_like(
        list(range(len(combinations))), like=output_keys.values
    )[mask]

    # Apply the mask to combinations and keys
    combinations = [combinations[i] for i in mask_indices]
    output_keys = Labels(names=output_keys.names, values=output_keys.values[mask])

    return output_keys, combinations


def _remove_redundant_keys(
    output_keys: Labels,
    combinations: List[Tuple[int, int]],
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Remove redundant keys from the ``output_keys`` produced by the provided
    ``combinations`` of blocks.

    These are the keys that correspond to blocks that have the same sorted l list. Only
    the block where the l values are already sorted (i.e. l1 <= l2 <= ... <= ln) is
    kept.

    Note: this key selection is only valid if the ``output_keys`` were produced by the
    tensor product of an arbitrary body-order tensor with a body-order 2 tensor, formed
    by a density correlation, i.e. a density auto-correlation. Note also that this
    selection will not preserve the norm of the blocks.

    As this is a special case, this function assumes that the relevant keys are named
    "l_{x}", where x is an integer from 1 to the correlation order of the output tensor.
    """
    # Infer the correlation order of the output TensorMap by the length of the "l" list
    # present in the keys
    nu_target, present = 1, True
    while present:
        if f"l_{nu_target + 1}" in output_keys.names:
            nu_target += 1
        else:
            present = False

    # As this function is only valid for the output of a density auto-correlation, we
    # assert the "k" list is present in the keys
    assert all([f"k_{k}" in output_keys.names for k in range(2, nu_target)])

    # Now find the column idxs of the "l" values in the "l" list
    l_list_idxs = [output_keys.names.index(f"l_{l}") for l in range(1, nu_target + 1)]

    # Identify keys of redundant blocks and remove them
    key_idxs_to_keep: List[int] = []
    for key_idx in range(len(output_keys)):
        key = output_keys.entry(key_idx)

        # Get the list of "l" values
        l_list_values = _dispatch.to_int_list(key.values[l_list_idxs])

        # Keep this key if the l list is already sorted
        if _dispatch.all(
            _dispatch.int_array_like(l_list_values, like=key.values)
            == _dispatch.int_array_like(sorted(l_list_values), like=key.values)
        ):
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


def _increment_property_name_suffices(
    tensor: TensorMap, increment_by: Optional[int] = None
) -> TensorMap:
    """
    Uses the :py:func:`_increment_numeric_suffix` function to increment all the property
    dimension names by ``increment_by`` and returns the resulting :py:class:`TensorMap`.
    """
    for name in tensor.property_names:
        tensor = operations.rename_dimension(
            tensor,
            "properties",
            name,
            _increment_numeric_suffix(name, increment_by),
        )

    return tensor


def _increment_numeric_suffix(name: str, increment_by: Optional[int] = 1) -> str:
    """
    Takes a string name that is suffix by "_{x}", where "x" is an integer,
    and increments the integer by ``increment_by``. Returns the new name.

    If the string is not suffixed by "_{x}", then "_1" is appended to the string.

    The only special cases are names suffixed by "_type", in which case the numeric
    index is inserted and incremented before the "_type" suffix.

    Examples:
        - fn("n", 1) -> "n_1"
        - fn("n", 0) -> "n_0"
        - fn("n_1", 1) -> "n_2"
        - fn("center_type", 3) -> "center_3_type"
        - fn("first_atom_1_type") -> "first_atom_2_type"
        - fn("center_type_1_type", 7) -> "center_type_8_type"
    """
    if not isinstance(increment_by, int):
        raise ValueError("`increment_by` must be an integer.")
    if name.endswith("type"):
        # Special name case: remove the "_type" suffix and add it back on at the end
        prefix = name[: name.rfind("_type")]
        suffix = "_type"
    else:
        prefix = name
        suffix = ""

    if prefix.rfind("_") == -1:  # no suffix present
        number = 0
    else:
        number = prefix[prefix.rfind("_") + 1 :]
        if number.isdigit():  # number found
            number = int(number)
            prefix = prefix[: prefix.rfind("_")]
        else:
            number = 0

    return prefix + "_" + str(number + increment_by) + suffix


# ======================================================================= #
# ======== Functions to perform the CG tensor products of blocks ======== #
# ======================================================================= #


def cg_tensor_product_blocks(
    block_1: TensorBlock,
    block_2: TensorBlock,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> List[TensorBlock]:
    """
    For a given pair of TensorBlocks and desired angular channels, combines the values
    arrays and returns a new TensorBlock.

    If the samples of the two blocks are different, the intersection samples dimensions
    are matched and used to reshape the block with fewer dimensions such that the tensor
    product can be performed.
    """
    # Reshape the blocks along the samples axis by matching samples dimensions.
    # TODO: can any reshaping be done on the TensorMap level instead of repeated every
    # time this function is called?
    block_1, block_2 = _match_samples_of_blocks(block_1, block_2)

    # Compute the new properties dimensions for the output TensorBlocks
    properties = _compute_labels_full_cartesian_product(
        block_1.properties, block_2.properties
    )

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _coefficients.cg_tensor_product(
        block_1.values, block_2.values, o3_lambdas, cg_coefficients, cg_backend
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


def _match_samples_of_blocks(
    block_1: TensorBlock, block_2: TensorBlock
) -> Tuple[TensorBlock]:
    """
    Reshapes the samples axes of either ``block_1`` or ``block_2`` to match the samples
    of the other.

    Assumes that the samples dimensions of the block with fewer dimensions are a subset
    of the dimensions of the other. If the dimensions are not a subset, an error is
    raised.

    TODO: implement for samples dimensions that are not a subset of the other.
    """
    # If the number of dimensions are the same, check they are equivalent and return
    if len(block_1.samples.names) == len(block_2.samples.names):
        if not block_1.samples == block_2.samples:
            raise ValueError("Samples dimensions of the two blocks are not equivalent.")

    # First find the block with fewer dimensions. Reorder to have this block on the
    # 'left' for simplicity, but record the original order for the final output
    swapped = False
    if len(block_1.samples.names) > len(block_2.samples.names):
        block_1, block_2 = block_2, block_1
        swapped = True

    for name in block_1.samples.names:
        if name not in block_2.samples.names:
            raise ValueError(
                "Samples dimensions of the two blocks are not compatible."
                " The block with fewer samples dimensions must be a subset"
                f" of the other block's dimensions. Got: {block_1.samples.names}"
                f" and {block_2.samples.names}. Offending dimension: {name}"
            )

    # Broadcast the values of block_1 along the samples dimensions to match those of
    # block_2
    dims_2 = [block_2.samples.names.index(name) for name in block_1.samples.names]
    matches = _dispatch.where(
        _dispatch.all(
            block_2.samples.values[:, dims_2][:, None] == block_1.samples.values,
            axis=2,
        )
    )[1].tolist()

    # Build new block and return
    block_1 = TensorBlock(
        values=block_1.values[matches],
        samples=block_2.samples,
        components=block_1.components,
        properties=block_1.properties,
    )
    if swapped:
        return block_2, block_1

    return block_1, block_2


def _compute_labels_full_cartesian_product(
    labels_1: Labels,
    labels_2: Labels,
) -> Labels:
    """
    Computes the full cartesian product of two arbitrary :py:class:`Labels` objects.

    In contrast to the :py:func:`_precompute_keys_full_product` function, this function
    has no awareness of angular channel combination rules or value matching. It simply
    takes all combinations of the labels entries in ``labels_1`` and ``labels_2``.

    This function assumes that there are no shared labels dimension names between
    ``labels_1`` and ``labels_2``.
    """
    # Create the new labels names by concatenating the names of the two input labels
    labels_names: List[str] = labels_1.names + labels_2.names

    # create cross product list of indices in a torch-scriptable way of
    # [[i, j] for i in range(len(labels_1.values)) for j in
    #             range(len(labels_2.values))]
    # [0, 1, 2], [0, 1] -> [[0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1]]
    labels_1_labels_2_product_idx = _dispatch.cartesian_prod(
        _dispatch.int_range_like(0, len(labels_2.values), like=labels_2.values),
        _dispatch.int_range_like(0, len(labels_1.values), like=labels_1.values),
    )
    return Labels(
        names=labels_names,
        values=_dispatch.int_array_like(
            [
                _dispatch.to_int_list(labels_2.values[indices[0]])
                + _dispatch.to_int_list(labels_1.values[indices[1]])
                for indices in labels_1_labels_2_product_idx
            ],
            like=labels_1.values,
        ),
    )
