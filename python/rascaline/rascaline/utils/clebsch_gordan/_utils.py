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
    match_keys: Optional[List[str]],
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

    Any key dimensions named in ``match_keys`` are matched between the two input keys. A
    CG tensor product is only computed for pairs of blocks that match in this key
    dimension. As such, this named dimension must be present in both ``keys_1`` and
    ``keys_2``.

    For any key dimensions present in ``keys_1`` or ``keys_2`` that are not either a)
    the standard CG keys: "o3_lambda" and "o3_sigma", nor b) the CG combination keys,
    i.e. "l" and "k" lists, nor c) named in ``match_keys``, the full tensor product of
    these dimensions are computed. These 'other' dimensions must therefore not have the
    same names in ``keys_1`` and ``keys_2``.

    If ``skip_redundant`` is True, then keys that represent redundant CG operations are
    not included in the output keys at each step. This is only applicable for use in the
    :py:class:`DensityCorrelations` calculator.
    """

    # Get the keys metadata for the combination of the 2 tensors
    output_keys, combinations = _precompute_keys_full_product(
        keys_1, keys_2, match_keys
    )

    if selected_keys is not None:
        output_keys, combinations = _apply_key_selection(
            output_keys,
            combinations,
            selected_keys,
        )

    if skip_redundant:
        output_keys, combinations = _remove_redundant_keys(output_keys, combinations)

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
    match_keys: Optional[List[str]],
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Given the keys of 2 TensorMaps, returns the keys that would be present after a full
    CG product of these TensorMaps.

    This function assumes that ``keys_1`` corresponds to a TensorMap with arbitrary body
    order, while `keys_2` corresponds to a TensorMap with body order 1.

    ``keys_1``  must follow the key name convention: ["o3_lambda", "o3_sigma", "l1",
    "l2", ..., "l<nu>", "k2", ..., "k<nu-1>", *other_keys].

    ``keys_2`` must follow the key name convention: ``["o3_lambda", "o3_sigma",
    *other_keys]``.

    In both ``keys_1`` and ``keys_2``, the first 2 dimensions of the keys correspond to
    the standard keys, i.e. "o3_lambda" and "o3_sigma".

    In ``keys_1``, the "l" and "k" lists, respectively, follow. The "lx" columns track
    the l values of the nu=1 blocks that were previously combined. The "kx" columns
    tracks the intermediate lambda values of nu > 1 blocks that have been combined.

    For both ``keys_1`` and ``keys_2``, the 'other' dimensions remain. In general, the
    full tensor product over these dimensions is takemn, and the names of these
    dimensions must not be shared between ``keys_1`` and ``keys_2``. The exception are
    any dimensions named in the input variable ``match_keys``. These dimensions must
    share a name in ``keys_1`` and ``keys_2``, and only the product of pairs of keys
    with matching values in these dimension are taken.

    For instance, a TensorMap of body order ``nu=4`` will have key names ``["o3_lambda",
    "o3_sigma", "l1", "l2", "l3", "l4", "k2", "k3", *match_keys, *other_keys]``. Two
    ``nu=1`` TensorMaps with blocks of order "l1" and "l2" were combined to form a nu=2
    TensorMap with blocks of order "k2". This was combined with a nu=1 TensorMap with
    blocks of order "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally,
    this was combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4. If
    ``match_keys=["center_type"]``, for instance, only products of keys with the same
    center type will be taken.

    .. math ::

        \\bra{
            n_1 lambda_1 ; n_2 lambda_2 k_2 ; ... ; n_{\nu-1} l_{\\nu-1} k_{\\nu-1} ;
            n_{\\nu} l_{\\nu} k_{\\nu}; \\lambda
        }
        \\ket{ \\rho^{\\otimes \\nu}; \\lambda M }

    The first object returned from this function is a Labels object corresponding to the
    keys of the output TensorMap that would be formed from this CG tensor product. The
    second object is a :py:class:`Combination` that stores the indices of the keys in
    ``keys_1`` and ``keys_2`` that were combined to form each key in the output
    TensorMap.
    """
    # Infer the correlation order of the first tensor by the length of the "l" list
    # present in the ``keys_1``
    nu_1, present = 1, True
    while present:
        if f"l_{nu_1 + 1}" in keys_1.names:
            nu_1 += 1
        else:
            present = False
    assert all([f"k_{k}" in keys_1.names for k in range(2, nu_1)])

    # Check there are no "l" or "k" lists in the second TensorMap
    for name in keys_2.names:
        if name.startswith("l_") or name.startswith("k_"):
            raise ValueError(
                "The second TensorMap must be of correlation order nu = 1,"
                " so must not have any angular channel combination keys, "
                " i.e. 'l_{x}' or 'k_{x}' columns."
            )

    nu = nu_1 + 1  # `nu` is the new correlation order of the output tensor

    # Categorise key names into:
    #     1) standard keys: ["o3_lambda", "o3_sigma"]
    #     2) cg combination keys (i.e. "l" and "k" lists):
    #            ["lambda_1", "lambda_2", ..., "l_{nu-1}"]
    #            and ["k_2", ..., "k_{nu-1}"].
    #     3) match keys: those in variable ``match_keys``
    #     4) other keys: all other key dimensions to take the full product of

    # 1) Standard keys
    standard_keys: List[str] = ["o3_lambda", "o3_sigma"]

    # 2) "l" list
    if nu_1 == 1:  # no "l_{x}" columns yet
        cg_combine_keys_l: List[str] = []
        new_cg_combine_keys_l = ["l_1", "l_2"]
    else:
        cg_combine_keys_l = [f"l_{angular_l}" for angular_l in range(1, nu_1 + 1)]
        new_cg_combine_keys_l = cg_combine_keys_l + [f"l_{nu}"]

    # 2) "k" list
    cg_combine_keys_k: List[str] = [f"k_{k}" for k in range(2, nu_1)]
    new_cg_combine_keys_k: List[str] = [f"k_{k}" for k in range(2, nu)]

    # Check names of keys_1: the standard keys and cg combination lists should be
    # present as the first key dimensions, in order. The rest are match/other keys.
    tmp_expected_1: List[str] = standard_keys + cg_combine_keys_l + cg_combine_keys_k
    if keys_1.names[: len(tmp_expected_1)] != tmp_expected_1:
        raise ValueError(
            f"keys_1 names do not match the expected format for the standard"
            f" and combination keys. Got {keys_1.names[: len(tmp_expected_1)]},"
            f" expecting {tmp_expected_1}"
        )
    other_keys_1: List[str] = keys_1.names[len(tmp_expected_1) :]

    # Check names of keys_2: only the standard keys should be present as the first key
    # dimensions. The rest are match/other keys.
    tmp_expected_2: List[str] = standard_keys
    if keys_2.names[: len(tmp_expected_2)] != tmp_expected_2:
        raise ValueError(
            f"keys_2 names do not match the expected format for the standard keys."
            f" Got {keys_2.names[: len(tmp_expected_2)]}, expecting {tmp_expected_2}"
        )
    other_keys_2: List[str] = keys_2.names[len(tmp_expected_2) :]

    # Check each of the match keys and pop them from the other keys lists, as we don't
    # want to take full products of these dimensions.
    if match_keys is None:
        match_keys = []
    for match_key in match_keys:
        if match_key not in keys_1.names:
            raise ValueError(
                f"match key {match_key} not found in keys_1: {keys_1.names}"
            )
        if match_key not in keys_2.names:
            raise ValueError(
                f"match key {match_key} not found in keys_2: {keys_2.names}"
            )
        assert match_key in other_keys_1 and match_key in other_keys_2
        other_keys_1.remove(match_key)
        other_keys_2.remove(match_key)

    # 4) Other keys. Check the intersection between keys_1 and keys_2 is zero
    for other_key in other_keys_1:
        if other_key in other_keys_2:
            raise ValueError(
                "The other key names must not be shared by keys_1 and keys_2."
                f" Other keys found in ``keys_1``: {other_keys_1} and in"
                f" ``keys_2``: {other_keys_2}. Please rename these dimensions"
                " or pass them in ``match_keys``."
            )
    new_other_keys: List[str] = other_keys_1 + other_keys_2

    # Now create the new key names (i.e. for the combined TensorMap)...
    new_names: List[str] = (
        standard_keys
        + new_cg_combine_keys_l
        + new_cg_combine_keys_k
        + match_keys
        + new_other_keys
    )

    # ... and build the values of the new keys by taking the full product, accounting
    # for angular channel combination rules and matched keys.
    new_key_values: List[List[int]] = []
    combinations: List[Tuple[int, int]] = []
    for key_1_i in range(len(keys_1)):
        for key_2_i in range(len(keys_2)):

            key_1 = keys_1.entry(key_1_i)
            key_2 = keys_2.entry(key_2_i)

            # Check the values of the match keys (if any). If they don't match, skip.
            matching = True
            match_key_values: List[int] = []
            for match_key in match_keys:
                match_key_value = key_1[keys_1.names.index(match_key)]
                if match_key_value != key_2[keys_2.names.index(match_key)]:
                    matching = False
                    break
                else:
                    match_key_values.append(match_key_value)
            if not matching:
                continue

            # Unpack standard key values
            lambda_1, sigma_1 = int(key_1["o3_lambda"]), int(key_1["o3_sigma"])
            lambda_2, sigma_2 = int(key_2["o3_lambda"]), int(key_2["o3_sigma"])

            # Extract the l and k lists from keys_1
            l_list = _dispatch.to_int_list(
                keys_1.values[
                    key_1_i,
                    len(standard_keys) : len(standard_keys) + len(cg_combine_keys_l),
                ]
            )
            k_list = _dispatch.to_int_list(
                keys_1.values[
                    key_1_i,
                    len(standard_keys)
                    + len(cg_combine_keys_l) : len(standard_keys)
                    + len(cg_combine_keys_l)
                    + len(cg_combine_keys_k),
                ]
            )

            # Create the list of values for the other keys
            new_other_vals: List[int] = []
            for new_other_key in new_other_keys:
                if new_other_key in keys_1.names:
                    new_other_vals.append(int(key_1[new_other_key]))
                else:
                    assert new_other_key in keys_2.names
                    new_other_vals.append(int(key_2[new_other_key]))

            # Now iterate over the non-zero angular channels and apply the custom
            # selections. The non-zero channels are given by the inclusive range
            # { |lambda_1 - lambda_2|, ..., |lambda_1 + lambda_2| }
            for o3_lambda in range(
                abs(lambda_1 - lambda_2), abs(lambda_1 + lambda_2) + 1
            ):

                # Calculate new sigma
                o3_sigma = int(
                    sigma_1 * sigma_2 * (-1) ** (lambda_1 + lambda_2 + o3_lambda)
                )

                # Build the new keys values. l_{nu} is `lambda_2`` (i.e.
                # "o3_lambda" of the key from `keys_2`. k_{nu-1} is
                # `lambda_1` (i.e. "o3_lambda" of the key from `keys_1`).
                new_vals: List[int] = (
                    [o3_lambda, o3_sigma]
                    + l_list
                    + [lambda_2]
                    + k_list
                    + [lambda_1]
                    + match_key_values
                    + new_other_vals
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
) -> Tuple[Labels, List[Tuple[int, int]]]:
    """
    Remove redundant keys from the ``output_keys`` produced by the provided
    ``combinations`` of blocks.

    These are the keys that correspond to blocks that have the same sorted l list. The
    block where the l values are already sorted (i.e. l1 <= l2 <= ... <= ln) is kept.
    """
    # Infer the correlation order of the output TensorMap by the length of the "l" list
    # present in the keys
    nu_target, present = 1, True
    while present:
        if f"l_{nu_target + 1}" in output_keys.names:
            nu_target += 1
        else:
            present = False
    assert all([f"k_{k}" in output_keys.names for k in range(2, nu_target)])

    # Identify keys of redundant blocks and remove them
    key_idxs_to_keep: List[int] = []
    standard_names: List[str] = ["o3_lambda", "o3_sigma"]
    for key_idx in range(len(output_keys)):
        key = output_keys.entry(key_idx)
        # Get the list of "l" values and sort
        standard_vals = _dispatch.to_int_list(key.values[: len(standard_names)])
        l_list = _dispatch.to_int_list(
            key.values[len(standard_names) : len(standard_names) + nu_target]
        )
        l_list_sorted = sorted(l_list)

        # Compare the sliced key with the one recreated when the l list is
        # sorted. If they are identical, this is the key of the block that we
        # want to compute a CG combination for.
        key_slice_tuple = _dispatch.int_array_like(
            standard_vals + l_list, like=key.values
        )
        key_slice_sorted_tuple = _dispatch.int_array_like(
            standard_vals + l_list_sorted, like=key.values
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


def _increment_numeric_suffix(name: str) -> str:
    """
    Takes a string name that is suffix by "_{x}", where "x" is a non-negative integer,
    and increments the integer by 1. Returns the new name.

    If the string is not suffixed by "_{x}", then "_1" is appended to the string.

    The only special cases are names suffixed by "_type", in which case the numeric
    index is inserted and incremented before the "_type" suffix.

    Examples:
        - "n" -> "n_1"
        - "n_1" -> "n_2"
        - "center_type" -> "center_1_type"
        - "first_atom_1_type" -> "first_atom_2_type"
        - "center_type_1_type" -> "center_type_2_type"
    """
    add_type_suffix = False
    if name.endswith("type"):
        # Special name case: remove the "_type" suffix and add it back on at the end
        name = name[: name.rfind("_type")]
        add_type_suffix = True

    if name.rfind("_") == -1:  # no "_" thus no suffix: add "_1"
        new_name = name + "_1"
        if add_type_suffix:
            new_name += "_type"
        return new_name

    number = name[name.rfind("_") + 1 :]  # attempt to extract the numeric suffix
    if number.isdigit():  # increment the number if found
        name = name[: name.rfind("_")]
        number = str(int(number) + 1)
    else:  # no numeric suffix, add one
        number = "1"

    new_name = name + "_" + number
    if add_type_suffix:
        new_name += "_type"
    return new_name


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
