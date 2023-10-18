"""
Module for computing Clebsch-gordan iterations with metatensor TensorMaps.
"""
import itertools
from typing import List, Optional, Tuple, Union

import metatensor
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

from .cg_coefficients import ClebschGordanReal


# ======================================================================
# ===== Functions to do CG combinations on single-center descriptors
# ======================================================================


def lambda_soap_vector(
    nu_1_tensor: TensorMap,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int]]] = None,
    parity_selection: Optional[Union[None, int, List[int]]] = None,
) -> TensorMap:
    """
    A higher-level wrapper for the :py:func:`n_body_iteration_single_center`
    function specifically for generating a lambda-SOAP vector in the metatensor
    format.

    A nu = 1 (i.e. 2-body) single-center descriptor is taken as input and
    conbined with itself in a single CG combination step to form a nu = 2
    (3-body) single-center descriptor, i.e. lambda-SOAP. Only the target angular
    channels given in `angular_selection` and target parities given in
    `parity_selection` are returned.

    The input `nu_1_tensor` may be, for instance, a rascaline.SphericalExpansion
    or rascaline.LodeSphericalExpansion.

    This function differs from :py:func`combine_single_center_to_body_order` in
    that the returned TensorMap has the redundant "order_nu" key dimension
    removed (which is by definition 2 for lambda-SOAP), and if a single parity
    is selected, the redundant "inversion_sigma" key dimension too.
    """
    if np.any([len(list(block.gradients())) > 0 for block in nu_1_tensor]):
        raise NotImplementedError(
            "CG combinations of gradients not currently supported. Check back soon."
        )
    # Generate lambda-SOAP
    lsoap = combine_single_center_to_body_order(
        nu_1_tensor=nu_1_tensor,
        target_body_order=2,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
        use_sparse=True,
    )

    # Drop the redundant key name "order_nu". This is by definition 2 for all
    # lambda-SOAP blocks.
    keys = lsoap.keys.remove(name="order_nu")
    lsoap = TensorMap(keys=keys, blocks=[b.copy() for b in lsoap.blocks()])

    # If a single parity is requested, drop the now redundant "inversion_sigma"
    # key name
    if len(np.unique(lsoap.keys.column("inversion_sigma"))) == 1:
        keys = lsoap.keys.remove(name="inversion_sigma")
        lsoap = TensorMap(keys=keys, blocks=[b.copy() for b in lsoap.blocks()])

    return lsoap


def combine_single_center_to_body_order(
    nu_1_tensor: TensorMap,
    target_body_order: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    use_sparse: bool = True,
) -> TensorMap:
    """
    Takes a nu = 1 (i.e. 2-body) single-center descriptor and combines it
    iteratively with itself to generate a descriptor of order
    ``target_body_order``.
    """
    if target_body_order < 1:
        raise ValueError("`target_body_order` must be > 0")

    if np.any([len(list(block.gradients())) > 0 for block in nu_1_tensor]):
        raise NotImplementedError(
            "CG combinations of gradients not currently supported. Check back soon."
        )

    # Standardize the metadata of the input tensor
    nu_1_tensor = _standardize_tensor_metadata(nu_1_tensor)

    # If the desired body order is 1, return the input tensor with standardized
    # metadata.
    if target_body_order == 1:
        return nu_1_tensor

    # Pre-compute the metadata needed to perform each CG iteration
    # Current design choice: only combine a nu = 1 tensor iteratively with
    # itself, i.e. nu=1 + nu=1 --> nu=2. nu=2 + nu=1 --> nu=3, etc.
    parity_selection = _parse_selection_filters(
        n_iterations=target_body_order - 1,
        selection_type="parity",
        selection=parity_selection,
    )
    angular_selection = _parse_selection_filters(
        n_iterations=target_body_order - 1,
        selection_type="angular",
        selection=angular_selection,
        angular_cutoff=angular_cutoff,
    )
    combination_metadata = _precompute_metadata(
        nu_1_tensor.keys,
        nu_1_tensor.keys,
        n_iterations=target_body_order - 1,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    # Define the cached CG coefficients, either as sparse dicts or dense arrays.
    # TODO: we pre-computed the combination metadata, so a more cleverly
    # constructed CG cache could be used to reduce memory overhead - i.e. we
    # don't necessarily need *all* CG coeffs up to `angular_max`, just the ones
    # that are actually used.
    angular_max = np.max(
        np.concatenate(
            [nu_1_tensor.keys.column("spherical_harmonics_l")]
            + [
                metadata[0].column("spherical_harmonics_l")
                for metadata in combination_metadata
            ]
        )
    )
    cg_cache = ClebschGordanReal(angular_max, use_sparse)

    # Create a copy of the nu = 1 tensor to combine with itself
    nu_x_tensor = nu_1_tensor.copy()

    # Iteratively combine block values
    for iteration in range(target_body_order - 1):
        # Combine blocks
        nu_x_blocks = []
        # TODO: is there a faster way of iterating over keys/blocks here?
        for nu_x_key, key_1, key_2, multi in zip(*combination_metadata[iteration]):
            # Combine the pair of block values, accounting for multiplicity
            nu_x_block = _combine_single_center_blocks(
                nu_x_tensor[key_1],
                nu_1_tensor[key_2],
                nu_x_key["spherical_harmonics_l"],
                cg_cache,
                correction_factor=np.sqrt(multi),
            )
            nu_x_blocks.append(nu_x_block)
        nu_x_keys = combination_metadata[iteration][0]
        nu_x_tensor = TensorMap(keys=nu_x_keys, blocks=nu_x_blocks)

    # Apply normalization factor to each block based on their permutational
    # multiplicity (i.e. how many ways in which they could have been formed)
    nu_x_tensor = _normalize_blocks(nu_x_tensor, target_body_order)

    # Move the [l1, l2, ...] keys to the properties
    if target_body_order > 1:
        nu_x_tensor = nu_x_tensor.keys_to_properties(
            [f"l{i}" for i in range(1, target_body_order + 1)]
            + [f"k{i}" for i in range(2, target_body_order)]
        )

    return nu_x_tensor


def _normalize_blocks(tensor: TensorMap, target_body_order: int) -> TensorMap:
    """
    For each block in `tensor`, uses values of the keys "l1", "l2", ..., "lx" to
    calculate the permutations P that correspond to the ways in which the block
    could have been formed from CG combinations. The normalization factor is
    then defined as (1 / sqrt(P)).

    This function assumes that the "lx" keys have *not* yet been moved to the
    properties, and is intended to be performed after all CG iterations have
    been performed.
    """
    l_names = [f"l{i}" for i in range(1, target_body_order + 1)]
    for key, block in tensor.items():
        l_vals = [key[l_name] for l_name in l_names]
        perm_set = set()
        for perm in itertools.permutations(l_vals):
            perm_set.add(perm)

        norm_factor = np.sqrt(len(perm_set))
        block.values[:] *= norm_factor

    return tensor


def combine_single_center_to_body_order_metadata_only(
    nu_1_tensor: TensorMap,
    target_body_order: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
) -> List[TensorMap]:
    """
    Performs a pseudo-CG combination of a nu = 1 (i.e. 2-body) single-center
    descriptor with itself to generate a descriptor of order
    ``target_body_order``.

    A list of TensorMaps is returned, where each has the complete * metadata *
    of the TensorMap that would be created by a full CG combination. No actual
    CG combinations of block values arrays are performed, instead arrays of
    zeros are returned in the output TensorMaps.

    This function is useful for producing pseudo-outputs of a CG iteration
    calculation with all the correct metadata, but far cheaper than if CG
    combinations were actually performed. This can help to quantify the size of
    descriptors produced, and observe the effect of selection filters on the
    expansion of features at each iteration.
    """
    if target_body_order <= 1:
        raise ValueError("`target_body_order` must be > 1")

    if np.any([len(list(block.gradients())) > 0 for block in nu_1_tensor]):
        raise NotImplementedError(
            "CG combinations of gradients not currently supported. Check back soon."
        )

    # Standardize the metadata of the input tensor
    nu_1_tensor = _standardize_tensor_metadata(nu_1_tensor)

    # Pre-compute the metadata needed to perform each CG iteration
    # Current design choice: only combine a nu = 1 tensor iteratively with
    # itself, i.e. nu=1 + nu=1 --> nu=2. nu=2 + nu=1 --> nu=3, etc.
    parity_selection = _parse_selection_filters(
        n_iterations=target_body_order - 1,
        selection_type="parity",
        selection=parity_selection,
    )
    angular_selection = _parse_selection_filters(
        n_iterations=target_body_order - 1,
        selection_type="angular",
        selection=angular_selection,
        angular_cutoff=angular_cutoff,
    )
    combination_metadata = _precompute_metadata(
        nu_1_tensor.keys,
        nu_1_tensor.keys,
        n_iterations=target_body_order - 1,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )

    # Create a copy of the nu = 1 tensor to combine with itself
    nu_x_tensor = nu_1_tensor

    # Iteratively combine block values
    nu_x_tensors = []
    for iteration in range(target_body_order - 1):
        # Combine blocks
        nu_x_blocks = []
        # TODO: is there a faster way of iterating over keys/blocks here?
        # TODO: only account for multiplicity at the end
        for nu_x_key, key_1, key_2, _ in zip(*combination_metadata[iteration]):
            nu_x_block = _combine_single_center_blocks(
                nu_x_tensor[key_1],
                nu_1_tensor[key_2],
                nu_x_key["spherical_harmonics_l"],
                cg_cache=None,
                correction_factor=1.0,
                return_metadata_only=True,
            )
            nu_x_blocks.append(nu_x_block)
        nu_x_keys = combination_metadata[iteration][0]
        nu_x_tensor = TensorMap(keys=nu_x_keys, blocks=nu_x_blocks)
        nu_x_tensors.append(nu_x_tensor)

    return [
        tensor.keys_to_properties(
            [f"l{i}" for i in range(1, tmp_bo + 1)]
            + [f"k{i}" for i in range(2, tmp_bo)]
        )
        for tmp_bo, tensor in enumerate(nu_x_tensors, start=2)
    ]


def combine_single_center_one_iteration(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    use_sparse: bool = True,
) -> TensorMap:
    """
    Takes two single-center descriptors of arbitrary body order and combines
    them in a single CG combination step.
    """
    # TODO: implement!
    raise NotImplementedError


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


def _standardize_tensor_metadata(tensor: TensorMap) -> TensorMap:
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
    keys = tensor.keys.insert(name="order_nu", values=np.array([1]), index=0)
    keys = keys.insert(name="inversion_sigma", values=np.array([1]), index=1)
    return TensorMap(keys=keys, blocks=[b.copy() for b in tensor.blocks()])


def _parse_selection_filters(
    n_iterations: int,
    selection_type: str = "parity",
    selection: Union[None, int, List[int], List[List[int]]] = None,
    angular_cutoff: Optional[int] = None,
) -> List[Union[None, List[int]]]:
    """
    Returns a list of length `n_iterations` with selection filters for each CG
    combination step, for either `selection_type` "parity" or `selection_type`
    "angular". For a given iteration, if no selection is to be applied, the
    element of the returned list will be None.

    The input argument `selection` will be parsed in the following ways.

    If `selection=None` is passed, then no filter is applied at any iteration. A
    list of [None, None, ...] is returned.

    If an `int` or single List[int] is specified, then this is used for the last
    iteration only. For example, if `n_iterations=3` and `selection=[+1]`, then
    the filter [None, None, [+1]] is returned.

    If a List[List[int]] is passed, then this is assumed to be the desired
    filter for each iteration, and is not modified.

    Basic checks are performed. ValueError is raised if specified parity
    selections are not in [-1, +1], or if specified angular selections are not
    >= 0.
    """
    if angular_cutoff is not None:
        if selection_type != "angular":
            raise ValueError(
                "`selection_type` must be 'angular' if specifying `angular_cutoff`"
            )
        if angular_cutoff < 1:
            raise ValueError("`angular_cutoff` must be >= 1")
    if selection is None:
        selection = [None] * n_iterations
    else:
        # If passed as int, use this for the last iteration only
        if isinstance(selection, int):
            selection = [None] * (n_iterations - 1) + [[selection]]
        else:
            if not isinstance(selection, List):
                raise TypeError(
                    "`selection` must be an int, List[int], or List[List[int]]"
                )
            if isinstance(selection[0], int):
                selection = [None] * (n_iterations - 1) + [selection]

    # Basic checks
    if not isinstance(selection, List):
        raise TypeError("`selection` must be an int, List[int], or List[List[int]]")
    for slct in selection:
        if slct is not None:
            if not np.all([isinstance(val, int) for val in slct]):
                raise TypeError(
                    "`selection` must be an int, List[int], or List[List[int]]"
                )
            if selection_type == "parity":
                if not np.all([val in [-1, +1] for val in slct]):
                    raise ValueError(
                        "specified layers in `selection` must only contain valid"
                        " parity values of -1 or +1"
                    )
                if not np.all([0 < len(slct) <= 2]):
                    raise ValueError(
                        "each parity filter must be a list of length 1 or 2,"
                        " with vals +1 and/or -1"
                    )
            elif selection_type == "angular":
                if not np.all([val >= 0 for val in slct]):
                    raise ValueError(
                        "specified layers in `selection` must only contain valid"
                        " angular channels >= 0"
                    )
                if angular_cutoff is not None:
                    if not np.all([val <= angular_cutoff for val in slct]):
                        raise ValueError(
                            "specified layers in `selection` must only contain valid"
                            " angular channels <= the specified `angular_cutoff`"
                        )
            else:
                raise ValueError(
                    "`selection_type` must be either 'parity' or 'angular'"
                )

    return selection


def _precompute_metadata(
    keys_1: Labels,
    keys_2: Labels,
    n_iterations: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[List[Union[None, List[int]]]] = None,
    parity_selection: Optional[List[Union[None, List[int]]]] = None,
) -> List[Tuple[Labels, List[List[int]]]]:
    """
    Computes all the metadata needed to perform `n_iterations` of CG combination
    steps, based on the keys of the 2 tensors being combined (`keys_1` and
    `keys_2`), the maximum angular channel cutoff (`angular_cutoff`), and the
    angular (`angular_selection`) and parity (`parity_selection`) selections to
    be applied at each iteration.
    """
    comb_metadata = []
    new_keys = keys_1
    for iteration in range(n_iterations):
        # Get the metadata for the combination of the 2 tensors
        i_comb_metadata = _precompute_metadata_one_iteration(
            keys_1=new_keys,
            keys_2=keys_2,
            angular_cutoff=angular_cutoff,
            angular_selection=angular_selection[iteration],
            parity_selection=parity_selection[iteration],
        )
        new_keys = i_comb_metadata[0]

        # Check that some keys are produced as a result of the combination
        if len(new_keys) == 0:
            raise ValueError(
                f"invalid selections: iteration {iteration + 1} produces no"
                " valid combinations. Check the `angular_selection` and"
                " `parity_selection` arguments."
            )

        # Now check the angular and parity selections are present in the new keys
        if angular_selection is not None:
            if angular_selection[iteration] is not None:
                for lam in angular_selection[iteration]:
                    if lam not in new_keys.column("spherical_harmonics_l"):
                        raise ValueError(
                            f"lambda = {lam} specified in `angular_selection`"
                            f" for iteration {iteration + 1}, but this is not a"
                            " valid angular channel based on the combination of"
                            " lower body-order tensors. Check the passed"
                            " `angular_selection` and try again."
                        )
        if parity_selection is not None:
            if parity_selection[iteration] is not None:
                for sig in parity_selection[iteration]:
                    if sig not in new_keys.column("inversion_sigma"):
                        raise ValueError(
                            f"sigma = {sig} specified in `parity_selection`"
                            f" for iteration {iteration + 1}, but this is not"
                            " a valid parity based on the combination of lower"
                            " body-order tensors. Check the passed"
                            " `parity_selection` and try again."
                        )

        comb_metadata.append(i_comb_metadata)

    return comb_metadata


def _precompute_metadata_one_iteration(
    keys_1: Labels,
    keys_2: Labels,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, List[int]]] = None,
    parity_selection: Optional[Union[None, List[int]]] = None,
) -> Tuple[Labels, List[List[int]]]:
    """
    Given the keys of 2 TensorMaps, returns the keys that would be present after
    a CG combination of these TensorMaps.

    Any angular or parity channel selections passed in `angular_selection` and
    `parity_selection` are applied such that only specified channels are present
    in the returned combined keys.

    Assumes that `keys_1` corresponds to a TensorMap with arbitrary body order,
    while `keys_2` corresponds to a TensorMap with body order 1.

    `keys_1`  must follow the key name convention:

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

        \bra{ n_1 l_1 ; n_2 l_2 k_2 ; ... ; n_{\nu-1} l_{\nu-1} k_{\nu-1} ;
        n_{\nu} l_{\nu} k_{\nu}; \lambda } \ket{ \rho^{\otimes \nu}; \lambda M }

    `keys_2` must follow the key name convention:

    ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]

    Returned is a tuple.

    The first element in the tuple is a Labels object corresponding to the keys
    created by a CG combination step.

    The second element is a list of list of ints. Each sublist corresponds to
    [lam1, lam2, correction_factor] terms. lam1 and lam2 tracks the lambda
    values of the blocks that combine to form the block indexed by the
    corresponding key. The correction_factor terms are the prefactors that
    account for the redundancy in the CG combination.

    The `parity_selection` argument can be used to return only keys with certain
    parities. This must be passed as a list with elements +1 and/or -1.
    """
    # Get the body order of the first TensorMap.
    unique_nu = np.unique(keys_1.column("order_nu"))
    if len(unique_nu) > 1:
        raise ValueError(
            "keys_1 must correspond to a tensor of a single body order."
            f" Found {len(unique_nu)} body orders: {unique_nu}"
        )
    nu1 = unique_nu[0]

    # Define nu value of output TensorMap
    nu = nu1 + 1

    # The body order of the second TensorMap should be nu = 1.
    assert np.all(keys_2.column("order_nu") == 1)

    # If nu1 = 1, the key names don't yet have any "lx" columns
    if nu1 == 1:
        l_list_names = []
        new_l_list_names = ["l1", "l2"]
    else:
        l_list_names = [f"l{l}" for l in range(1, nu1 + 1)]
        new_l_list_names = l_list_names + [f"l{nu}"]

    # Check key names
    assert np.all(
        keys_1.names
        == ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]
        + l_list_names
        + [f"k{k}" for k in range(2, nu1)]
    )
    assert np.all(
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
        nonzero_lams = np.arange(abs(lam1 - lam2), abs(lam1 + lam2) + 1)

        # Now iterate over the non-zero angular channels and apply the custom
        # selections
        for lam in nonzero_lams:
            # Skip combination if it forms an angular channel of order greater
            # than the specified maximum cutoff `angular_cutoff`.
            if angular_cutoff is not None:
                if lam > angular_cutoff:
                    continue

            # Skip combination if it creates an angular channel that has not
            # been explicitly selected
            if angular_selection is not None:
                if lam not in angular_selection:
                    continue

            # Calculate new sigma
            sig = sig1 * sig2 * (-1) ** (lam1 + lam2 + lam)

            # Skip combination if it creates a parity that has not been
            # explicitly selected
            if parity_selection is not None:
                if sig not in parity_selection:
                    continue

            # Extract the l and k lists from keys_1
            l_list = key_1.values[4 : 4 + nu1].tolist()
            k_list = key_1.values[4 + nu1 :].tolist()

            # Build the new keys values. l{nu} is `lam2`` (i.e.
            # "spherical_harmonics_l" of the key from `keys_2`. k{nu-1} is
            # `lam1` (i.e. "spherical_harmonics_l" of the key from `keys_1`).
            new_vals = [nu, sig, lam, a] + l_list + [lam2] + k_list + [lam1]
            new_key_values.append(new_vals)
            keys_1_entries.append(key_1)
            keys_2_entries.append(key_2)

    # Define new keys as the full product of keys_1 and keys_2
    nu_x_keys = Labels(names=new_names, values=np.array(new_key_values))

    # Now account for multiplicty
    key_idxs_to_keep = []
    mult_dict = {}
    for key_idx, key in enumerate(nu_x_keys):
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
        if np.all(key_slice_tuple == key_slice_sorted_tuple):
            key_idxs_to_keep.append(key_idx)

        # Now count the multiplicity of each sorted l_list
        if mult_dict.get(key_slice_sorted_tuple) is None:
            mult_dict[key_slice_sorted_tuple] = 1
        else:
            mult_dict[key_slice_sorted_tuple] += 1

    # Build a reduced Labels object for the combined keys, with redundancies removed
    combined_keys_red = Labels(
        names=new_names,
        values=np.array([nu_x_keys[idx].values for idx in key_idxs_to_keep]),
    )

    # Create a of LabelsEntry objects that correspond to the original keys in
    # `keys_1` and `keys_2` that combined to form the combined key
    keys_1_entries_red = [keys_1_entries[idx] for idx in key_idxs_to_keep]
    keys_2_entries_red = [keys_2_entries[idx] for idx in key_idxs_to_keep]

    # Define the multiplicity of each key
    mult_list = [
        mult_dict[tuple(nu_x_keys[idx].values[: 4 + (nu + 1)].tolist())]
        for idx in key_idxs_to_keep
    ]

    return combined_keys_red, keys_1_entries_red, keys_2_entries_red, mult_list


# ==================================================================
# ===== Functions to perform the CG combinations of blocks
# ==================================================================


def _combine_single_center_blocks(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lam: int,
    cg_cache,
    correction_factor: float = 1.0,
    return_metadata_only: bool = False,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired angular channel, combines the
    values arrays and returns a new TensorBlock.
    """

    # Do the CG combination - single center so no shape pre-processing required
    if return_metadata_only:
        combined_values = _combine_arrays_sparse(
            block_1.values, block_2.values, lam, cg_cache, return_empty_array=True
        )
    else:
        combined_values = _combine_arrays(
            block_1.values, block_2.values, lam, cg_cache, return_empty_array=False
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
        values=combined_values * correction_factor,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=np.arange(-lam, lam + 1).reshape(-1, 1),
            ),
        ],
        properties=Labels(
            names=prop_names,
            values=np.array(
                [
                    np.concatenate((b2, b1))
                    for b2 in block_2.properties.values
                    for b1 in block_1.properties.values
                ]
            ),
        ),
    )

    return combined_block


def _combine_arrays(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
    cg_cache,
    return_empty_array: bool = False,
) -> np.ndarray:
    """
    Couples arrays corresponding to the irreducible spherical components of 2
    angular channels l1 and l2 using the appropriate Clebsch-Gordan
    coefficients. As l1 and l2 can be combined to form multiple lambda channels,
    this function returns the coupling to a single specified channel `lambda`.

    `arr_1` has shape (n_i, 2 * l1 + 1, n_p) and `arr_2` has shape (n_i, 2 * l2
    + 1, n_q). n_i is the number of samples, n_p and n_q are the number of
    properties in each array. The number of samples in each array must be the
    same.

    The ouput array has shape (n_i, 2 * lambda + 1, n_p * n_q), where lambda is
    the input parameter `lam`.

    The Clebsch-Gordan coefficients are cached in `cg_cache`. Currently, these
    must be produced by the ClebschGordanReal class in this module.

    Either performs the operation in a dense or sparse manner, depending on the
    value of `sparse`.

    `return_empty_array` can be used to return an empty array of the correct
    shape, without performing the CG combination step. This can be useful for
    probing the outputs of CG iterations in terms of metadata without the
    computational cost of performing the CG combinations - i.e. using the
    function :py:func:`combine_single_center_to_body_order_metadata_only`.
    """
    if return_empty_array:
        return _combine_arrays_sparse(arr_1, arr_2, lam, cg_cache, True)

    # Check the first dimension of the arrays are the same (i.e. same samples)
    if cg_cache.sparse:
        return _combine_arrays_sparse(arr_1, arr_2, lam, cg_cache, False)
    return _combine_arrays_dense(arr_1, arr_2, lam, cg_cache)


def _combine_arrays_sparse(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
    cg_cache,
    return_empty_array: bool = False,
) -> np.ndarray:
    """
    TODO: finish docstring.

    Performs a Clebsch-Gordan combination step on 2 arrays using sparse
    operations.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: sparse dictionary with keys (m1, m2, mu) and array values
        being sparse blocks of shape <TODO: fill out>

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]
    """
    # Samples dimensions must be the same
    assert arr_1.shape[0] == arr_2.shape[0]

    # Define other useful dimensions
    n_i = arr_1.shape[0]  # number of samples
    n_p = arr_1.shape[2]  # number of properties in arr_1
    n_q = arr_2.shape[2]  # number of properties in arr_2

    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2

    # Initialise output array
    arr_out = np.zeros((n_i, 2 * lam + 1, n_p * n_q))

    if return_empty_array:
        return arr_out

    # Get the corresponding Clebsch-Gordan coefficients
    cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

    # Fill in each mu component of the output array in turn
    for m1, m2, mu in cg_coeffs.keys():
        # Broadcast arrays, multiply together and with CG coeff
        arr_out[:, mu, :] += (
            arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeffs[(m1, m2, mu)]
        ).reshape(n_i, n_p * n_q)

    return arr_out


def _combine_arrays_dense(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lam: int,
    cg_cache,
) -> np.ndarray:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using a dense
    operation.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: dense array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lam +
        1)]

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]

    TODO: do we need this example here? Could it be moved to a test?

    >>> N_SAMPLES = 30
    >>> N_Q_PROPERTIES = 10
    >>> N_P_PROPERTIES = 8
    >>> L1 = 2
    >>> L2 = 3
    >>> LAM = 2
    >>> arr_1 = np.random.rand(N_SAMPLES, 2 * L1 + 1, N_Q_PROPERTIES)
    >>> arr_2 = np.random.rand(N_SAMPLES, 2 * L2 + 1, N_P_PROPERTIES)
    >>> cg_cache = {(L1, L2, LAM): np.random.rand(2 * L1 + 1, 2 * L2 + 1, 2 * LAM + 1)}
    >>> out1 = _clebsch_gordan_dense(arr_1, arr_2, LAM, cg_cache)
    >>> # (samples l1_m  q_features) (samples l2_m p_features),
    >>> #   (l1_m  l2_m  lambda_mu)
    >>> # --> (samples, lambda_mu q_features p_features)
    >>> # in einsum l1_m is l, l2_m is k, lambda_mu is L
    >>> out2 = np.einsum("slq, skp, lkL -> sLqp", arr_1, arr_2, cg_cache[(L1, L2, LAM)])
    >>> # --> (samples lambda_mu (q_features p_features))
    >>> out2 = out2.reshape(arr_1.shape[0], 2 * LAM + 1, -1)
    >>> print(np.allclose(out1, out2))
    True
    """
    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2
    cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

    # (samples None None l1_mu q) * (samples l2_mu p None None) -> (samples l2_mu p l1_mu q)
    # we broadcast it in this way so we only need to do one swapaxes in the next step
    arr_out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]

    # (samples l2_mu p l1_mu q) -> (samples q p l1_mu l2_mu)
    arr_out = arr_out.swapaxes(1, 4)

    # samples (q p l1_mu l2_mu) -> (samples (q p) (l1_mu l2_mu))
    arr_out = arr_out.reshape(
        -1, arr_1.shape[2] * arr_2.shape[2], arr_1.shape[1] * arr_2.shape[1]
    )

    # (l1_mu l2_mu lam_mu) -> ((l1_mu l2_mu) lam_mu)
    cg_coeffs = cg_coeffs.reshape(-1, 2 * lam + 1)

    # (samples (q p) (l1_mu l2_mu)) @ ((l1_mu l2_mu) lam_mu) -> samples (q p) lam_mu
    arr_out = arr_out @ cg_coeffs

    # (samples (q p) lam_mu) -> (samples lam_mu (q p))
    return arr_out.swapaxes(1, 2)
