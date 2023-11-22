"""
Module for computing Clebsch-gordan iterations with metatensor TensorMaps.
"""
import itertools
from typing import List, Optional, Tuple, Union

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from . import _dispatch
from ._cg_cache import ClebschGordanReal


# ======================================================================
# ===== Public API functions
# ======================================================================


def correlate_density(
    density: TensorMap,
    correlation_order: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> List[TensorMap]:
    """
    Takes iterative Clebsch-Gordan (CG) tensor products of a density descriptor
    with itself to the desired correlation order. Returns a list of TensorMaps
    corresponding to the density correlations output from the specified
    iterations.

    A density descriptor necessarily is body order 2 (i.e. correlation order 1),
    but can be single- or multi-center. The output is a list of density
    correlations for each iteration specified in `output_selection`, up to the
    target order passed in `correlation_order`.

    This function is an iterative special case of the more general
    :py:func:`correlate_tensors`. As a density is being correlated with itself,
    some redundant CG tensor products can be skipped with the `skip_redundant`
    keyword.

    Selections on the angular and parity channels at each iteration can also be
    controlled with arguments `angular_cutoff`, `angular_selection` and
    `parity_selection`.

    :param density: A density descriptor of body order 2 (correlation order 1),
        in metatensor.TensorMap format. This may be, for example, a rascaline
        :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
        Alternatively, this could be multi-center descriptor, such as a pair
        density.
    :param correlation_order: The desired correlation order of the output
        descriptor. Must be >= 1.
    :param angular_cutoff: The maximum angular channel to compute at any given
        CG iteration, applied globally to all iterations until the target
        correlation order is reached.
    :param angular_selection: A list of angular channels to output at each
        iteration. If a single list is passed, this is applied to the final
        iteration only. If a list of lists is passed, this is applied to each
        iteration. If None is passed, all angular channels are output at each
        iteration.
    :param parity_selection: A list of parity channels to output at each
        iteration. If a single list is passed, this is applied to the final
        iteration only. If a list of lists is passed, this is applied to each
        iteration. If None is passed, all parity channels are output at each
        iteration.
    :param skip_redundant: Whether to skip redundant CG combinations. Defaults
        to False, which means all combinations are performed. If a list of bool
        is passed, this is applied to each iteration. If a single bool is
        passed, this is applied to all iterations.
    :param output_selection: A list of bools specifying whether to output a
        TensorMap for each iteration. If a single bool is passed as True,
        outputs from all iterations will be returned. If a list of bools is
        passed, this controls the output at each corresponding iteration. If
        None is passed, only the final iteration is output.

    :return List[TensorMap]: A list of TensorMaps corresponding to the density
        correlations output from the specified iterations.
    """
    return _correlate_density(
        density,
        correlation_order,
        angular_cutoff,
        angular_selection,
        parity_selection,
        skip_redundant,
        output_selection,
        compute_metadata_only=False,
        sparse=True,  # sparse CG cache by default
    )


def correlate_density_metadata(
    density: TensorMap,
    correlation_order: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> List[TensorMap]:
    """
    Returns the metadata-only TensorMaps that would be output by the function
    :py:func:`correlate_density` under the same settings, without perfoming the
    actual Clebsch-Gordan tensor products. See this function for full
    documentation.

    :param density: A density descriptor of body order 2 (correlation order 1),
        in metatensor.TensorMap format. This may be, for example, a rascaline
        :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
        Alternatively, this could be multi-center descriptor, such as a pair
        density.
    :param correlation_order: The desired correlation order of the output
        descriptor. Must be >= 1.
    :param angular_cutoff: The maximum angular channel to compute at any given
        CG iteration, applied globally to all iterations until the target
        correlation order is reached.
    :param angular_selection: A list of angular channels to output at each
        iteration. If a single list is passed, this is applied to the final
        iteration only. If a list of lists is passed, this is applied to each
        iteration. If None is passed, all angular channels are output at each
        iteration.
    :param parity_selection: A list of parity channels to output at each
        iteration. If a single list is passed, this is applied to the final
        iteration only. If a list of lists is passed, this is applied to each
        iteration. If None is passed, all parity channels are output at each
        iteration.
    :param skip_redundant: Whether to skip redundant CG combinations. Defaults
        to False, which means all combinations are performed. If a list of bool
        is passed, this is applied to each iteration. If a single bool is
        passed, this is applied to all iterations.
    :param output_selection: A list of bools specifying whether to output a
        TensorMap for each iteration. If a single bool is passed as True,
        outputs from all iterations will be returned. If a list of bools is
        passed, this controls the output at each corresponding iteration. If
        None is passed, only the final iteration is output.

    :return List[TensorMap]: A list of TensorMaps corresponding to the metadata
        that would be output by :py:func:`correlate_density` under the same
        settings.
    """

    return _correlate_density(
        density,
        correlation_order,
        angular_cutoff,
        angular_selection,
        parity_selection,
        skip_redundant,
        output_selection,
        compute_metadata_only=True,
    )


# ====================================================================
# ===== Private functions that do the work on the TensorMap level
# ====================================================================


def _correlate_density(
    density: TensorMap,
    correlation_order: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
    compute_metadata_only: bool = False,
    sparse: bool = True,
) -> List[TensorMap]:
    """
    Performs the density correlations for public functions
    :py:func:`correlate_density` and :py:func:`correlate_density_metadata`.
    """
    if correlation_order <= 1:
        raise ValueError("`correlation_order` must be > 1")
    if _dispatch.any([len(list(block.gradients())) > 0 for block in density]):
        raise NotImplementedError(
            "Clebsch Gordan combinations with gradients not yet implemented."
            " Use metatensor.remove_gradients to remove gradients from the input."
        )
    n_iterations = correlation_order - 1  # num iterations
    density = _standardize_metadata(density)  # standardize metadata
    density_correlation = density  # create a copy to combine with itself

    # Parse the various selection filters
    angular_selection, parity_selection = _parse_int_selections(
        n_iterations=n_iterations,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
    )
    skip_redundant, output_selection = _parse_bool_selections(
        n_iterations,
        skip_redundant=skip_redundant,
        output_selection=output_selection,
    )

    # Pre-compute the metadata needed to perform each CG iteration
    metadata = _precompute_metadata(
        density.keys,
        density.keys,
        n_iterations=n_iterations,
        angular_cutoff=angular_cutoff,
        angular_selection=angular_selection,
        parity_selection=parity_selection,
        skip_redundant=skip_redundant,
    )
    # Compute CG coefficient cache
    if compute_metadata_only:
        cg_cache = None
    else:
        angular_max = max(
            _dispatch.concatenate(
                [density.keys.column("spherical_harmonics_l")]
                + [mdata[2].column("spherical_harmonics_l") for mdata in metadata]
            )
        )
        # TODO: metadata has been precomputed, so perhaps we don't need to
        # compute all CG coefficients up to angular_max here.
        # TODO: use sparse cache by default until we understamd under which
        # circumstances (and if) dense is faster.
        cg_cache = ClebschGordanReal(angular_max, sparse=sparse)

    # Perform iterative CG tensor products
    density_correlations = []
    for iteration in range(n_iterations):
        # Define the correlation order of the current iteration
        correlation_order_it = iteration + 2

        blocks_out = []
        # TODO: is there a faster way of iterating over keys/blocks here?
        for key_1, key_2, key_out in zip(*metadata[iteration]):
            block_out = _combine_single_center_blocks(
                density_correlation[key_1],
                density[key_2],
                key_out["spherical_harmonics_l"],
                cg_cache,
                compute_metadata_only=compute_metadata_only,
            )
            blocks_out.append(block_out)
        keys_out = metadata[iteration][2]
        density_correlation = TensorMap(keys=keys_out, blocks=blocks_out)

        # If this tensor is to be included in the output, move the [l1, l2, ...]
        # keys to properties and store
        if output_selection[iteration]:
            density_correlations.append(
                density_correlation.keys_to_properties(
                    [f"l{i}" for i in range(1, correlation_order_it + 1)]
                    + [f"k{i}" for i in range(2, correlation_order_it)]
                )
            )

    # Drop redundant key names. TODO: these should be part of the global
    # matadata associated with the TensorMap. Awaiting this functionality in
    # metatensor.
    for i, tensor in enumerate(density_correlations):
        keys = tensor.keys
        if len(_dispatch.unique(tensor.keys.column("order_nu"))) == 1:
            keys = keys.remove(name="order_nu")
        if len(_dispatch.unique(tensor.keys.column("inversion_sigma"))) == 1:
            keys = keys.remove(name="inversion_sigma")
        density_correlations[i] = TensorMap(
            keys=keys, blocks=[b.copy() for b in tensor.blocks()]
        )

    return density_correlations


def correlate_tensors(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[None, int, List[int], List[List[int]]]] = None,
) -> TensorMap:
    """
    Performs the Clebsch Gordan tensor product of two TensorMaps that correspond
    to densities or density correlations. Returns a new TensorMap corresponding
    to a higher correlation-order descriptor.

    The two input tensors can be single- or multi-center, and of arbitrary (and
    different) correlation order, but must contain the same samples.
    """
    raise NotImplementedError


# ==================================================================
# ===== Functions to handle metadata
# ==================================================================


def _standardize_metadata(tensor: TensorMap) -> TensorMap:
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


def _parse_int_selections(
    n_iterations: int,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[int, List[int], List[List[int]]]] = None,
    parity_selection: Optional[Union[int, List[int], List[List[int]]]] = None,
) -> List[List[List[int]]]:
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
        if angular_cutoff < 1:
            raise ValueError("`angular_cutoff` must be >= 1")

    selections = []
    for selection_type, selection in zip(
        ["angular", "parity"], [angular_selection, parity_selection]
    ):
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
                if not _dispatch.all([isinstance(val, int) for val in slct]):
                    raise TypeError(
                        "`selection` must be an int, List[int], or List[List[int]]"
                    )
                if selection_type == "parity":
                    if not _dispatch.all([val in [-1, +1] for val in slct]):
                        raise ValueError(
                            "specified layers in `selection` must only contain valid"
                            " parity values of -1 or +1"
                        )
                    if not _dispatch.all([0 < len(slct) <= 2]):
                        raise ValueError(
                            "each parity filter must be a list of length 1 or 2,"
                            " with vals +1 and/or -1"
                        )
                elif selection_type == "angular":
                    if not _dispatch.all([val >= 0 for val in slct]):
                        raise ValueError(
                            "specified layers in `selection` must only contain valid"
                            " angular channels >= 0"
                        )
                    if angular_cutoff is not None:
                        if not _dispatch.all([val <= angular_cutoff for val in slct]):
                            raise ValueError(
                                "specified layers in `selection` must only contain valid"
                                " angular channels <= the specified `angular_cutoff`"
                            )
                else:
                    raise ValueError(
                        "`selection_type` must be either 'parity' or 'angular'"
                    )
        selections.append(selection)

    return selections


def _parse_bool_selections(
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
        raise TypeError("`skip_redundant` must be a bool or list of bools")
    if not len(skip_redundant) == n_iterations:
        raise ValueError(
            "`skip_redundant` must be a bool or list of bools of length"
            " `correlation_order` - 1"
        )
    if output_selection is None:
        output_selection = [False] * (n_iterations - 1) + [True]
    else:
        if isinstance(output_selection, bool):
            output_selection = [output_selection] * n_iterations
        if not isinstance(output_selection, List):
            raise TypeError("`output_selection` must be passed as a list of bools")

    if not len(output_selection) == n_iterations:
        raise ValueError(
            "`output_selection` must be a list of bools of length"
            " corresponding to the number of CG iterations"
        )
    if not _dispatch.all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a list of bools")
    if not _dispatch.all([isinstance(v, bool) for v in output_selection]):
        raise TypeError("`output_selection` must be passed as a list of bools")

    return skip_redundant, output_selection


def _precompute_metadata(
    keys_1: Labels,
    keys_2: Labels,
    n_iterations: int,
    angular_cutoff: int,
    angular_selection: List[Union[None, List[int]]],
    parity_selection: List[Union[None, List[int]]],
    skip_redundant: List[bool],
) -> List[Tuple[Labels, List[List[int]]]]:
    """
    Computes all the metadata needed to perform `n_iterations` of CG combination
    steps, based on the keys of the 2 tensors being combined (`keys_1` and
    `keys_2`), the maximum angular channel cutoff (`angular_cutoff`), and the
    angular (`angular_selection`) and parity (`parity_selection`) selections to
    be applied at each iteration.
    """
    metadata = []
    keys_out = keys_1
    for iteration in range(n_iterations):
        # Get the metadata for the combination of the 2 tensors
        i_metadata = _precompute_metadata_one_iteration(
            keys_1=keys_out,
            keys_2=keys_2,
            angular_cutoff=angular_cutoff,
            angular_selection=angular_selection[iteration],
            parity_selection=parity_selection[iteration],
            skip_redundant=skip_redundant[iteration],
        )
        keys_out = i_metadata[2]

        # Check that some keys are produced as a result of the combination
        if len(keys_out) == 0:
            raise ValueError(
                f"invalid selections: iteration {iteration + 1} produces no"
                " valid combinations. Check the `angular_selection` and"
                " `parity_selection` arguments."
            )

        # Now check the angular and parity selections are present in the new keys
        if angular_selection is not None:
            if angular_selection[iteration] is not None:
                for lam in angular_selection[iteration]:
                    if lam not in keys_out.column("spherical_harmonics_l"):
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
                    if sig not in keys_out.column("inversion_sigma"):
                        raise ValueError(
                            f"sigma = {sig} specified in `parity_selection`"
                            f" for iteration {iteration + 1}, but this is not"
                            " a valid parity based on the combination of lower"
                            " body-order tensors. Check the passed"
                            " `parity_selection` and try again."
                        )

        metadata.append(i_metadata)

    return metadata


def _precompute_metadata_one_iteration(
    keys_1: Labels,
    keys_2: Labels,
    angular_cutoff: Optional[int] = None,
    angular_selection: Optional[Union[None, List[int]]] = None,
    parity_selection: Optional[Union[None, List[int]]] = None,
    skip_redundant: bool = False,
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

    Returned is a tuple. The first element in the tuple is a Labels object
    corresponding to the keys created by a CG combination step.

    The second element is a list of list of ints. Each sublist corresponds to
    [lam1, lam2, correction_factor] terms. lam1 and lam2 tracks the lambda
    values of the blocks that combine to form the block indexed by the
    corresponding key. The correction_factor terms are the prefactors that
    account for the redundancy in the CG combination.

    The `parity_selection` argument can be used to return only keys with certain
    parities. This must be passed as a list with elements +1 and/or -1.

    The `skip_redundant` arg can be used to skip the calculation of redundant
    block combinations - i.e. those that have equivalent sorted l lists. Only
    the one for which l1 <= l2 <= ... <= ln is calculated.
    """
    # Get the body order of the first TensorMap.
    unique_nu = _dispatch.unique(keys_1.column("order_nu"))
    if len(unique_nu) > 1:
        raise ValueError(
            "keys_1 must correspond to a tensor of a single body order."
            f" Found {len(unique_nu)} body orders: {unique_nu}"
        )
    nu1 = unique_nu[0]

    # Define nu value of output TensorMap
    nu = nu1 + 1

    # The body order of the second TensorMap should be nu = 1.
    assert _dispatch.all(keys_2.column("order_nu") == 1)

    # If nu1 = 1, the key names don't yet have any "lx" columns
    if nu1 == 1:
        l_list_names = []
        new_l_list_names = ["l1", "l2"]
    else:
        l_list_names = [f"l{l}" for l in range(1, nu1 + 1)]
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
    keys_out = Labels(
        names=new_names,
        values=_dispatch.int_array_like(new_key_values, like=keys_1.values),
    )

    # Don't skip the calculation of redundant blocks
    if skip_redundant is False:
        return keys_1_entries, keys_2_entries, keys_out

    # Now account for multiplicty
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
        names=new_names,
        values=_dispatch.int_array_like(
            [keys_out[idx].values for idx in key_idxs_to_keep], like=keys_1.values
        ),
    )

    # Create a of LabelsEntry objects that correspond to the original keys in
    # `keys_1` and `keys_2` that combined to form the combined key
    keys_1_entries_red = [keys_1_entries[idx] for idx in key_idxs_to_keep]
    keys_2_entries_red = [keys_2_entries[idx] for idx in key_idxs_to_keep]

    return keys_1_entries_red, keys_2_entries_red, keys_out_red


# ==================================================================
# ===== Functions to perform the CG combinations of blocks
# ==================================================================


def _combine_single_center_blocks(
    block_1: TensorBlock,
    block_2: TensorBlock,
    lam: int,
    cg_cache,
    compute_metadata_only: bool = False,
) -> TensorBlock:
    """
    For a given pair of TensorBlocks and desired angular channel, combines the
    values arrays and returns a new TensorBlock.
    """

    # Do the CG combination - single center so no shape pre-processing required
    if compute_metadata_only:
        combined_values = _dispatch.combine_arrays(
            block_1.values, block_2.values, lam, cg_cache, return_empty_array=True
        )
    else:
        combined_values = _dispatch.combine_arrays(
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
        values=combined_values,
        samples=block_1.samples,
        components=[
            Labels(
                names=["spherical_harmonics_m"],
                values=_dispatch.int_range_like(
                    min_val=-lam, max_val=lam + 1, like=block_1.values
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
