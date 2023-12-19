"""
Module for computing Clebsch-gordan tensor product iterations on density (i.e.
correlation order 1) tensors in TensorMap form. A special (and iterative) case
of the more general :py:mod:`correlate_tensors` module.
"""
from typing import List, Optional, Union

from metatensor import Labels, TensorMap

from . import _cg_cache, _clebsch_gordan, _dispatch


# ======================================================================
# ===== Public API functions
# ======================================================================


def correlate_density(
    density: TensorMap,
    correlation_order: int,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> Union[TensorMap, List[TensorMap]]:
    """
    Takes iterative Clebsch-Gordan (CG) tensor products of a density descriptor
    with itself up to the desired correlation order. Returns
    :py:class:`TensorMap`(s) corresponding to the density correlations output
    from the specified iteration(s).

    A density descriptor necessarily is body order 2 (i.e. correlation order 1),
    but can be single- or multi-center. The output is a :py:class:`list` of
    density correlations for each iteration specified in `output_selection`, up
    to the target order passed in `correlation_order`. By default only the last
    correlation (i.e. the correlation of order ``correlation_order``) is
    returned.

    This function is an iterative special case of the more general
    :py:mod:`correlate_tensors`. As a density is being correlated with itself,
    some redundant CG tensor products can be skipped with the `skip_redundant`
    keyword.

    Selections on the angular and parity channels at each iteration can also be
    controlled with arguments `selected_keys`.

    :param density: A density descriptor of body order 2 (correlation order 1),
        in :py:class:`TensorMap` format. This may be, for example, a rascaline
        :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
        Alternatively, this could be multi-center descriptor, such as a pair
        density.
    :param correlation_order: The desired correlation order of the output
        descriptor. Must be >= 1.
    :param angular_cutoff: The maximum angular channel to compute at any given
        CG iteration, applied globally to all iterations until the target
        correlation order is reached.
    :param selected_keys: :py:class:`Labels` or `List[:py:class:`Labels`]`
        specifying the angular and/or parity channels to output at each
        iteration. All :py:class:`Labels` objects passed here must only contain
        key names "spherical_harmonics_l" and "inversion_sigma". If a single
        :py:class:`Labels` object is passed, this is applied to the final
        iteration only. If a :py:class:`list` of :py:class:`Labels` objects is
        passed, each is applied to its corresponding iteration. If None is
        passed, all angular and parity channels are output at each iteration,
        with the global `angular_cutoff` applied if specified.
    :param skip_redundant: Whether to skip redundant CG combinations. Defaults
        to False, which means all combinations are performed. If a
        :py:class:`list` of :py:class:`bool` is passed, this is applied to each
        iteration. If a single :py:class:`bool` is passed, this is applied to
        all iterations.
    :param output_selection: A :py:class:`list` of :py:class:`bool` specifying
        whether to output a :py:class:`TensorMap` for each iteration. If a
        single :py:class:`bool` is passed as True, outputs from all iterations
        will be returned. If a :py:class:`list` of :py:class:`bool` is passed,
        this controls the output at each corresponding iteration. If None is
        passed, only the final iteration is output.

    :return: A :py:class:`list` of :py:class:`TensorMap` corresponding to the
        density correlations output from the specified iterations. If the output
        from a single iteration is requested, a :py:class:`TensorMap` is
        returned instead.
    """
    return _correlate_density(
        density,
        correlation_order,
        angular_cutoff,
        selected_keys,
        skip_redundant,
        output_selection,
        compute_metadata_only=False,
        sparse=True,  # sparse CG cache by default
    )


def correlate_density_metadata(
    density: TensorMap,
    correlation_order: int,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
) -> Union[TensorMap, List[TensorMap]]:
    """
    Returns the metadata-only :py:class:`TensorMap`(s) that would be output by
    the function :py:func:`correlate_density` under the same settings, without
    perfoming the actual Clebsch-Gordan tensor products. See this function for
    full documentation.
    """

    return _correlate_density(
        density,
        correlation_order,
        angular_cutoff,
        selected_keys,
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
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    skip_redundant: Optional[Union[bool, List[bool]]] = False,
    output_selection: Optional[Union[bool, List[bool]]] = None,
    compute_metadata_only: bool = False,
    sparse: bool = True,
) -> Union[TensorMap, List[TensorMap]]:
    """
    Performs the density correlations for public functions
    :py:func:`correlate_density` and :py:func:`correlate_density_metadata`.
    """
    # Check inputs
    if correlation_order <= 1:
        raise ValueError("`correlation_order` must be > 1")
    # TODO: implement combinations of gradients too
    if _dispatch.any([len(list(block.gradients())) > 0 for block in density]):
        raise NotImplementedError(
            "Clebsch Gordan combinations with gradients not yet implemented."
            " Use metatensor.remove_gradients to remove gradients from the input."
        )
    # Check metadata
    if not (
        _dispatch.all(density.keys.names == ["spherical_harmonics_l", "species_center"])
        or _dispatch.all(
            density.keys.names
            == ["spherical_harmonics_l", "species_center", "species_neighbor"]
        )
    ):
        raise ValueError(
            "input `density` must have key names"
            ' ["spherical_harmonics_l", "species_center"] or'
            ' ["spherical_harmonics_l", "species_center", "species_neighbor"]'
        )
    if not _dispatch.all(density.component_names == ["spherical_harmonics_m"]):
        raise ValueError(
            "input `density` must have a single component"
            " axis with name `spherical_harmonics_m`"
        )
    n_iterations = correlation_order - 1  # num iterations
    density = _clebsch_gordan._standardize_keys(density)  # standardize metadata
    density_correlation = density  # create a copy to combine with itself

    # Parse the selected keys
    selected_keys = _clebsch_gordan._parse_selected_keys(
        n_iterations=n_iterations,
        angular_cutoff=angular_cutoff,
        selected_keys=selected_keys,
        like=density.keys.values,
    )
    # Parse the bool flags that control skipping of redundant CG combinations
    # and TensorMap output from each iteration
    skip_redundant, output_selection = _clebsch_gordan._parse_bool_iteration_filters(
        n_iterations,
        skip_redundant=skip_redundant,
        output_selection=output_selection,
    )

    # Pre-compute the keys needed to perform each CG iteration
    key_metadata = _clebsch_gordan._precompute_keys(
        density.keys,
        density.keys,
        n_iterations=n_iterations,
        selected_keys=selected_keys,
        skip_redundant=skip_redundant,
    )
    # Compute CG coefficient cache
    if compute_metadata_only:
        cg_cache = None
    else:
        angular_max = max(
            _dispatch.concatenate(
                [density.keys.column("spherical_harmonics_l")]
                + [mdata[2].column("spherical_harmonics_l") for mdata in key_metadata]
            )
        )
        # TODO: keys have been precomputed, so perhaps we don't need to
        # compute all CG coefficients up to angular_max here.
        # TODO: use sparse cache by default until we understand under which
        # circumstances (and if) dense is faster.
        cg_cache = _cg_cache.ClebschGordanReal(angular_max, sparse=sparse)

    # Perform iterative CG tensor products
    density_correlations = []
    for iteration in range(n_iterations):
        # Define the correlation order of the current iteration
        correlation_order_it = iteration + 2

        # Combine block pairs
        blocks_out = []
        for key_1, key_2, key_out in zip(*key_metadata[iteration]):
            block_out = _clebsch_gordan._combine_blocks_same_samples(
                density_correlation[key_1],
                density[key_2],
                key_out["spherical_harmonics_l"],
                cg_cache,
                compute_metadata_only=compute_metadata_only,
            )
            blocks_out.append(block_out)
        keys_out = key_metadata[iteration][2]
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

    # Return a single TensorMap in the simple case
    if len(density_correlations) == 1:
        return density_correlations[0]

    # Otherwise return a list of TensorMaps
    return density_correlations
