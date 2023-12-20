"""
Module for computing the Clebsch-gordan tensor product on arbitrary tensors in
TensorMap form, where the samples are equivalent. 

For a special iterative case specifically for densities, see the
:py:mod:`correlate_density` module.
"""
from typing import List, Optional, Union

from metatensor import Labels, TensorMap

from . import _cg_cache, _clebsch_gordan, _dispatch


# ======================================================================
# ===== Public API functions
# ======================================================================


def correlate_tensors(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
) -> TensorMap:
    """
    Takes a Clebsch-Gordan (CG) tensor product of two tensors in
    :py:class:`TensorMap` form, and returns the result in :py:class:`TensorMap`.

    The keys of `tensor_1` and `tensor_2` must have dimensions
    "inversion_sigma", "spherical_harmonics_l", "species_center", and any "l{x}"
    keys (where x is a positive integer) leftover from previous CG tensor
    products.

    This function is an iterative special case of the more general
    :py:mod:`correlate_tensors`. As a density is being correlated with itself,
    some redundant CG tensor products can be skipped with the `skip_redundant`
    keyword.

    :param tensor_1: the first tensor to correlate.
    :param tensor_2: the second tensor to correlate.
    :param selected_keys: :py:class:`Labels` or `List[:py:class:`Labels`]`
        specifying the angular and/or parity channels to output at each
        iteration. All :py:class:`Labels` objects passed here must only contain
        key names "spherical_harmonics_l" and "inversion_sigma". If a single
        :py:class:`Labels` object is passed, this is applied to the final
        iteration only. If a :py:class:`list` of :py:class:`Labels` objects is
        passed, each is applied to its corresponding iteration. If None is
        passed, all angular and parity channels are output at each iteration,
        with the global `angular_cutoff` applied if specified.

    :return: A :py:class:`TensorMap` corresponding to the correlated tensors.
    """
    return _correlate_tensors(
        tensor_1,
        tensor_2,
        selected_keys,
        compute_metadata_only=False,
        sparse=True,  # sparse CG cache by default
    )


def correlate_tensors_metadata(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
) -> TensorMap:
    """
    Returns the metadata-only :py:class:`TensorMap`(s) that would be output by
    the function :py:func:`correlate_tensors` under the same settings, without
    perfoming the actual Clebsch-Gordan tensor product. See this function for
    full documentation.
    """
    return _correlate_tensors(
        tensor_1,
        tensor_2,
        selected_keys,
        compute_metadata_only=True,
        sparse=True,  # sparse CG cache by default
    )


# ====================================================================
# ===== Private functions that do the work on the TensorMap level
# ====================================================================


def _correlate_tensors(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    compute_metadata_only: bool = False,
    sparse: bool = True,
) -> TensorMap:
    """
    Performs the Clebsch-Gordan tensor product for public functions
    :py:func:`correlate_tensors` and :py:func:`correlate_tensors_metadata`.
    """
    # Check inputs
    # TODO: implement combinations of gradients too
    for tensor in [tensor_1, tensor_2]:
        if _dispatch.any([len(list(block.gradients())) > 0 for block in tensor]):
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented."
                " Use metatensor.remove_gradients to remove gradients from the input."
            )
        # # Check metadata
        # if not (
        #     _dispatch.all(tensor.keys.names[:2] == ["spherical_harmonics_l", "species_center"])
        #     or _dispatch.all(
        #         tensor.keys.names[:3]
        #         == ["spherical_harmonics_l", "species_center", "species_neighbor"]
        #     )
        # ):
        #     raise ValueError(
        #         "input tensors must have key names"
        #         ' ["spherical_harmonics_l", "species_center"] or'
        #         ' ["spherical_harmonics_l", "species_center", "species_neighbor"]'
        #         ' as the first two or three keys'
        #     )
        if not _dispatch.all(tensor.component_names == ["spherical_harmonics_m"]):
            raise ValueError(
                "input tensors must have a single component"
                " axis with name `spherical_harmonics_m`"
            )
    # tensor_1 = _clebsch_gordan._standardize_keys(tensor_1)
    # tensor_2 = _clebsch_gordan._standardize_keys(tensor_2)

    # Parse the selected keys
    selected_keys = _clebsch_gordan._parse_selected_keys(
        n_iterations=1,
        selected_keys=selected_keys,
        like=tensor_1.keys.values,
    )

    # Pre-compute the keys needed to perform each CG tensor product
    key_metadata = _clebsch_gordan._precompute_keys(
        tensor_1.keys,
        tensor_2.keys,
        n_iterations=1,
        selected_keys=selected_keys,
        skip_redundant=[False],
    )[0]

    # Compute CG coefficient cache
    if compute_metadata_only:
        cg_cache = None
    else:
        angular_max = max(
            _dispatch.concatenate(
                [tensor_1.keys.column("spherical_harmonics_l")]
                + [tensor_2.keys.column("spherical_harmonics_l")]
                + [key_metadata[2].column("spherical_harmonics_l")]
            )
        )
        # TODO: keys have been precomputed, so perhaps we don't need to
        # compute all CG coefficients up to angular_max here.
        # TODO: use sparse cache by default until we understand under which
        # circumstances (and if) dense is faster.
        cg_cache = _cg_cache.ClebschGordanReal(angular_max, sparse=sparse)

    # Perform CG tensor product by combining block pairs
    blocks_out = []
    for key_1, key_2, key_out in zip(*key_metadata):
        block_out = _clebsch_gordan._combine_blocks_same_samples(
            tensor_1[key_1],
            tensor_2[key_2],
            key_out["spherical_harmonics_l"],
            cg_cache,
            compute_metadata_only=compute_metadata_only,
        )
        blocks_out.append(block_out)

    # Build the TensorMap
    return TensorMap(keys=key_metadata[2], blocks=blocks_out)
