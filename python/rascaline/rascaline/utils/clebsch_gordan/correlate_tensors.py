"""
Module for computing the Clebsch-gordan tensor product on arbitrary tensors in
TensorMap form. For a special iterative case specifically for densities, see the
:py:mod:`correlate_density` module.
"""
from typing import List, Optional, Union

from metatensor import Labels, TensorMap


# from . import _cg_cache, _clebsch_gordan, _dispatch


# ======================================================================
# ===== Public API functions
# ======================================================================


def correlate_tensors(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
) -> TensorMap:
    """
    Takes a Clebsch-Gordan (CG) tensor product of two tensors in
    :py:class:`TensorMap` form, and returns the result in :py:class:`TensorMap`.

    This function is an iterative special case of the more general
    :py:mod:`correlate_tensors`. As a density is being correlated with itself,
    some redundant CG tensor products can be skipped with the `skip_redundant`
    keyword.

    :param tensor_1: the first tensor to correlate.
    :param tensor_2: the second tensor to correlate.
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

    :return: A :py:class:`TensorMap` corresponding to the correlated tensors.
    """
    return _correlate_tensors(
        tensor_1,
        tensor_2,
        angular_cutoff,
        selected_keys,
        compute_metadata_only=False,
        sparse=True,  # sparse CG cache by default
    )


def correlate_tensors_metadata(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    angular_cutoff: Optional[int] = None,
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
        angular_cutoff,
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
    angular_cutoff: Optional[int] = None,
    selected_keys: Optional[Union[Labels, List[Labels]]] = None,
    compute_metadata_only: bool = False,
    sparse: bool = True,
) -> TensorMap:
    """
    Performs the Clebsch-Gordan tensor product for public functions
    :py:func:`correlate_tensors` and :py:func:`correlate_tensors_metadata`.
    """
    raise NotImplementedError("TODO!")
