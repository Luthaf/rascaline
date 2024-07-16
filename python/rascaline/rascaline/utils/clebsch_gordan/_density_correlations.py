"""
This module provides convenience calculators for preforming density correlations, i.e.
the (iterative) CG tensor products of density (body order 2) tensors.

All of these calculators wrap the :py:class:`TensorCorrelator` class, handling the
higher-level metadata manipulation to produce the desired output tensors.
"""

from typing import List, Optional, Tuple

from .. import _dispatch
from .._backend import (
    Labels,
    TensorMap,
    TorchModule,
    operations,
    torch_jit_export,
)
from . import _utils
from ._tensor_correlator import TensorCorrelator


class DensityCorrelations(TorchModule):
    """
    Iterative products of a density to form higher arbitrary body order tensors.

    :param max_angular: :py:class:`int`, the maximum angular momentum to compute CG
        coefficients for.
    :param arrays_backend: :py:class:`str`, the backend to use for array operations. If
        ``None``, the backend is automatically selected based on the environment.
        Possible values are "numpy" and "torch".
    :param cg_backend: :py:class:`str`, the backend to use for the CG tensor product. If
        ``None``, the backend is automatically selected based on the arrays backend.
    """
    def __init__(
        self,
        n_correlations: int,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = False,
        *,
        tensor_correlator: Optional[TensorCorrelator] = None,
        max_angular: Optional[int] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__()

        self._n_correlations = n_correlations
        self._angular_cutoff = angular_cutoff
        self._skip_redundant = skip_redundant

        # Initialize the TensorCorrelator calculator if not provided
        if tensor_correlator is None:
            if max_angular is None:
                raise ValueError(
                    "If ``tensor_correlator`` is not provided, ``max_angular`` must be."
                )
            if angular_cutoff is not None:
                max_angular = min(max_angular, angular_cutoff)

            self._tensor_correlator = TensorCorrelator(
                max_angular=max_angular,
                arrays_backend=arrays_backend,
                cg_backend=cg_backend,
            )
        else:
            if _dispatch.any(
                [param is not None for param in [arrays_backend, cg_backend]]
            ):
                raise ValueError(
                    "If ``tensor_correlator`` is provided, ``arrays_backend`` and "
                    " ``cg_backend`` should be None."
                )
            self._tensor_correlator = tensor_correlator

    def forward(
        self,
        tensor: TensorMap,
        density: TensorMap,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            tensor,
            density,
            selected_keys,
        )

    @torch_jit_export
    def compute_metadata(
        self,
        tensor: TensorMap,
        density: TensorMap,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._density_correlations(
            tensor,
            density,
            selected_keys,
            compute_metadata=True,
        )

    def compute(
        self,
        tensor: TensorMap,
        density: TensorMap,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Takes ``n_correlations`` of iterative CG tensor products of a tensor with a
        density.

        .. math::

            \\T^{\\nu=\\nu'+n_{corr}} = T^{\\nu=\\nu'}
                                        \\otimes \\rho^{\\nu=1} \\ldots \\otimes
                                        \\rho^{\\nu=1}

        where T is the input ``tensor`` of arbitrary correlation order \\nu' and \\rho
        is the input ``density`` tensor of correlation order 1 (body order 2).

        As the density is by definition a correlation order 1 tensor, the correlation
        order of ``tensor`` will be increased by ``n_correlations`` from its original
        correlation order.

        ``tensor`` and ``density`` must have metadata that is compatible for a CG tensor
        product by the :py:class:`TensorCorrelator` class. For every iteration after the
        first, the property dimension names of ``density`` are incremented numerically
        by 1 so that the metadata is compatible for the next tensor product.

        ``selected_keys`` can be passed to select the keys to compute on the final
        iteration. If ``None``, all keys are computed. To limit the maximum angular
        momenta to compute on **intermediate** iterations, pass ``angular_cutoff``.

        If ``angular_cutoff`` and ``selected_keys`` are both passed, ``angular_cutoff``
        is ignored on the final iteration.

        ``skip_redundant`` can be passed to skip redundant computations on intermediate
        iterations.

        :param tensor: :py:class:`TensorMap`, the input tensor of arbitrary correlation
            order.
        :param density: :py:class:`TensorMap`, the input density tensor of correlation
            order 1.
        :param selected_keys: :py:class:`Labels`, the keys to compute on the final
            iteration. If ``None``, all keys are computed.

        """
        return self._density_correlations(
            tensor,
            density,
            selected_keys,
            compute_metadata=False,
        )

    def _density_correlations(
        self,
        tensor: TensorMap,
        density: TensorMap,
        selected_keys: Optional[Labels] = None,
        compute_metadata: bool = False,
    ) -> TensorMap:
        """
        Computes the density correlations.
        """
        # Parse selection filters
        selected_keys, angular_cutoff = _parse_selection_filters(
            self._n_correlations, selected_keys, self._angular_cutoff
        )

        # Perform iterative CG tensor products
        new_lambda_names = []
        density_correlations = tensor
        for i_correlation in range(self._n_correlations):

            # Rename density property dimensions
            if i_correlation > 0:  # metadata assumed ok on first iteration
                for name in density.property_names:
                    density = operations.rename_dimension(
                        density,
                        "properties",
                        name,
                        _utils._increment_numeric_suffix(name),
                    )

            # Define new key dimension names for tracking intermediate correlations
            if i_correlation == 0:
                o3_lambda_1_name = f"l_{i_correlation + 1}"
            else:
                o3_lambda_1_name = f"k_{i_correlation + 1}"
            o3_lambda_2_name = f"l_{i_correlation + 2}"
            new_lambda_names.extend([o3_lambda_1_name, o3_lambda_2_name])

            # Compute CG tensor product
            density_correlations = self._tensor_correlator._cg_tensor_product(
                density_correlations,
                density,
                o3_lambda_1_name,
                o3_lambda_2_name,
                selected_keys=selected_keys[i_correlation],
                angular_cutoff=angular_cutoff[i_correlation],
                skip_redundant=self._skip_redundant,
                compute_metadata=compute_metadata,
            )

        return density_correlations


def _parse_selection_filters(
    n_correlations: int,
    selected_keys: Optional[Labels],
    angular_cutoff: Optional[int],
) -> Tuple[List]:

    # Parse selected_keys
    selected_keys_ = [None] * (n_correlations - 1)
    selected_keys_ += [selected_keys]

    # Parse angular_cutoff and selected_keys
    angular_cutoff_ = [angular_cutoff] * (n_correlations - 1)
    if selected_keys is None:
        angular_cutoff_ += [angular_cutoff]
    else:
        angular_cutoff_ += [None]

    return selected_keys_, angular_cutoff_
