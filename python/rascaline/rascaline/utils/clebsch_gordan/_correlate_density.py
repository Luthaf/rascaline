"""
Module for computing Clebsch-gordan tensor product iterations on density (i.e.
correlation order 1) tensors in TensorMap form, where the samples are
equivalent.
"""

from typing import List, Optional, Union

import numpy as np

from .. import _dispatch
from .._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    TorchModule,
    TorchScriptClass,
    torch_jit_export,
    torch_jit_is_scripting,
)
from . import _coefficients, _utils


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ======================================================================
# ===== Public API functions
# ======================================================================


class DensityCorrelations(TorchModule):
    """
    Takes iterative Clebsch-Gordan (CG) tensor products of a density descriptor with
    itself up to the desired body order. Returns :py:class:`TensorMap`
    corresponding to the density correlations output from the specified iteration(s).

    The input density descriptor necessarily is body order 2 (i.e. correlation order 1),
    but can be single- or multi-center. The output is a :py:class:`list` of density
    correlations for each iteration specified in ``output_selection``, up to the target
    body order passed in ``body_order``. By default only the output tensor from the last
    correlation step is returned.

    Key dimensions can be matched with the ``match_keys`` argument. Products are only
    taken between pairs of keys with equal values in the specified dimensions.

    As a density is being correlated with itself, some redundant CG tensor products can
    be skipped with the ``skip_redundant`` keyword.

    Global selections on the maximum angular channel computed at each iteration can be
    set with the ``angular_cutoff`` argument, while ``selected_keys`` allows control
    over computation of specific combinations of angular and parity channels.

    :param max_angular: The maximum angular order for which CG coefficients should be
        computed and stored. This must be large enough to cover the maximum angular
        order reached in the CG iterations on a density input to the :py:meth:`compute`
        method.
    :param body_order: The desired correlation order of the output descriptor. Must be
        >= 2. The resulting final :py:class:`TensorMap` will be of body-order
        ``body_order``, or correlation order ``body_order - 1``.
    :param angular_cutoff: The maximum angular channel to compute at any given CG
        iteration, applied globally to all iterations until the target body order is
        reached.
    :param selected_keys: :py:class:`Labels` or list of :py:class:`Labels` specifying
        the angular and/or parity channels to output at each iteration. All
        :py:class:`Labels` objects passed here must only contain key names
        ``"o3_lambda"`` and ``"o3_sigma"``. If a single :py:class:`Labels` object is
        given, this is applied to the final iteration only. If a list of
        :py:class:`Labels` is given, each is applied to its corresponding iteration. If
        None is passed, all angular and parity channels are kept at each iteration, with
        the global ``angular_cutoff`` applied if specified.
    :param match_keys: A :py:class:`list` of :py:class:`str` specifying the names of key
        dimensions to match when performing CG tensor products. At each iteration, only
        products of keys with equal values in these specified dimensions are computed.
    :param skip_redundant: Whether to skip redundant CG combinations. Defaults to
        ``True``, which means all combinations are performed. If a :py:class:`list` of
        :py:class:`bool` is passed, this is applied to each iteration. If a single
        :py:class:`bool` is passed, this is applied to all iterations. Note that when
        redundant combinations are skipped in a CG tensor product step, the norm of the
        output tensor is not preserved. To preserve the norm, set this flag to
        ``False``.
    :param output_selection: A :py:class:`list` of :py:class:`bool` specifying whether
        to output a :py:class:`TensorMap` for each iteration. If a single
        :py:class:`bool` is passed as ``True``, outputs from all iterations will be
        returned. If a :py:class:`list` of :py:class:`bool` is passed, this controls the
        output at each corresponding iteration. If None is passed, only the final
        iteration is output.
    :param keep_l_in_keys: should the metadata that tracks values of angular momenta
        that were combined together be kept in the keys or moved to the properties?

        Keys named ``l_{i}`` correspond to the input ``components``, with ``l_1`` being
        the last entry in ``components`` and ``l_N`` the first one. Keys named ``k_{i}``
        correspond to intermediary spherical components created during the calculation,
        i.e. a ``k_{i}`` used to be ``o3_lambda``.

        This defaults to ``False`` for :py:class:`TensorMap` output at each requested
        iteration, such that the keys are moved to properties. If you want to use the
        output of this class for further CG tensor products, this should be set to
        ``True``.
    :param arrays_backend: Determines the array backend, either ``"numpy"`` or
        ``"torch"``.
    :param cg_backend: Determines the backend for the CG combination. It can be
        ``"python-sparse"``, or ``"python-dense"``. If the CG combination performs on
        the sparse coefficients, it means that for each ``(l1, l2, lambda)`` block the
        ``(m1, m2, mu)`` coefficients are stored in a sparse format only storing the
        nonzero coefficients. If this is not given, the most optimal choice is
        determined given available packages and ``arrays_backend``.

        - ``"python-dense"``: Uses the python implementation performing the combinations
          with the dense CG coefficients.
        - ``"python-sparse"``: Uses the python implementation performing the
          combinations with the sparse CG coefficients.

    :return: A :py:class:`list` of :py:class:`TensorMap` corresponding to the density
        correlations output from the specified iterations. If the output from a single
        iteration is requested, a :py:class:`TensorMap` is returned instead.
    """

    _selected_keys: List[Union[Labels, None]]

    def __init__(
        self,
        max_angular: int,
        body_order: int,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Union[Labels, List[Labels]]] = None,
        match_keys: Optional[List[str]] = None,
        skip_redundant: Optional[Union[bool, List[bool]]] = True,
        output_selection: Optional[Union[bool, List[bool]]] = None,
        keep_l_in_keys: Optional[Union[bool, List[bool]]] = False,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:
        super().__init__()
        if arrays_backend is None:
            if torch_jit_is_scripting():
                arrays_backend = "torch"
            else:
                if isinstance(Labels, TorchScriptClass):
                    arrays_backend = "torch"
                else:
                    arrays_backend = "numpy"
        elif arrays_backend == "numpy":
            if torch_jit_is_scripting():
                raise ValueError(
                    "Module is torch scripted but 'numpy' was given as `arrays_backend`"
                )
            arrays_backend = "numpy"
        elif arrays_backend == "torch":
            arrays_backend = "torch"
        else:
            raise ValueError(
                f"Unknown `arrays_backend` {arrays_backend}."
                "Only 'numpy' and 'torch' are supported."
            )

        # Choosing the optimal cg combine backend
        if cg_backend is None:
            if arrays_backend == "torch":
                self._cg_backend = "python-dense"
            else:
                self._cg_backend = "python-sparse"
        else:
            self._cg_backend = cg_backend

        if max_angular < 0:
            raise ValueError(
                f"Given `max_angular={max_angular}` negative. "
                "Must be greater equal 0."
            )
        self._max_angular = max_angular
        self._cg_coefficients = _coefficients.calculate_cg_coefficients(
            lambda_max=self._max_angular,
            cg_backend=self._cg_backend,
            use_torch=(arrays_backend == "torch"),
        )

        # Check inputs
        if body_order <= 2:
            raise ValueError("`body_order` must be > 2")
        correlation_order = body_order - 1  # use correlation order internally
        self._correlation_order = correlation_order

        n_iterations = correlation_order - 1  # num iterations

        # Parse the selected keys
        self._angular_cutoff = angular_cutoff
        if arrays_backend == "torch":
            array_like = torch.empty(0)
        elif arrays_backend == "numpy":
            array_like = np.empty(0)
        self._selected_keys: List[Union[Labels, None]] = _utils.parse_selected_keys(
            n_iterations=n_iterations,
            array_like=array_like,
            angular_cutoff=self._angular_cutoff,
            selected_keys=selected_keys,
        )
        if match_keys is None:
            match_keys = []
        self._match_keys = match_keys

        # Parse the bool flags that are applied at each iteration
        self._skip_redundant = _utils.parse_bool_iteration_filter(
            n_iterations, skip_redundant, "skip_redundant"
        )
        self._output_selection = _utils.parse_bool_iteration_filter(
            n_iterations, output_selection, "output_selection"
        )
        self._keep_l_in_keys = _utils.parse_bool_iteration_filter(
            n_iterations, keep_l_in_keys, "keep_l_in_keys"
        )

    def forward(self, density: TensorMap) -> Union[TensorMap, List[TensorMap]]:
        """
        Calls the :py:meth:`DensityCorrelations.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.
        """
        return self.compute(density)

    def compute(self, density: TensorMap) -> Union[TensorMap, List[TensorMap]]:
        """
        Computes the density correlations by taking iterative Clebsch-Gordan (CG) tensor
        products of the input `density` descriptor with itself.

        :param density: A density descriptor of body order 2 (correlation order 1), in
            :py:class:`TensorMap` format. This may be, for example, a rascaline
            :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
            Alternatively, this could be multi-center descriptor, such as a pair
            density.
        """
        return self._correlate_density(
            density,
            compute_metadata=False,
        )

    @torch_jit_export
    def compute_metadata(
        self,
        density: TensorMap,
    ) -> Union[TensorMap, List[TensorMap]]:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        :param density: A density descriptor of body order 2 (correlation order 1), in
            :py:class:`TensorMap` format. This may be, for example, a rascaline
            :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
            Alternatively, this could be multi-center descriptor, such as a pair
            density.
        """
        return self._correlate_density(
            density,
            compute_metadata=True,
        )

    # ====================================================================
    # ===== Private functions that do the work on the TensorMap level
    # ====================================================================
    def _correlate_density(
        self, density: TensorMap, compute_metadata: bool
    ) -> Union[TensorMap, List[TensorMap]]:

        n_iterations = self._correlation_order - 1  # num iterations
        density = _utils.standardize_keys(density)  # standardize metadata

        # Check metadata
        assert density.keys.names[:3] == [
            "order_nu",
            "o3_lambda",
            "o3_sigma",
        ]  # the 'standard' keys

        # As we are combining a density with itself, all other
        # dimensions must be 'matched' dimensions. If they are not, raise an error.
        for name in density.keys.names[3:]:
            if name not in self._match_keys:
                raise ValueError(
                    f"dimension '{name}' in `density` is not a 'match_key' dimension."
                    " As this function performs self-correlations, any key dimensions"
                    " that are not 'o3_lambda' or 'o3_sigma' must be matched."
                )
        # Check components
        if not density.component_names == ["o3_mu"]:
            raise ValueError(
                "input `density` must have a single component" " axis with name `o3_mu`"
            )

        density_correlation = density  # create a copy to combine with itself

        # TODO: implement combinations of gradients too
        # we have to create a bool array with dispatch to be TorchScript compatible
        contains_gradients = all(
            [len(list(block.gradients())) > 0 for _, block in density.items()]
        )
        if contains_gradients:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented. "
                "Use `metatensor.remove_gradients` to remove gradients from the input."
            )

        max_angular = _dispatch.max(density.keys.column("o3_lambda"))
        if max_angular > self._max_angular:
            raise ValueError(
                "the largest `o3_lambda` in the density to correlate is "
                f"{max_angular}, but this class was initialized with "
                f"`max_angular={self._max_angular}`"
            )

        # Perform iterative CG tensor products
        density_correlations: List[TensorMap] = []
        if compute_metadata:
            cg_backend = "metadata"
        else:
            cg_backend = self._cg_backend

        cg_coefficients = self._cg_coefficients.to(dtype=density.dtype)

        keys_iter = density.keys
        for iteration in range(n_iterations):
            # Define the correlation order of the current iteration
            correlation_order_it = iteration + 2

            # check which keys will need to be combined
            keys_iter, combinations = _utils.precompute_keys(
                keys_iter,
                density.keys,
                selected_keys=self._selected_keys[iteration],
                match_keys=self._match_keys,
                skip_redundant=self._skip_redundant[iteration],
            )

            # Check that some keys are produced as a result of the combination
            if len(keys_iter) == 0:
                raise ValueError(
                    f"invalid selections: iteration {iteration + 1} produces no"
                    " valid combinations. Check the `angular_cutoff` and"
                    " `selected_keys` args and try again."
                )

            max_angular = _dispatch.max(keys_iter.column("o3_lambda"))
            if max_angular > self._max_angular:
                raise ValueError(
                    "correlations of this density would require a `max_angular` of "
                    f"{max_angular}, but this class was initialized with "
                    f"`max_angular={self._max_angular}`"
                )

            blocks_iter: List[TensorBlock] = []
            for combination in combinations:
                blocks_iter.extend(
                    _utils.cg_tensor_product_blocks_same_samples(
                        density_correlation.block(combination.first),
                        density.block(combination.second),
                        combination.o3_lambdas,
                        cg_coefficients,
                        cg_backend,
                    )
                )

            density_correlation = TensorMap(keys=keys_iter, blocks=blocks_iter)

            # If this tensor is to be included in the output, and if requested, move the
            # [l1, l2, ...] keys to properties and store
            if self._output_selection[iteration]:
                if self._keep_l_in_keys[iteration]:
                    density_correlations.append(density_correlation)
                else:  # move 'l' and 'k' keys to properties
                    density_correlations.append(
                        density_correlation.keys_to_properties(
                            [f"l_{i}" for i in range(1, correlation_order_it + 1)]
                            + [f"k_{i}" for i in range(2, correlation_order_it)]
                        )
                    )

        # Replace "order_nu" key dimension used internally with "body_order"
        for i, tensor in enumerate(density_correlations):
            keys = tensor.keys
            tmp_body_order = tensor.keys.column("order_nu") + 1
            assert len(_dispatch.unique(tmp_body_order)) == 1
            keys = keys.insert(name="body_order", values=tmp_body_order, index=0)
            keys = keys.remove(name="order_nu")
            density_correlations[i] = TensorMap(
                keys=keys, blocks=[b.copy() for b in tensor.blocks()]
            )

        # Return a single TensorMap in the simple case
        if len(density_correlations) == 1:
            return density_correlations[0]

        # Otherwise return a list of TensorMaps
        return density_correlations
