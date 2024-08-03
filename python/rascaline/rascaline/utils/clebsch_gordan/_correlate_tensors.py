"""
Module for computing a Clebsch-Gordan tensor product between a tensor of arbitrary body
order and a density (i.e. correlation order 1) in TensorMap form, where the samples are
different.
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
    operations,
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


class CorrelateTensorWithDensity(TorchModule):
    """
    Takes a single Clebsch-Gordan (CG) tensor products of a tensor descriptor with a
    density descriptor. Returns :py:class:`TensorMap` corresponding to the correlated
    output.

    TODO: edit docstring

    The input density descriptor necessarily is body order 2 (i.e. correlation order 1),
    but can be single- or multi-center. The output is a :py:class:`list` of density
    correlations for each iteration specified in ``output_selection``, up to the target
    body order passed in ``body_order``. By default only the output tensor from the last
    correlation step is returned.

    Key dimensions can be matched with the ``match_keys`` argument. Products are only
    taken between pairs of keys with equal values in the specified dimensions.

    Global selections on the maximum angular channel computed can be set with the
    ``angular_cutoff`` argument, while ``selected_keys`` allows control over computation
    of specific combinations of angular and parity channels.

    :param max_angular: The maximum angular order for which CG coefficients should be
        computed and stored. This must be large enough to cover the maximum angular
        order reached in the CG iterations on a density input to the :py:meth:`compute`
        method.
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

    _selected_keys: Union[Labels, None]

    def __init__(
        self,
        max_angular: int,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Labels] = None,
        match_keys: Optional[List[str]] = None,
        match_samples: Optional[List[str]] = None,
        keep_l_in_keys: Optional[bool] = False,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
        cg_coefficients: Optional[TensorMap] = None,
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
        if cg_coefficients is None:  # compute CG coefficients
            self._cg_coefficients = _coefficients.calculate_cg_coefficients(
                lambda_max=self._max_angular,
                cg_backend=self._cg_backend,
                use_torch=(arrays_backend == "torch"),
            )
        else:  # using a pre-computed set of CG coefficients
            self._cg_coefficients = cg_coefficients

        # Parse the selected keys
        self._angular_cutoff = angular_cutoff
        if arrays_backend == "torch":
            array_like = torch.empty(0)
        elif arrays_backend == "numpy":
            array_like = np.empty(0)
        self._selected_keys: List[Union[Labels, None]] = _utils.parse_selected_keys(
            n_iterations=1,
            array_like=array_like,
            angular_cutoff=self._angular_cutoff,
            selected_keys=selected_keys,
        )[
            0
        ]  # only 1 iteration
        if match_keys is None:
            match_keys = []
        self._match_keys = match_keys
        if match_samples is None:
            match_samples = []
        self._match_samples = match_samples

        # Parse the bool flags that are applied at each iteration
        self._keep_l_in_keys = _utils.parse_bool_iteration_filter(
            n_iterations=1, bool_filter=keep_l_in_keys, filter_name="keep_l_in_keys"
        )[
            0
        ]  # only 1 iteration

    def forward(self, tensor: TensorMap, density: TensorMap) -> TensorMap:
        """
        Calls the :py:meth:`CorrelateTensorWithDensity.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.
        """
        return self.compute(density)

    def compute(self, tensor: TensorMap, density: TensorMap) -> TensorMap:
        """
        Computes the density correlations by taking iterative Clebsch-Gordan (CG) tensor
        products of the input `density` descriptor with itself.

        :param density: A density descriptor of body order 2 (correlation order 1), in
            :py:class:`TensorMap` format. This may be, for example, a rascaline
            :py:class:`SphericalExpansion` or :py:class:`LodeSphericalExpansion`.
            Alternatively, this could be multi-center descriptor, such as a pair
            density.
        """
        return self._correlate_tensor_with_density(
            tensor,
            density,
            compute_metadata=False,
        )

    @torch_jit_export
    def compute_metadata(self, tensor: TensorMap, density: TensorMap) -> TensorMap:
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
        return self._correlate_tensor_with_density(
            tensor,
            density,
            compute_metadata=True,
        )

    # ====================================================================
    # ===== Private functions that do the work on the TensorMap level
    # ====================================================================
    def _correlate_tensor_with_density(
        self, tensor: TensorMap, density: TensorMap, compute_metadata: bool
    ) -> TensorMap:

        # Check metadata - the 'standard' keys
        if tensor.keys.names[:3] != ["order_nu", "o3_lambda", "o3_sigma"]:
            raise ValueError(
                "the first three key dimensions of `tensor` must 'order_nu',"
                " 'o3_lambda', and 'o3_sigma'"
            )
        if density.keys.names[:3] != ["order_nu", "o3_lambda", "o3_sigma"]:
            raise ValueError(
                "the first three key dimensions of `tensor` must 'order_nu',"
                " 'o3_lambda', and 'o3_sigma'"
            )

        # Check components
        if not tensor.component_names == ["o3_mu"]:
            raise ValueError(
                "input `tensor` must have a single component" " axis with name `o3_mu`"
            )
        if not density.component_names == ["o3_mu"]:
            raise ValueError(
                "input `density` must have a single component" " axis with name `o3_mu`"
            )

        # TODO: implement combinations of gradients too.
        # We have to create a bool array with dispatch to be TorchScript compatible
        contains_gradients_tensor = all(
            [len(list(block.gradients())) > 0 for _, block in tensor.items()]
        )
        contains_gradients_density = all(
            [len(list(block.gradients())) > 0 for _, block in density.items()]
        )
        if contains_gradients_tensor or contains_gradients_density:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented. "
                "Use `metatensor.remove_gradients` to remove gradients from the input."
            )

        max_angular = max(
            _dispatch.max(density.keys.column("o3_lambda")),
            _dispatch.max(density.keys.column("o3_lambda")),
        )
        if max_angular > self._max_angular:
            raise ValueError(
                "the largest `o3_lambda` in the density to correlate is "
                f"{max_angular}, but this class was initialized with "
                f"`max_angular={self._max_angular}`"
            )

        # Perform iterative CG tensor products
        # density_correlations: List[TensorMap] = []
        if compute_metadata:
            cg_backend = "metadata"
        else:
            cg_backend = self._cg_backend

        cg_coefficients = self._cg_coefficients.to(dtype=tensor.dtype)

        # Compute the keys from all combinations
        new_keys, combinations = _utils.precompute_keys(
            tensor.keys,
            density.keys,
            selected_keys=self._selected_keys,
            match_keys=self._match_keys,
            skip_redundant=False,
        )

        # Check that some keys are produced as a result of the combination
        if len(new_keys) == 0:
            raise ValueError(
                f"invalid selections: combination produces no"
                " valid combinations. Check the `angular_cutoff` and"
                " `selected_keys` args and try again."
            )

        # Check that the maximum angular order is not exceeded
        max_angular = _dispatch.max(new_keys.column("o3_lambda"))
        if max_angular > self._max_angular:
            raise ValueError(
                "correlations of this density would require a `max_angular` of "
                f"{max_angular}, but this class was initialized with "
                f"`max_angular={self._max_angular}`"
            )

        # Do the CG combinations and build the new TensorMap
        new_blocks: List[TensorBlock] = []
        for combination in combinations:
            new_blocks.extend(
                _utils.cg_tensor_product_blocks_different_samples(
                    tensor.block(combination.first),
                    density.block(combination.second),
                    self._match_samples,
                    combination.o3_lambdas,
                    cg_coefficients,
                    cg_backend,
                )
            )
        tensor_correlation = TensorMap(keys=new_keys, blocks=new_blocks)

        # Move the "l" and "k" keys to properties if requested
        new_correlation_order = tensor_correlation.keys.column("order_nu")[0]
        if not self._keep_l_in_keys:
            tensor_correlation = tensor_correlation.keys_to_properties(
                [f"l_{i}" for i in range(1, new_correlation_order + 1)]
                + [f"k_{i}" for i in range(2, new_correlation_order)]
            )

        return tensor_correlation
