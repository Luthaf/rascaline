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
    LabelsEntry,
    TensorBlock,
    TensorMap,
    TorchModule,
    TorchScriptClass,
    torch_jit_export,
    torch_jit_is_scripting,
)
from . import _cg_cache, _clebsch_gordan


try:
    import mops  # noqa F401

    HAS_MOPS = True
except ImportError:
    HAS_MOPS = False

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
    itself up to the desired correlation order. Returns :py:class:`TensorMap`
    corresponding to the density correlations output from the specified iteration(s).

    A density descriptor necessarily is body order 2 (i.e. correlation order 1), but can
    be single- or multi-center. The output is a :py:class:`list` of density correlations
    for each iteration specified in `output_selection`, up to the target order passed in
    `correlation_order`. By default only the last correlation (i.e. the correlation of
    order ``correlation_order``) is returned.

    This function is an iterative special case of the more general
    :py:func:`correlate_tensors`. As a density is being correlated with itself, some
    redundant CG tensor products can be skipped with the `skip_redundant` keyword.

    Selections on the angular and parity channels at each iteration can also be
    controlled with arguments `angular_cutoff`, `angular_selection` and
    `parity_selection`.

    :param max_angular: The maximum angular order for which CG coefficients should be
        computed and stored. This must be large enough to cover the maximum angular
        order reached in the CG iterations on a density input to the :py:meth:`compute`
        method.
    :param correlation_order: The desired correlation order of the output descriptor.
        Must be >= 1.
    :param angular_cutoff: The maximum angular channel to compute at any given CG
        iteration, applied globally to all iterations until the target correlation order
        is reached.
    :param selected_keys: :py:class:`Labels` or `List[:py:class:`Labels`]` specifying
        the angular and/or parity channels to output at each iteration. All
        :py:class:`Labels` objects passed here must only contain key names
        "o3_lambda" and "o3_sigma". If a single :py:class:`Labels`
        object is passed, this is applied to the final iteration only. If a
        :py:class:`list` of :py:class:`Labels` objects is passed, each is applied to its
        corresponding iteration. If None is passed, all angular and parity channels are
        output at each iteration, with the global `angular_cutoff` applied if specified.
    :param skip_redundant: Whether to skip redundant CG combinations. Defaults to False,
        which means all combinations are performed. If a :py:class:`list` of
        :py:class:`bool` is passed, this is applied to each iteration. If a single
        :py:class:`bool` is passed, this is applied to all iterations.
    :param output_selection: A :py:class:`list` of :py:class:`bool` specifying whether
        to output a :py:class:`TensorMap` for each iteration. If a single
        :py:class:`bool` is passed as True, outputs from all iterations will be
        returned. If a :py:class:`list` of :py:class:`bool` is passed, this controls the
        output at each corresponding iteration. If None is passed, only the final
        iteration is output.
    :param arrays_backend: Determines the array backend, either "numpy" or "torch".
    :param cg_backend: Determines the backend for the CG combination. It can be even
        "python-sparse", "python-dense" or "mops". If the CG combination performs on the
        sparse coefficients, it means that for each (l1, l2, lambda) block the (m1, m2,
        mu) coefficients are stored in a sparse format only storing the nonzero
        coefficients. If the parameter are None, the most optimal choice is determined
        given available packages and ``arrays_backend``.

        - "python-dense": Uses the python implementation performing the combinations
          with the dense CG coefficients.
        - "python-sparse": Uses the python implementation performing the
          combinations with the sparse CG coefficients.
        - "mops": Uses the ``mops`` package to optimize the sparse combinations. At
          the moment it is only available with ``arrays_backend="numpy"``

    :return: A :py:class:`list` of :py:class:`TensorMap` corresponding to the density
        correlations output from the specified iterations. If the output from a single
        iteration is requested, a :py:class:`TensorMap` is returned instead.
    """

    _selected_keys: List[Union[Labels, None]]

    def __init__(
        self,
        max_angular: int,
        correlation_order: int,
        angular_cutoff: Optional[int] = None,
        selected_keys: Optional[Union[Labels, List[Labels]]] = None,
        skip_redundant: Optional[Union[bool, List[bool]]] = False,
        output_selection: Optional[Union[bool, List[bool]]] = None,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ):
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
            if arrays_backend == "numpy" and HAS_MOPS:
                self._cg_backend = "mops"
            else:
                self._cg_backend = "python-sparse"
        elif cg_backend == "python-dense":
            self._cg_backend = "python-dense"
        elif cg_backend == "python-sparse":
            self._cg_backend = "python-sparse"
        elif cg_backend == "mops":
            if arrays_backend == "torch":
                raise NotImplementedError(
                    "'torch' was determined or given as `arrays_backend` "
                    "and 'mops' was given as `cg_backend`, "
                    "but mops does not support torch backend yet"
                )
            else:
                assert arrays_backend == "numpy"
                if not HAS_MOPS:
                    raise ImportError(
                        "mops is not installed, but 'mops' was given as `cg_backend`"
                    )
                self._cg_backend = "mops"

        else:
            raise ValueError(
                f"Unknown `cg_backend` {cg_backend}."
                "Only 'python-dense', 'python-sparse' and 'mops' are supported."
            )

        if max_angular < 0:
            raise ValueError(
                f"Given `max_angular={max_angular}` negative. "
                "Must be greater equal 0."
            )
        self._max_angular = max_angular
        self._cg_coefficients = _cg_cache.calculate_cg_coefficients(
            lambda_max=self._max_angular,
            sparse=(self._cg_backend in ["python-sparse", "mops"]),
            use_torch=(arrays_backend == "torch"),
        )

        # Check inputs
        if correlation_order <= 1:
            raise ValueError("`correlation_order` must be > 1")
        self._correlation_order = correlation_order

        n_iterations = correlation_order - 1  # num iterations
        # Parse the selected keys
        self._angular_cutoff = angular_cutoff

        if arrays_backend == "torch":
            array_like = torch.empty(0)
        elif arrays_backend == "numpy":
            array_like = np.empty(0)

        self._selected_keys: List[Union[Labels, None]] = (
            _clebsch_gordan._parse_selected_keys(
                n_iterations=n_iterations,
                array_like=array_like,
                angular_cutoff=self._angular_cutoff,
                selected_keys=selected_keys,
            )
        )
        # Parse the bool flags that control skipping of redundant CG combinations
        # and TensorMap output from each iteration
        self._skip_redundant, self._output_selection = (
            _clebsch_gordan._parse_bool_iteration_filters(
                n_iterations,
                skip_redundant=skip_redundant,
                output_selection=output_selection,
            )
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

        # Check metadata
        if not (
            density.keys.names == ["o3_lambda", "o3_sigma", "center_type"]
            or density.keys.names
            == ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]
        ):
            raise ValueError(
                "input `density` must have key names"
                " ['o3_lambda', 'o3_sigma', 'center_type'] or"
                " ['o3_lambda', 'o3_sigma', 'center_type', 'neighbor_type']"
            )
        if not density.component_names == ["o3_mu"]:
            raise ValueError(
                "input `density` must have a single component" " axis with name `o3_mu`"
            )
        n_iterations = self._correlation_order - 1  # num iterations
        density = _clebsch_gordan._standardize_keys(density)  # standardize metadata
        density_correlation = density  # create a copy to combine with itself

        # TODO: implement combinations of gradients too
        # we have to create a bool array with dispatch to be TorchScript compatible
        contains_gradients = all(
            [len(list(block.gradients())) > 0 for _, block in density.items()]
        )
        if contains_gradients:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented."
                " Use metatensor.remove_gradients to remove gradients from the input."
            )
        # Pre-compute the keys needed to perform each CG iteration
        key_metadata = _clebsch_gordan._precompute_keys(
            density.keys,
            density.keys,
            n_iterations=n_iterations,
            selected_keys=self._selected_keys,
            skip_redundant=self._skip_redundant,
        )
        max_angular = max(
            _dispatch.max(density.keys.column("o3_lambda")),
            max(
                [
                    int(_dispatch.max(mdata[2].column("o3_lambda")))
                    for mdata in key_metadata
                ]
            ),
        )
        if self._max_angular < max_angular:
            raise ValueError(
                f"The density you provide requires max_angular={max_angular} "
                f"but on initialization max_angular={self._max_angular} was given"
            )

        # Perform iterative CG tensor products
        density_correlations: List[TensorMap] = []
        if compute_metadata:
            cg_backend = "metadata"
        else:
            cg_backend = self._cg_backend

        for iteration in range(n_iterations):
            # Define the correlation order of the current iteration
            correlation_order_it = iteration + 2

            # Combine block pairs
            blocks_out: List[TensorBlock] = []
            key_metadata_i = key_metadata[iteration]
            for j in range(len(key_metadata_i[0])):
                key_1: LabelsEntry = key_metadata_i[0][j]
                key_2: LabelsEntry = key_metadata_i[1][j]
                lambda_out: int = int(key_metadata_i[2].column("o3_lambda")[j])
                block_out = _clebsch_gordan._combine_blocks_same_samples(
                    density_correlation.block(key_1),
                    density.block(key_2),
                    lambda_out,
                    self._cg_coefficients,
                    cg_backend,
                )
                blocks_out.append(block_out)
            keys_out = key_metadata[iteration][2]
            density_correlation = TensorMap(keys=keys_out, blocks=blocks_out)

            # If this tensor is to be included in the output, move the [l1, l2, ...]
            # keys to properties and store
            if self._output_selection[iteration]:
                density_correlations.append(
                    density_correlation.keys_to_properties(
                        [f"l_{i}" for i in range(1, correlation_order_it + 1)]
                        + [f"k_{i}" for i in range(2, correlation_order_it)]
                    )
                )

        # Drop redundant key names. TODO: these should be part of the global
        # metadata associated with the TensorMap. Awaiting this functionality in
        # metatensor.
        for i, tensor in enumerate(density_correlations):
            keys = tensor.keys
            if len(_dispatch.unique(tensor.keys.column("order_nu"))) == 1:
                keys = keys.remove(name="order_nu")
            density_correlations[i] = TensorMap(
                keys=keys, blocks=[b.copy() for b in tensor.blocks()]
            )

        # Return a single TensorMap in the simple case
        if len(density_correlations) == 1:
            return density_correlations[0]

        # Otherwise return a list of TensorMaps
        return density_correlations
