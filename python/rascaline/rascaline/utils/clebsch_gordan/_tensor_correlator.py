"""
Module for computing the Clebsch-Gordan tensor product between two arbitrary
:py:class:`TensorMap` objects.
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


class TensorCorrelator(TorchModule):
    """
    A general class for computing the Clebsch-Gordan tensor product between two
    arbitrary :py:class:`TensorMap`.

    The constructor computes and stores the CG coefficients. The :py:meth:`compute`
    method computes the CG tensor product between two tensors.

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
        max_angular: int,
        arrays_backend: Optional[str] = None,
        cg_backend: Optional[str] = None,
    ) -> None:

        super().__init__()

        # Assign the arrays backend
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

        # Assign the CG backend
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

        # Compute the CG coefficients
        self._cg_coefficients = _coefficients.calculate_cg_coefficients(
            lambda_max=self._max_angular,
            cg_backend=self._cg_backend,
            use_torch=(arrays_backend == "torch"),
        )

    def forward(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_name: str,
        o3_lambda_2_name: str,
        selected_keys: Optional[Labels] = None,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = False,
    ) -> TensorMap:
        """
        Calls the :py:meth:`TensorCorrelator.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            tensor_1,
            tensor_2,
            o3_lambda_1_name=o3_lambda_1_name,
            o3_lambda_2_name=o3_lambda_2_name,
            selected_keys=selected_keys,
            angular_cutoff=angular_cutoff,
            skip_redundant=skip_redundant,
        )

    @torch_jit_export
    def compute_metadata(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_name: str,
        o3_lambda_2_name: str,
        selected_keys: Optional[Labels] = None,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = False,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._cg_tensor_product(
            tensor_1,
            tensor_2,
            o3_lambda_1_name=o3_lambda_1_name,
            o3_lambda_2_name=o3_lambda_2_name,
            selected_keys=selected_keys,
            skip_redundant=skip_redundant,
            angular_cutoff=angular_cutoff,
            compute_metadata=True,
        )

    def compute(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_name: str,
        o3_lambda_2_name: str,
        selected_keys: Optional[Labels] = None,
        angular_cutoff: Optional[int] = None,
        skip_redundant: bool = False,
    ) -> TensorMap:
        """
        Computes the correlation between ``tensor_1`` and ``tensor_2`` by taking the
        Clebsch-Gordan (CG) tensor product. Assumes the metadata of ``tensor_1`` and
        ``tensor_2`` has been modified to be compatible for the CG tensor product,
        according to the following rules:

        Both ``tensor_1`` and ``tensor_2`` must have named key dimensions "o3_lambda"
        and "o3_sigma" as these are used to determine the symmetry of output blocks upon
        combination.

        ``o3_lambda_1_name`` and ``o3_lambda_2_name`` define the output key dimension
        names that store the "o3_lambda" values of the blocks combined from ``tensor_1``
        and ``tensor_2`` respectively.

        Any other key dimensions that have equivalent names in both ``tensor_1`` and
        ``tensor_2`` will be matched, such that only blocks that have equal values in
        these dimensions will be combined.

        Any other named key dimensions that are present in ``tensor_1`` but not in
        ``tensor_2``, and vice versa, will have the full product computed.

        ``tensor_1`` and ``tensor_2`` must have a single component axis with a single
        key dimension named "o3_mu". ``tensor_1`` and ``tensor_2`` may have different
        samples, but all samples in the tensor with fewer samples dimensions must be
        present in the samples of the other tensor.

        A :py:class:`Labels` object can be passed in ``selected_keys`` to selected fro
        specific combinations of "o3_lambda" and "o3_sigma" that are computed.

        Note: ``skip_redundant`` must only be set to true when computing an
        auto-correlation of a density tensor, i.e. where ``tensor_1`` is an arbitrary
        body-order tensor formed from iterative combination of a density (body-order 2)
        tensor, and ``tensor_2`` is the same body-order 2 tensor.

        ``angular_cutoff`` can be used to limit the maximum angular momentum of the
        output blocks. Combinations of blocks that produce an angular momentum greater
        than this value will be skipped.

        :param tensor_1: A tensor of arbitrary body order, in :py:class:`TensorMap`
            format that will be correlated with ``tensor_2``.
        :param tensor_2: A tensor of arbitrary body order, in :py:class:`TensorMap`
            format that will be correlated with ``tensor_1``.
        :param o3_lambda_1_name: :py:class:`str`, the name of the output key dimension
            that stores the "o3_lambda" values of the blocks combined from ``tensor_1``.
        :param o3_lambda_2_name: :py:class:`str`, the name of the output key dimension
            that stores the "o3_lambda" values of the blocks combined from ``tensor_2``.
        :param selected_keys: A list of keys to select from the output tensor. If
            ``None``, all keys will be computed. This parameter must not be passed if
            passing ``angular_cutoff``.
        :param angular_cutoff: An optional integer that limits the maximum angular
            momentum of the output blocks. Combinations of blocks that produce an
            angular momentum greater than this value will be skipped. This parameter
            must not be passed if passing ``selected_keys``.
        :param skip_redundant: If ``True``, will skip the computation of redundant keys
            in the output tensor. This is only valid when computing an auto-correlation
            of a density tensor.

        :return: A :py:class:`TensorMap` object containing the Clebsch-Gordan tensor
            product of ``tensor_1`` and ``tensor_2``.
        """

        return self._cg_tensor_product(
            tensor_1,
            tensor_2,
            o3_lambda_1_name=o3_lambda_1_name,
            o3_lambda_2_name=o3_lambda_2_name,
            selected_keys=selected_keys,
            angular_cutoff=angular_cutoff,
            skip_redundant=skip_redundant,
            compute_metadata=False,
        )

    def _cg_tensor_product(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_name: str,
        o3_lambda_2_name: str,
        selected_keys: Labels,
        angular_cutoff: Optional[int],
        skip_redundant: bool,
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan tensor product between ``tensor_1`` and
        ``tensor_2``.

        Executes the following steps:

            1. Checks the metadata of ``tensor_1`` and ``tensor_2`` to ensure that they
               are compatible for the Clebsch-Gordan tensor product. Ensures neither has
               gradients (not currently supported).

            2. Computes the full product of the two keys, accounting for CG combination
               rules, and matching key dimensions that have the same name.

            3. Applies the key selection to remove keys that do not need computing. In
               the special case of a density correlation, where ``skip_redudant=True``,
               an extra filtering of keys is applied to remove redundant keys.

            4. For each key pair to be combined (and corresponding output angular
               channel), computes the CG tensor product on the block values.

            5. Returns the resulting tensor in :py:class:`TensorMap` format.
        """
        # 1. Check the inputs
        self._check_inputs(tensor_1, tensor_2, selected_keys, angular_cutoff)

        # 2. Compute the full product of the two keys
        output_keys, combinations = _utils._compute_output_keys(
            tensor_1.keys, tensor_2.keys, o3_lambda_1_name, o3_lambda_2_name
        )

        # 3. Apply key selections
        if selected_keys is not None:
            output_keys, combinations = _utils._apply_key_selection(
                output_keys,
                combinations,
                selected_keys,
            )
        if angular_cutoff is not None:
            output_keys, combinations = _utils._apply_angular_cutoff_selection(
                output_keys, combinations, angular_cutoff
            )
        if skip_redundant:  # should only be used for density auto-correlations
            output_keys, combinations = _utils._remove_redundant_keys(
                output_keys, combinations
            )
        # Group keys such that all o3_lambda values produced by the same 2 blocks are
        # together. This allows the block combination and CG product to be separated and
        # thus not unnecessarily repeated.
        output_keys, combinations = _utils._group_combinations_of_same_blocks(
            output_keys, combinations
        )

        # 4. Do the CG tensor product for each block combination
        output_blocks: List[TensorBlock] = []
        for combination in combinations:
            output_blocks.extend(
                _utils.cg_tensor_product_blocks(
                    tensor_1.block(combination.first),
                    tensor_2.block(combination.second),
                    o3_lambdas=combination.o3_lambdas,
                    cg_coefficients=self._cg_coefficients,
                    cg_backend="metadata" if compute_metadata else self._cg_backend,
                )
            )

        # 5. Build and return the resulting TensorMap
        return TensorMap(output_keys, output_blocks)

    def _check_inputs(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        selected_keys: Optional[Labels] = None,
        angular_cutoff: Optional[int] = None,
    ) -> None:
        """
        Checks the metadata of ``tensor_1`` and ``tensor_2`` to ensure that they are
        compatible for the Clebsch-Gordan tensor product.

        This checks that:

            - Both tensors have "o3_lambda" and "o3_sigma" key dimensions.
            - Both tensors have a single component axis with a single dimension named
              "o3_mu".
            - The sum of maximum "o3_lambda" values in each tensor is not greater than
              the maximum angular momentum ``self._max_angular`` used to calculate the
              CG coefficients in the constructor.
            - There is no intersection of property names between the two tensors.
        """
        # Check symmetry dimensions
        for tensor in [tensor_1, tensor_2]:
            if "o3_lambda" not in tensor.keys.names:
                raise ValueError(
                    "input tensors must have a named key dimension `'o3_lambda`"
                )
            if "o3_sigma" not in tensor.keys.names:
                raise ValueError(
                    "input tensors must have a named key dimension `'o3_sigma`"
                )
            if not tensor.component_names == ["o3_mu"]:
                raise ValueError(
                    "input tensors must have a single component"
                    " axis with a single named dimensions `'o3_mu'`"
                )

        if selected_keys is not None and angular_cutoff is not None:
            raise ValueError(
                "only one of `selected_keys` and `angular_cutoff` should be passed"
            )

        # Check maximum angular momentum
        max_angular = _dispatch.max(tensor_1.keys.column("o3_lambda")) + _dispatch.max(
            tensor_2.keys.column("o3_lambda")
        )
        if selected_keys is not None:
            max_angular = min(max_angular, _dispatch.max(selected_keys.column("o3_lambda")))
        if angular_cutoff is not None:
            max_angular = min(max_angular, angular_cutoff)
        if max_angular > self._max_angular:
            raise ValueError(
                f"the maximum angular momentum `{max_angular}` required by the"
                " input tensors and key selection / angular cutoff exceeds the"
                f" `max_angular={self._max_angular}` set in the constructor used"
                " to calculate the CG coefficients"
            )

        # Check property names
        for property_name in tensor_1.property_names:
            if property_name in tensor_2.property_names:
                raise ValueError(
                    f"property name `{property_name}` present in both input tensors."
                    " As all property dimensions are combined, they must have"
                    " different names in the two tensors. Use"
                    " `metatensor.rename_dimension` and try again."
                )

        # Check no gradients as not currently supported.
        contains_gradients_1 = all(
            [len(list(block.gradients())) > 0 for _, block in tensor_1.items()]
        )
        contains_gradients_2 = all(
            [len(list(block.gradients())) > 0 for _, block in tensor_2.items()]
        )
        if contains_gradients_1 or contains_gradients_2:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented."
                " Use `metatensor.remove_gradients` to remove gradients from the input"
                " tensors."
            )
