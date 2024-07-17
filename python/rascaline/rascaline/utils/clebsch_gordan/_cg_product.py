"""
Module for computing the Clebsch-Gordan tensor product between two arbitrary
:py:class:`TensorMap` objects.
"""

from typing import Callable, List, Optional

import numpy as np

from .. import _dispatch
from .._backend import (
    Device,
    DType,
    Labels,
    TensorBlock,
    TensorMap,
    TorchModule,
    TorchScriptClass,
    torch_jit_is_scripting,
)
from . import _coefficients, _utils


try:
    import torch
except ImportError:
    pass


class ClebschGordanProduct(TorchModule):
    """
    A general class for computing the Clebsch-Gordan (CG) tensor product between two
    arbitrary :py:class:`TensorMap`.

    The constructor computes and stores the CG coefficients. The :py:meth:`compute`
    method computes the CG tensor product between two tensors.
    """

    def __init__(
        self,
        *,
        max_angular: int,
        cg_backend: Optional[str] = None,
        keys_filter: Optional[Callable[[Labels], List[int]]] = None,
        arrays_backend: Optional[str] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """
        :param max_angular: :py:class:`int`, the maximum angular momentum to compute CG
            coefficients for.
        :param cg_backend: :py:class:`str`, the backend to use for the CG tensor
            product. If ``None``, the backend is automatically selected based on the
            arrays backend ("numpy" when importing this class from ``rascaline.utils``,
            and "torch" when importing from ``rascaline.torch.utils``).
        :param keys_filter: A function to remove more keys from the output. This is
            applied after any user-provided ``key_selection`` in :py:meth:`compute`.
            This function should take one argument ``keys: Labels``, and return the
            indices of keys to keep.
        :param arrays_backend: :py:class:`str`, the backend to use for array operations.
            If ``None``, the backend is automatically selected based on the environment.
            Possible values are "numpy" and "torch".
        :param dtype: the scalar type to use to store coefficients
        :param device: the computational device to use for calculations. This must be
            ``"cpu"`` if ``array_backend="numpy"``.
        """

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
            # all good
            pass
        else:
            raise ValueError(
                f"Unknown `arrays_backend` {arrays_backend}."
                "Only 'numpy' and 'torch' are supported."
            )

        if dtype is None:
            if arrays_backend == "torch":
                dtype = torch.get_default_dtype()
            elif arrays_backend == "numpy":
                dtype = np.float64

        if device is None:
            if arrays_backend == "torch":
                device = torch.get_default_device()
            elif arrays_backend == "numpy":
                device = "cpu"

        if arrays_backend == "numpy":
            if device != "cpu":
                raise ValueError("`device` can only be 'cpu' for numpy arrays")

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
            arrays_backend=arrays_backend,
            dtype=dtype,
            device=device,
        )

        if keys_filter is None:
            self._keys_filter = _keys_filter_noop
        else:
            self._keys_filter = keys_filter

    def compute(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan (CG) tensor product between ``tensor_1`` and
        ``tensor_2``.

        This function assumes the metadata of ``tensor_1`` and ``tensor_2`` has been
        modified to be compatible for the CG tensor product, according to the following
        rules:

        - both ``tensor_1`` and ``tensor_2`` must have key dimensions ``"o3_lambda"``
          and ``"o3_sigma"`` as these are used to determine the symmetry of output
          blocks upon combination;
        - ``o3_lambda_1_new_name`` and ``o3_lambda_2_new_name`` define the output key
          dimension names that store the ``"o3_lambda"`` values of the blocks combined
          from ``tensor_1`` and ``tensor_2`` respectively;
        - any other key dimensions that have equivalent names in both ``tensor_1`` and
          ``tensor_2`` will be matched, such that only blocks that have equal values in
          these dimensions will be combined;
        - any other named key dimensions that are present in ``tensor_1`` but not in
            ``tensor_2``, and vice versa, will have the full product computed;
        - ``tensor_1`` and ``tensor_2`` must have a single component axis with a single
          key dimension named ``"o3_mu"``. ``tensor_1`` and ``tensor_2`` may have
          different samples, but all samples in the tensor with fewer samples dimensions
          must be present in the samples of the other tensor.

        A :py:class:`Labels` object can be passed in ``selected_keys`` to select
        specific keys to compute. The full set of output keys are computed, then are
        filtered to match the key dimensions passed in ``selected_keys``.

        This parameter can be used to match the output tensor to a given target basis
        set definition, and/or enhance performance by limiting the combinations
        computed.

        For instance, passing just the ``"o3_lambda"`` key dimension with a range of
        values ``0, ..., max_angular`` can be used to perform angular selection,
        limiting the maximum angular momentum of the output blocks.

        Note that using ``selected_keys`` to perform any kind of selection will reduce
        the information content of the output tensor. This may be important if the
        returned tensor is used in further CG tensor products.

        :param tensor_1: The first :py:class:`TensorMap`, containing data with SO(3)
            character.
        :param tensor_2: The first :py:class:`TensorMap`, containing data with SO(3)
            character.
        :param o3_lambda_1_new_name: :py:class:`str`, the name of the output key
            dimension that stores the ``"o3_lambda"`` values of the blocks combined from
            ``tensor_1``.
        :param o3_lambda_2_new_name: :py:class:`str`, the name of the output key
            dimension that stores the ``"o3_lambda"`` values of the blocks combined from
            ``tensor_2``.
        :param selected_keys: :py:class:`Labels`, the keys to compute on the final
            iteration. If ``None``, all keys are computed. Subsets of key dimensions can
            be passed to compute output blocks that match in these dimensions.

        :return: A :py:class:`TensorMap` containing the Clebsch-Gordan tensor product of
            ``tensor_1`` and ``tensor_2``.
        """

        return self._cg_tensor_product(
            tensor_1,
            tensor_2,
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
            compute_metadata=False,
        )

    def forward(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Calls the :py:meth:`ClebschGordanProduct.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            tensor_1,
            tensor_2,
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
        )

    def compute_metadata(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
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
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
            compute_metadata=True,
        )

    def _cg_tensor_product(
        self,
        tensor_1: TensorMap,
        tensor_2: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels],
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

            3. Applies the key selection to remove keys that do not need computing. This
               includes both ``selected_keys`` and any user-provided ``keys_filter``.

            4. For each key pair to be combined (and corresponding output angular
               channel), computes the CG tensor product on the block values.

            5. Returns the resulting tensor in :py:class:`TensorMap` format.
        """
        # 1. Check the inputs
        self._check_inputs(tensor_1, tensor_2, selected_keys)

        # 2. Compute the full product of the two keys
        output_keys, combinations = _utils._compute_output_keys(
            tensor_1.keys, tensor_2.keys, o3_lambda_1_new_name, o3_lambda_2_new_name
        )

        # 3. a) Apply key selections
        if selected_keys is not None:
            output_keys, combinations = _utils._apply_key_selection(
                output_keys,
                combinations,
                selected_keys,
            )

        # 3. b) Apply key filter
        keys_to_keep = self._keys_filter(output_keys)
        output_keys = Labels(output_keys.names, output_keys.values[keys_to_keep])
        combinations = [combinations[i] for i in keys_to_keep]

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

        # Check maximum angular momentum
        max_angular: int = int(
            _dispatch.max(tensor_1.keys.column("o3_lambda"))
            + _dispatch.max(tensor_2.keys.column("o3_lambda"))
        )
        if selected_keys is not None:
            # Update `max_angular` if angular selection is being performed via
            # `selected_keys`
            if "o3_lambda" in selected_keys.names:
                # Check max of 'o3_lambda' column is valid
                for lam in selected_keys.column("o3_lambda"):
                    if lam > self._max_angular:
                        raise ValueError(
                            "the maximum angular momentum value found in key dimension"
                            " `'o3_lambda'` in `selected_keys` exceeds"
                            f" `max_angular={self._max_angular}`"
                            " used to calculate the CG coefficients in the constructor."
                        )
                max_angular = min(
                    max_angular, int(_dispatch.max(selected_keys.column("o3_lambda")))
                )
            if "o3_sigma" in selected_keys.names:
                # Check 'o3_sigma' column is valid
                for o3_sigma in selected_keys.column("o3_sigma"):
                    if int(o3_sigma) not in [-1, 1]:
                        raise ValueError(
                            "key dimension `'o3_sigma'` in `selected_keys` must only"
                            " contain values of -1 or 1."
                        )
        if max_angular > int(self._max_angular):
            raise ValueError(
                f"the maximum angular momentum `{max_angular}` required by the"
                " input tensors (and perhaps lowered by angular selection via"
                f" `selected_keys`) exceeds the `max_angular={self._max_angular}`"
                " used to calculate the CG coefficients in the constructor."
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


def _keys_filter_noop(keys: Labels):
    """A ``key_filter`` for ``ClebschGordanProduct`` that does nothing"""
    return [i for i in range(len(keys))]
