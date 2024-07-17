"""
This module contains utilities to convert cartesian TensorMap to spherical and
respectively.
"""

from typing import List, Optional

import numpy as np

from .. import _dispatch
from .._backend import (
    Array,
    Labels,
    TensorBlock,
    TensorMap,
    TorchTensor,
    torch_jit_is_scripting,
)
from . import _coefficients


def cartesian_to_spherical(
    tensor: TensorMap,
    components: List[str],
    keep_l_in_keys: Optional[bool] = None,
    remove_blocks_threshold: Optional[float] = 1e-9,
    cg_backend: Optional[str] = None,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` of arbitrary rank from cartesian form to a spherical form.

    Starting from a tensor on a basis of product of cartesian coordinates, this function
    computes the same tensor using a basis of spherical harmonics ``Y^M_L``. For
    example, a rank 1 tensor with a single "xyz" component would be represented as a
    single L=1 spherical harmonic; while a rank 5 tensor using a product basis ``ℝ^3 ⊗
    ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3`` would require multiple blocks up to L=5 spherical harmonics.

    A single :py:class:`TensorBlock` in the input might correspond to multiple
    :py:class:`TensorBlock` in the output. The output keys will contain all the
    dimensions of the input keys, plus ``o3_lambda`` (indicating the spherical harmonics
    degree) and ``o3_sigma`` (indicating that this block is a proper- or improper tensor
    with ``+1`` and ``-1`` respectively). If ``keep_l_in_keys`` is ``True`` or if the
    input tensor is a tensor of rank 3 or more, the keys will also contain multiple
    ``l_{i}`` and  ``k_{i}`` dimensions, which indicate which angular momenta have been
    coupled together in which order to get this block.

    ``components`` specifies which ones of the components of the input
    :py:class:`TensorMap` should be transformed from cartesian to spherical. All these
    components will be replaced in the output by a single ``o3_mu`` component,
    corresponding to the spherical harmonics ``M``.

    By default, symmetric tensors will only contain blocks corresponding to
    ``o3_sigma=1``. This is achieved by checking the norm of the blocks after the full
    calculation; and dropping any block with a norm below ``remove_blocks_epsilon``. To
    keep all blocks regardless of their norm, you can set
    ``remove_blocks_epsilon=None``.

    :param tensor: input tensor, using a cartesian product basis
    :param components: components of the input tensor to transform into spherical
        components
    :param keep_l_in_keys: should the output contains the values of angular momenta that
        were combined together? This defaults to ``False`` for rank 1 and 2 tensors,
        and ``True`` for all other tensors.

        Keys named ``l_{i}`` correspond to the input ``components``, with ``l_1`` being
        the last entry in ``components`` and ``l_N`` the first one. Keys named ``k_{i}``
        correspond to intermediary spherical components created during the calculation,
        i.e. a ``k_{i}`` used to be ``o3_lambda``.
    :param remove_blocks_threshold: Numerical tolerance to use when determining if a
        block's norm is zero or not. Blocks with zero norm will be excluded from the
        output. Set this to ``None`` to keep all blocks in the output.
    :param cg_backend: Backend to use for Clebsch-Gordan calculations. This can be
        ``"python-dense"`` or ``"python-sparse"`` for dense or sparse operations
        respectively. If ``None``, this is automatically determined.
    :param cg_coefficients: Cache containing Clebsch-Gordan coefficients. This is
        optional except when using this function from TorchScript. The coefficients
        should be computed with :py:func:`calculate_cg_coefficients`, using the same
        ``cg_backend`` as this function.

    :return: :py:class:`TensorMap` containing spherical components instead of cartesian
        components.
    """
    if len(tensor) == 0 or len(components) == 0:
        # nothing to do
        return tensor

    if not isinstance(components, list):
        raise TypeError(f"`components` should be a list, got {type(components)}")

    if keep_l_in_keys is None:
        if len(components) < 3:
            keep_l_in_keys = False
        else:
            keep_l_in_keys = True

    axes_to_convert: List[int] = []
    all_component_names = tensor.component_names
    for component in components:
        if component in all_component_names:
            idx = all_component_names.index(component)
            axes_to_convert.append(idx + 1)
        else:
            raise ValueError(f"'{component}' is not part of this tensor components")

    for key, block in tensor.items():
        for idx in axes_to_convert:
            values_list = _dispatch.to_int_list(block.components[idx - 1].values[:, 0])
            if values_list != [0, 1, 2]:
                name = block.components[idx - 1].names[0]
                raise ValueError(
                    f"component '{name}' in block for {key.print()} should have "
                    f"[0, 1, 2] as values, got {values_list} instead"
                )

    # we need components to be consecutive
    if list(range(axes_to_convert[0], axes_to_convert[-1] + 1)) != axes_to_convert:
        raise ValueError(
            f"this function only supports consecutive components, {components} are not"
        )

    key_names = tensor.keys.names
    if "o3_lambda" in key_names:
        raise ValueError(
            "this tensor already has an `o3_lambda` key, "
            "is it already in spherical form?"
        )

    for i in range(len(components)):
        if f"l_{i}" in key_names:
            raise ValueError(
                f"this tensor already has an `l_{i}` key, "
                "is it already in spherical form?"
            )

    tensor_rank = len(components)
    if tensor_rank > 2 and not keep_l_in_keys:
        raise ValueError(
            "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
        )

    if isinstance(tensor.block(0).values, TorchTensor):
        arrays_backend = "torch"
        values = tensor.block(0).values
        dtype = values.dtype
        device = values.device
    elif isinstance(tensor.block(0).values, np.ndarray):
        arrays_backend = "numpy"
        values = tensor.block(0).values
        dtype = values.dtype
        device = "cpu"
    else:
        raise TypeError(
            f"unknown array type in tensor ({type(tensor.block(0).values)}), "
            "only numpy and torch are supported"
        )

    if cg_backend is None:
        # TODO: benchmark & change the default?
        if arrays_backend == "torch":
            cg_backend = "python-dense"
        else:
            cg_backend = "python-sparse"

    new_component_names: List[str] = []
    for idx in range(len(tensor.component_names)):
        if idx + 1 in axes_to_convert:
            if len(axes_to_convert) == 1:
                new_component_names.append("o3_mu")
            else:
                new_component_names.append(f"__internal_to_convert_{idx}")
        else:
            new_component_names.append("")

    # Step 1: transform xyz dimensions to o3_lambda=1 dimensions
    # This is done with `roll`, since (y, z, x) is the same as m = (-1, 0, 1)
    new_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        values = _dispatch.roll(
            block.values,
            shifts=[-1] * len(axes_to_convert),
            axis=axes_to_convert,
        )

        new_components: List[Labels] = []
        for idx, component in enumerate(block.components):
            if idx + 1 in axes_to_convert:
                new_components.append(
                    Labels(
                        names=[new_component_names[idx]],
                        values=_dispatch.int_array_like(
                            [[-1], [0], [1]], component.values
                        ),
                    )
                )
            else:
                new_components.append(component)

        new_blocks.append(
            TensorBlock(values, block.samples, new_components, block.properties)
        )

    if tensor_rank == 1:
        new_keys = tensor.keys
        # we are done, add o3_lambda/o3_sigma/l_1 to the keys & return
        if keep_l_in_keys:
            new_keys = new_keys.insert(
                0,
                "l_1",
                _dispatch.int_array_like([1] * len(new_keys), new_keys.values),
            )

        new_keys = new_keys.insert(
            0,
            "o3_sigma",
            _dispatch.int_array_like([1] * len(tensor.keys), tensor.keys.values),
        )

        new_keys = new_keys.insert(
            0,
            "o3_lambda",
            _dispatch.int_array_like([1] * len(tensor.keys), tensor.keys.values),
        )

        return TensorMap(new_keys, new_blocks)

    # Step 2: if there is more than one dimension, couple them with CG coefficients
    #
    # We start from an array of shape [..., 3, 3, 3, 3, 3, ...] with as many 3 as
    # `len(components)`. Then we iteratively combine the two rightmost components into
    # as many new lambda/mu entries as required, until there is only one component left.
    # Each step will create multiple blocks (corresponding to the different o3_lambda
    # created by combining two o3_lambda=1 terms), that might on their turn create more
    # blocks if more combinations are required.
    #
    # For example, with a rank 3 tensor we go through the following:
    #
    # - Step 1: [..., 3, 3, 3, ...] => [..., 3, 1, ...] (o3_lambda=0, o3_sigma=+1)
    #                               => [..., 3, 3, ...] (o3_lambda=1, o3_sigma=-1)
    #                               => [..., 3, 5, ...] (o3_lambda=2, o3_sigma=+1)
    #
    # - Step 2: [..., 3, 1, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #           [..., 3, 3, ...] => [..., 1, ...] (o3_lambda=0, o3_sigma=-1)
    #                            => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #           [..., 3, 5, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #                            => [..., 7, ...] (o3_lambda=3, o3_sigma=+1)
    if cg_coefficients is None:
        if torch_jit_is_scripting():
            raise ValueError(
                "in TorchScript mode, `cg_coefficients` must be pre-computed "
                "and given to this function explicitly"
            )
        else:
            cg_coefficients = _coefficients.calculate_cg_coefficients(
                lambda_max=len(axes_to_convert),
                cg_backend=cg_backend,
                arrays_backend=arrays_backend,
                dtype=dtype,
                device=device,
            )

    iteration_index = 0
    while len(axes_to_convert) > 1:
        tensor = _do_coupling(
            tensor=tensor,
            component_1=axes_to_convert[-2] - 1,
            component_2=axes_to_convert[-1] - 1,
            cg_coefficients=cg_coefficients,
            cg_backend=cg_backend,
            keep_l_in_keys=keep_l_in_keys,
            iteration_index=iteration_index,
        )

        axes_to_convert.pop()
        iteration_index += 1

    if remove_blocks_threshold is None:
        return tensor

    # Step 3: for symmetry reasons, some of the blocks will be zero everywhere (for
    # example o3_sigma=-1 blocks if the input tensor is fully symmetric). If the user
    # gave us a threshold, we remove all blocks with a norm below this threshold.
    new_keys_values: List[Array] = []
    new_blocks: List[TensorBlock] = []
    for key_idx, block in enumerate(tensor.blocks()):
        key = tensor.keys.entry(key_idx)
        values = block.values.reshape(-1, 1)
        norm = values.T @ values
        if norm > remove_blocks_threshold:
            new_keys_values.append(key.values.reshape(1, -1))
            new_blocks.append(
                TensorBlock(
                    values=block.values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

    return TensorMap(
        Labels(tensor.keys.names, _dispatch.concatenate(new_keys_values, axis=0)),
        new_blocks,
    )


def _do_coupling(
    tensor: TensorMap,
    component_1: int,
    component_2: int,
    keep_l_in_keys: bool,
    iteration_index: int,
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> TensorMap:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    This function takes in a :py:class:`TensorMap` where two of the components
    (indicated by ``component_1`` and ``component_2``) behave like spherical harmonics
    ``Y^m1_l1`` and ``Y^m2_l2``, and project it onto a single spherical harmonic
    ``Y^M_L``. This transformation uses the following relation:

    ``|L M> = |l1 l2 L M> = \\sum_{m1 m2} <l1 m1 l2 m2|L M> |l1 m1> |l2 m2>``

    where ``<l1 m1 l2 m2|L M>`` are Clebsch-Gordan coefficients.

    The output will contain many blocks for each block in the input, matching all the
    different ``L`` (called ``o3_lambda`` in the code) required to do a full projection.

    This process can be iterated: a multi-dimensional array that is the product of many
    ``Y^m_l`` can be turned into a set of multiple terms transforming as a single
    ``Y^M_L``.

    :param tensor: input :py:class:`TensorMap`
    :param components_1: first component of the ``tensor`` behaving like spherical
        harmonics
    :param components_2: second component of the ``tensor`` behaving like spherical
        harmonics
    :param keep_l_in_keys: whether ``l1`` and ``l2`` (the original spherical harmonic
        degrees) should be kept in the keys. This can be useful to undo this
        transformation (or even required if there is more than one path to get to a
        given value for ``o3_lambda``)
    :param iteration_index: when iterating the coupling, this should be the number of
        iterations already done (i.e. the number of time this function has been called)
    :param cg_coefficients: pre-computed set of Clebsch-Gordan coefficients
    :param cg_backend: which backend to use for the calculations

    :return: :py:class:`TensorMap` using the coupled basis. This will contain the same
        keys as the input ``tensor``, plus ``o3_lambda``. The components in positions
        ``components_1`` and ``components_2`` will be replaced by a single ``o3_mu``
        component.
    """
    assert component_2 == component_1 + 1

    new_keys = tensor.keys

    if "o3_lambda" in tensor.keys.names:
        old_sigmas = new_keys.column("o3_sigma")
        new_keys = new_keys.remove("o3_sigma")
    else:
        old_sigmas = _dispatch.int_array_like([1] * len(new_keys), new_keys.values)

    if keep_l_in_keys:
        array_of_ones = _dispatch.int_array_like([1] * len(new_keys), new_keys.values)
        if "o3_lambda" in tensor.keys.names:
            assert iteration_index > 0
            new_keys = new_keys.rename("o3_lambda", f"k_{iteration_index}")
            new_keys = new_keys.insert(0, f"l_{iteration_index + 2}", array_of_ones)
        else:
            assert iteration_index == 0
            new_keys = new_keys.insert(0, "l_1", array_of_ones)
            new_keys = new_keys.insert(0, "l_2", array_of_ones)

    new_keys_values: List[List[int]] = []
    new_blocks: List[TensorBlock] = []
    for key_idx, block in enumerate(tensor.blocks()):
        key = new_keys.entry(key_idx)
        old_sigma = int(old_sigmas[key_idx])

        # get l1, l2 from the block's shape
        block_shape = block.values.shape
        l1 = (block_shape[component_1 + 1] - 1) // 2
        l2 = (block_shape[component_2 + 1] - 1) // 2

        # reshape the values to look like (n_s, 2*l1 + 1, 2*l2 + 1, n_p)
        shape_before = 1
        for axis in range(component_1 + 1):
            shape_before *= block_shape[axis]

        shape_after = 1
        for axis in range(component_2 + 2, len(block_shape)):
            shape_after *= block_shape[axis]

        array = block.values.reshape(
            shape_before,
            block.values.shape[component_1 + 1],
            block.values.shape[component_2 + 1],
            shape_after,
        )

        # generate the set of o3_lambda to compute
        o3_lambdas = list(range(max(l1, l2) - min(l1, l2), (l1 + l2) + 1))

        # actual calculation
        outputs = _coefficients.cg_couple(
            array, o3_lambdas, cg_coefficients, cg_backend
        )

        # create one block for each output of `cg_couple`
        for o3_lambda, values in zip(o3_lambdas, outputs):
            o3_sigma = int(old_sigma * (-1) ** (l1 + l2 + o3_lambda))
            new_keys_values.append(
                [o3_lambda, o3_sigma] + _dispatch.to_int_list(key.values)
            )

            new_shape = list(block.values.shape)
            new_shape.pop(component_2 + 1)
            new_shape[component_1 + 1] = 2 * o3_lambda + 1

            new_components = block.components
            new_components.pop(component_2)
            new_components[component_1] = Labels(
                "o3_mu",
                _dispatch.int_array_like(
                    [[mu] for mu in range(-o3_lambda, o3_lambda + 1)], new_keys.values
                ),
            )

            new_blocks.append(
                TensorBlock(
                    values.reshape(new_shape),
                    samples=block.samples,
                    components=new_components,
                    properties=block.properties,
                )
            )

    new_keys = Labels(
        ["o3_lambda", "o3_sigma"] + new_keys.names,
        _dispatch.int_array_like(new_keys_values, new_keys.values),
    )
    return TensorMap(new_keys, new_blocks)
