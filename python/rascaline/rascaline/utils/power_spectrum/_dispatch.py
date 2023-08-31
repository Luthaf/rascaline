"""Helper functions to dispatch methods between numpy and torch.

The functions are similar to those in metatensor-operations. Missing functions may
already exist there. Functions are ordered alphabetically.
"""

from typing import List, Optional

import numpy as np


try:
    import torch
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def _check_all_torch_tensor(arrays: List[TorchTensor]):
    for array in arrays:
        if not isinstance(array, TorchTensor):
            raise TypeError(
                f"expected argument to be a torch.Tensor, but got {type(array)}"
            )


def _check_all_np_ndarray(arrays):
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"expected argument to be a np.ndarray, but got {type(array)}"
            )


def concatenate(arrays: List[TorchTensor], axis: int):
    """
    Concatenate a group of arrays along a given axis.

    This function has the same behavior as ``numpy.concatenate(arrays, axis)``
    and ``torch.concatenate(arrays, axis)``.

    Passing `axis` as ``0`` is equivalent to :py:func:`numpy.vstack`, ``1`` to
    :py:func:`numpy.hstack`, and ``2`` to :py:func:`numpy.dstack`, though any
    axis index > 0 is valid.
    """
    if isinstance(arrays[0], TorchTensor):
        _check_all_torch_tensor(arrays)
        return torch.concatenate(arrays, axis)
    elif isinstance(arrays[0], np.ndarray):
        _check_all_np_ndarray(arrays)
        return np.concatenate(arrays, axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def empty_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an uninitialized array, with the given ``shape``, and similar dtype,
    device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.

    This is the equivalent to ``np.empty_like(array, shape=shape)``.
    """
    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()
        return torch.empty(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        return np.empty_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def list_to_array(array, data: List[List[int]]):
    """Create an object from data with the same type as ``array``."""
    if isinstance(array, TorchTensor):
        return torch.tensor(data)
    elif isinstance(array, np.ndarray):
        return np.array(data)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def matmul(a, b):
    """Matrix product of two arrays."""
    if isinstance(a, TorchTensor):
        _check_all_torch_tensor([b])
        return torch.matmul(a, b)
    elif isinstance(a, np.ndarray):
        _check_all_np_ndarray([b])
        return np.matmul(a, b)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def to_index_array(array):
    """Returns an array that is suitable for indexing a dimension of
    a different array.
    After a few checks (int, 1D), this operation will convert the dtype to
    torch.long (which is, in some torch versions, the only acceptable type
    of index tensor). Numpy arrays are left unchanged.
    """
    if len(array.shape) != 1:
        raise ValueError("Index arrays must be 1D")

    if isinstance(array, TorchTensor):
        if torch.is_floating_point(array):
            raise ValueError("Index arrays must be integers")
        return array.to(torch.long)
    elif isinstance(array, np.ndarray):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Index arrays must be integers")
        return array
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def unique(array, axis: Optional[int] = None):
    """Find the unique elements of an array."""
    if isinstance(array, TorchTensor):
        return torch.unique(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def zeros_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an array filled with zeros, with the given ``shape``, and similar
    dtype, device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.

    This is the equivalent to ``np.zeros_like(array, shape=shape)``.
    """
    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()

        return torch.zeros(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        return np.zeros_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
