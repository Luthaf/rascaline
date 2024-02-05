"""
Module containing dispatch functions for numpy/torch CG combination operations.
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


def where(array):
    """Return the indices where `array` is True.

    This function has the same behavior as ``np.where(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.where(array)
    elif isinstance(array, np.ndarray):
        return np.where(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def abs(array):
    """
    Returns the absolute value of the elements in the array.

    It is equivalent of np.abs(array) and torch.abs(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.abs(array)
    elif isinstance(array, np.ndarray):
        return np.abs(array).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def argsort(array):
    """
    Returns the sorted arguments of the elements in the array.

    It is equivalent of np.argsort(array) and torch.argsort(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.argsort(array)
    elif isinstance(array, np.ndarray):
        return np.argsort(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def unique(array, axis: Optional[int] = None):
    """Find the unique elements of an array."""
    if isinstance(array, TorchTensor):
        return torch.unique(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, axis=axis)


def arange_like(array, start: int, end: int):
    """
    Create an array from start to end with ascending integers.

    It is equivalent of np.arange(start, end) and torch.arange(start, end) for the given
    array dtype and device.
    """
    if isinstance(array, TorchTensor):
        return torch.arange(
            start,
            end,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        )
    elif isinstance(array, np.ndarray):
        return np.arange(start, end)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def int_range_like(min_val, max_val, like):
    """Returns an array of integers from min to max, non-inclusive, based on the
    type of `like`"""
    if isinstance(like, TorchTensor):
        return torch.arange(min_val, max_val, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.arange(min_val, max_val).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def int_array_like(int_list: List[int], like):
    """
    Converts the input list of int to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        return torch.tensor(int_list, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def concatenate(arrays, axis: Optional[int] = 0):
    """Concatenate arrays along an axis."""
    if isinstance(arrays[0], TorchTensor):
        return torch.cat(arrays, dim=axis)
    elif isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def all(array, axis: Optional[int] = None):
    """Test whether all array elements along a given axis evaluate to True.

    This function has the same behavior as
    ``np.all(array,axis=axis)``.
    """
    if isinstance(array, bool):
        array = np.array(array)
    if isinstance(array, list):
        array = np.array(array)

    if isinstance(array, TorchTensor):
        # torch.all has two implementation, and picks one depending if more than one
        # parameter is given. The second one does not supports setting dim to `None`
        if axis is None:
            return torch.all(input=array)
        else:
            return torch.all(input=array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.all(a=array, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def any(array):
    """Test whether any array elements along a given axis evaluate to True.

    This function has the same behavior as
    ``np.any(array)``.
    """
    if isinstance(array, bool):
        array = np.array(array)
    if isinstance(array, list):
        array = np.array(array)
    if isinstance(array, TorchTensor):
        return torch.any(array)
    elif isinstance(array, np.ndarray):
        return np.any(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def zeros_like(shape, like):
    """Return an array of zeros with the same shape and type as a given array.

    This function has the same behavior as
    ``np.zeros_like(array)``.
    """
    if isinstance(like, TorchTensor):
        return torch.zeros(
            shape,
            requires_grad=like.requires_grad,
            dtype=like.dtype,
            device=like.device,
        )
    elif isinstance(like, np.ndarray):
        return np.zeros(shape, dtype=like.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def allclose(
    a: TorchTensor,
    b: TorchTensor,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
):
    """Compare two arrays using ``allclose``

    This function has the same behavior as
    ``np.allclose(array1, array2, rtol, atol, equal_nan)``.
    """
    if isinstance(a, TorchTensor):
        _check_all_torch_tensor([b])
        return torch.allclose(
            input=a, other=b, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    elif isinstance(a, np.ndarray):
        _check_all_np_ndarray([b])
        return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def swapaxes(array, axis0: int, axis1: int):
    """Swaps axes of an array."""
    if isinstance(array, TorchTensor):
        return torch.swapaxes(array, axis0, axis1)
    elif isinstance(array, np.ndarray):
        return np.swapaxes(array, axis0, axis1)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
