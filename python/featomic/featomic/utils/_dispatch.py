"""Helper functions to dispatch methods between numpy and torch.

The functions are similar to those in metatensor-operations. Missing functions may
already exist there. Functions are ordered alphabetically.
"""

import itertools
from typing import List, Optional, Union

import numpy as np

from ._backend import Device, DType, torch_jit_is_scripting


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
    Create an uninitialized array, with the given ``shape``, and similar dtype, device
    and other options as ``array``.

    If ``shape`` is ``None``, the array shape is used instead. ``requires_grad`` is only
    used for torch tensors, and set the corresponding value on the returned array.

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
    Create an array filled with zeros, with the given ``shape``, and similar dtype,
    device and other options as ``array``.

    If ``shape`` is ``None``, the array shape is used instead. ``requires_grad`` is only
    used for torch tensors, and set the corresponding value on the returned array.

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
        if shape is None:
            shape = array.shape
        return np.zeros_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


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


def contiguous(array):
    """
    Returns a contiguous array.

    It is equivalent of np.ascontiguousarray(array) and tensor.contiguous(). In
    the case of numpy, C order is used for consistency with torch. As such, only
    C-contiguity is checked.
    """
    if isinstance(array, TorchTensor):
        if array.is_contiguous():
            return array
        return array.contiguous()
    elif isinstance(array, np.ndarray):
        if array.flags["C_CONTIGUOUS"]:
            return array
        return np.ascontiguousarray(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def to_int_list(array) -> List[int]:
    if isinstance(array, TorchTensor):
        # we need to do it this way because of
        # https://github.com/pytorch/pytorch/issues/76295
        return array.to(dtype=torch.int64).tolist()
    elif isinstance(array, np.ndarray):
        return array.tolist()
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def int_range_like(min_val: int, max_val: int, like):
    """
    Returns an array of integers from min to max, non-inclusive, based on the type of
    `like`

    It is equivalent of np.arange(start, end) and torch.arange(start, end) for the given
    array dtype and device.
    """
    if isinstance(like, TorchTensor):
        return torch.arange(min_val, max_val, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.arange(min_val, max_val).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def int_array_like(int_list: Union[List[int], List[List[int]]], like):
    """
    Converts the input list of int to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        if torch.jit.isinstance(int_list, List[int]):
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
        else:
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def real_array_like(float_list: List[float], like):
    """
    Converts the input list of float to a numpy array or torch tensor
    based on the array type of `like`.
    """
    if isinstance(like, TorchTensor):
        return torch.tensor(float_list, dtype=torch.float64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(float_list).astype(np.float64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def bool_array_like(bool_list: List[bool], like):
    """
    Converts the input list of bool to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        return torch.tensor(bool_list, dtype=torch.bool, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(bool_list).astype(bool)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def cartesian_prod(array1, array2):
    """
    Imitates like itertools.product(array1, array2)
    """
    if isinstance(array1, TorchTensor) and isinstance(array2, TorchTensor):
        return torch.cartesian_prod(array1, array2)
    elif isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        # using itertools should be fastest way according to
        # https://stackoverflow.com/a/28684982
        return np.array(list(itertools.product(array1, array2)))
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


def max(array):
    """
    Takes the maximun value of the array.

    This function has the same behavior as
    ``np.max(array)`` or ``torch.max(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.max(array)
    elif isinstance(array, np.ndarray):
        return np.max(array)
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


def conjugate(array):
    """
    Conjugate the array

    This function has the same behavior as
    ``np.conjugate(array)`` or ``torch.conj(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.conj(array)
    elif isinstance(array, np.ndarray):
        return np.conjugate(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def real(array):
    """
    Takes the real part of the array

    This function has the same behavior as
    ``np.real(array)`` or ``torch.real(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.real(array)
    elif isinstance(array, np.ndarray):
        return np.real(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def imag(array):
    """
    Takes the imag part of the array

    This function has the same behavior as ``np.imag(array)`` or ``torch.imag(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.imag(array)
    elif isinstance(array, np.ndarray):
        return np.imag(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def roll(array, shifts: List[int], axis: List[int]):
    """
    Roll array elements along a given axis.

    This function has the same behavior as ``np.roll(array)`` or ``torch.roll(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.roll(array, shifts=shifts, dims=axis)
    elif isinstance(array, np.ndarray):
        return np.roll(array, shift=shifts, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def to(
    array,
    backend: Optional[str] = None,
    dtype: Optional[DType] = None,
    device: Optional[Union[str, Device]] = None,
):
    """Convert the array to the specified backend, dtype, and device"""
    if isinstance(array, TorchTensor):
        if backend is None:
            backend = "torch"

        if dtype is None:
            dtype = array.dtype
        if device is None:
            device = array.device
        if isinstance(device, str):
            device = torch.device(device)

        if backend == "torch":
            return array.to(dtype=dtype).to(device=device)

        elif backend == "numpy":
            if torch_jit_is_scripting():
                raise ValueError("cannot call numpy conversion when torch-scripting")
            else:
                return array.detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown backend: {backend}")

    elif isinstance(array, np.ndarray):
        if backend is None:
            backend = "numpy"

        if backend == "numpy":
            return np.array(array, dtype=dtype)

        elif backend == "torch":
            return torch.tensor(array, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
