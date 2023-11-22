"""
Module containing dispatch functions for numpy/torch CG combination operations.
"""
from typing import List, Optional, Union
import numpy as np

from ._cg_cache import HAS_MOPS

if HAS_MOPS:
    from mops import sparse_accumulation_of_products as sap


try:
    import torch
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


# ============ CG combinations  ============


def combine_arrays(
    arr_1: Union[np.ndarray, TorchTensor],
    arr_2: Union[np.ndarray, TorchTensor],
    lam: int,
    cg_cache,
    return_empty_array: bool = False,
) -> Union[np.ndarray, TorchTensor]:
    """
    Couples arrays `arr_1` and `arr_2` corresponding to the irreducible
    spherical components of 2 angular channels l1 and l2 using the appropriate
    Clebsch-Gordan coefficients. As l1 and l2 can be combined to form multiple
    lambda channels, this function returns the coupling to a single specified
    channel `lambda`. The angular channels l1 and l2 are inferred from the size
    of the components axis (axis 1) of the input arrays.

    `arr_1` has shape (n_i, 2 * l1 + 1, n_p) and `arr_2` has shape (n_i, 2 * l2
    + 1, n_q). n_i is the number of samples, n_p and n_q are the number of
    properties in each array. The number of samples in each array must be the
    same.

    The ouput array has shape (n_i, 2 * lambda + 1, n_p * n_q), where lambda is
    the input parameter `lam`.

    The Clebsch-Gordan coefficients are cached in `cg_cache`. Currently, these
    must be produced by the ClebschGordanReal class in this module. These
    coefficients can be stored in either sparse dictionaries or dense arrays.

    The combination operation is dispatched such that numpy arrays or torch
    tensors are automatically handled.

    `return_empty_array` can be used to return an empty array of the correct
    shape, without performing the CG combination step. This can be useful for
    probing the outputs of CG iterations in terms of metadata without the
    computational cost of performing the CG combinations - i.e. using the
    function :py:func:`combine_single_center_to_body_order_metadata_only`.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: either a sparse dictionary with keys (m1, m2, mu) and array
        values being sparse blocks of shape <TODO: fill out>, or a dense array
        of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lam + 1)].

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]
    """
    # If just precomputing metadata, return an empty array
    if return_empty_array:
        return sparse_combine(arr_1, arr_2, lam, cg_cache, return_empty_array=True)

    # Otherwise, perform the CG combination
    # Spare CG cache
    if cg_cache.sparse:
        return sparse_combine(arr_1, arr_2, lam, cg_cache, return_empty_array=False)

    # Dense CG cache
    return dense_combine(arr_1, arr_2, lam, cg_cache)


def sparse_combine(
    arr_1: Union[np.ndarray, TorchTensor],
    arr_2: Union[np.ndarray, TorchTensor],
    lam: int,
    cg_cache,
    return_empty_array: bool = False,
) -> Union[np.ndarray, TorchTensor]:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using sparse
    operations. The angular channel of each block is inferred from the size of
    its component axis, and the blocks are combined to the desired output
    angular channel `lam` using the appropriate Clebsch-Gordan coefficients.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: sparse dictionary with keys (m1, m2, mu) and array values
        being sparse blocks of shape <TODO: fill out>

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]
    """
    # Samples dimensions must be the same
    assert arr_1.shape[0] == arr_2.shape[0]

    # Infer l1 and l2 from the len of the length of axis 1 of each tensor
    l1 = (arr_1.shape[1] - 1) // 2
    l2 = (arr_2.shape[1] - 1) // 2

    # Define other useful dimensions
    n_i = arr_1.shape[0]  # number of samples
    n_p = arr_1.shape[2]  # number of properties in arr_1
    n_q = arr_2.shape[2]  # number of properties in arr_2

    if return_empty_array:  # used when only computing metadata
        return zeros_like((n_i, 2 * lam + 1, n_p * n_q), like=arr_1)

    if isinstance(arr_1, np.ndarray) and HAS_MOPS:
        # Reshape
        arr_1 = np.repeat(arr_1[:, :, :, None], n_q, axis=3).reshape(
            n_i, 2 * l1 + 1, n_p * n_q
        )
        arr_2 = np.repeat(arr_2[:, :, None, :], n_p, axis=2).reshape(
            n_i, 2 * l2 + 1, n_p * n_q
        )

        arr_1 = swapaxes(arr_1, 1, 2).reshape(n_i * n_p * n_q, 2 * l1 + 1)
        arr_2 = swapaxes(arr_2, 1, 2).reshape(n_i * n_p * n_q, 2 * l2 + 1)

        # Do SAP
        arr_out = sap(
            arr_1, arr_2, *cg_cache._coeffs[(l1, l2, lam)], output_size=2 * lam + 1
        )
        assert arr_out.shape == (n_i * n_p * n_q, 2 * lam + 1)

        # Reshape back
        arr_out = arr_out.reshape(n_i, n_p * n_q, 2 * lam + 1)
        arr_out = swapaxes(arr_out, 1, 2)

        return arr_out

    if isinstance(arr_1, np.ndarray) or isinstance(arr_1, TorchTensor):
        # Initialise output array
        arr_out = zeros_like((n_i, 2 * lam + 1, n_p * n_q), like=arr_1)

        # Get the corresponding Clebsch-Gordan coefficients
        cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

        # Fill in each mu component of the output array in turn
        for m1, m2, mu in cg_coeffs.keys():
            # Broadcast arrays, multiply together and with CG coeff
            arr_out[:, mu, :] += (
                arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeffs[(m1, m2, mu)]
            ).reshape(n_i, n_p * n_q)

        return arr_out

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def dense_combine(
    arr_1: Union[np.ndarray, TorchTensor],
    arr_2: Union[np.ndarray, TorchTensor],
    lam: int,
    cg_cache,
) -> Union[np.ndarray, TorchTensor]:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using a dense
    operation. The angular channel of each block is inferred from the size of
    its component axis, and the blocks are combined to the desired output
    angular channel `lam` using the appropriate Clebsch-Gordan coefficients.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lam: int value of the resulting coupled channel
    :param cg_cache: dense array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lam +
        1)]

    :returns: array of shape [n_samples, (2*lam+1), q_properties * p_properties]
    """
    if isinstance(arr_1, np.ndarray) or isinstance(arr_1, TorchTensor):
        # Infer l1 and l2 from the len of the length of axis 1 of each tensor
        l1 = (arr_1.shape[1] - 1) // 2
        l2 = (arr_2.shape[1] - 1) // 2
        cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

        # (samples None None l1_mu q) * (samples l2_mu p None None) -> (samples l2_mu p l1_mu q)
        # we broadcast it in this way so we only need to do one swapaxes in the next step
        arr_out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]

        # (samples l2_mu p l1_mu q) -> (samples q p l1_mu l2_mu)
        arr_out = swapaxes(arr_out, 1, 4)

        # samples (q p l1_mu l2_mu) -> (samples (q p) (l1_mu l2_mu))
        arr_out = arr_out.reshape(
            -1, arr_1.shape[2] * arr_2.shape[2], arr_1.shape[1] * arr_2.shape[1]
        )

        # (l1_mu l2_mu lam_mu) -> ((l1_mu l2_mu) lam_mu)
        cg_coeffs = cg_coeffs.reshape(-1, 2 * lam + 1)

        # (samples (q p) (l1_mu l2_mu)) @ ((l1_mu l2_mu) lam_mu) -> samples (q p) lam_mu
        arr_out = arr_out @ cg_coeffs

        # (samples (q p) lam_mu) -> (samples lam_mu (q p))
        return swapaxes(arr_out, 1, 2)

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


# ============ Other functions  ============


def unique(array, axis: Optional[int] = None):
    """Find the unique elements of an array."""
    if isinstance(array, TorchTensor):
        return torch.unique(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, axis=axis)


def int_range_like(min_val, max_val, like):
    """Returns an array of integers from min to max, non-inclusive, based on the
    type of `like`"""
    if isinstance(like, TorchTensor):
        return torch.arange(int_list, dtype=torch.int64, device=like.device)
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
        return torch.all(array)
    elif isinstance(array, np.ndarray):
        return np.all(array)
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


def swapaxes(array, axis0: int, axis1: int):
    """Swaps axes of an array."""
    if isinstance(array, TorchTensor):
        return torch.swapaxes(array, axis0, axis1)
    elif isinstance(array, np.ndarray):
        return np.swapaxes(array, axis0, axis1)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
