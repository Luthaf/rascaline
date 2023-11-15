"""
Module containing dispatch functions for numpy/torch CG combination operations.
"""
from typing import Union
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


def _combine_arrays(
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
        return _sparse_combine(arr_1, arr_2, lam, cg_cache, True)

    # Otherwise, perform the CG combination
    # Spare CG cache
    if cg_cache.sparse:
        return _sparse_combine(arr_1, arr_2, lam, cg_cache, False)

    # Dense CG cache
    return _dense_combine(arr_1, arr_2, lam, cg_cache)


def _sparse_combine(
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
    if isinstance(arr_1, np.ndarray):
        # Samples dimensions must be the same
        assert arr_1.shape[0] == arr_2.shape[0]

        # Define other useful dimensions
        n_i = arr_1.shape[0]  # number of samples
        n_p = arr_1.shape[2]  # number of properties in arr_1
        n_q = arr_2.shape[2]  # number of properties in arr_2

        # Infer l1 and l2 from the len of the length of axis 1 of each tensor
        l1 = (arr_1.shape[1] - 1) // 2
        l2 = (arr_2.shape[1] - 1) // 2

        # Initialise output array
        arr_out = np.zeros((n_i, 2 * lam + 1, n_p * n_q))

        if return_empty_array:
            return arr_out

        # Get the corresponding Clebsch-Gordan coefficients
        cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

        # Fill in each mu component of the output array in turn
        for m1, m2, mu in cg_coeffs.keys():
            # Broadcast arrays, multiply together and with CG coeff
            arr_out[:, mu, :] += (
                arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeffs[(m1, m2, mu)]
            ).reshape(n_i, n_p * n_q)

        return arr_out

    elif isinstance(arr_1, TorchTensor):
        pass

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def _dense_combine(
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
    if isinstance(arr_1, np.ndarray):
        # Infer l1 and l2 from the len of the length of axis 1 of each tensor
        l1 = (arr_1.shape[1] - 1) // 2
        l2 = (arr_2.shape[1] - 1) // 2
        cg_coeffs = cg_cache.coeffs[(l1, l2, lam)]

        # (samples None None l1_mu q) * (samples l2_mu p None None) -> (samples l2_mu p l1_mu q)
        # we broadcast it in this way so we only need to do one swapaxes in the next step
        arr_out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]

        # (samples l2_mu p l1_mu q) -> (samples q p l1_mu l2_mu)
        arr_out = arr_out.swapaxes(1, 4)

        # samples (q p l1_mu l2_mu) -> (samples (q p) (l1_mu l2_mu))
        arr_out = arr_out.reshape(
            -1, arr_1.shape[2] * arr_2.shape[2], arr_1.shape[1] * arr_2.shape[1]
        )

        # (l1_mu l2_mu lam_mu) -> ((l1_mu l2_mu) lam_mu)
        cg_coeffs = cg_coeffs.reshape(-1, 2 * lam + 1)

        # (samples (q p) (l1_mu l2_mu)) @ ((l1_mu l2_mu) lam_mu) -> samples (q p) lam_mu
        arr_out = arr_out @ cg_coeffs

        # (samples (q p) lam_mu) -> (samples lam_mu (q p))
        return arr_out.swapaxes(1, 2)

    elif isinstance(arr_1, TorchTensor):
        pass

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
