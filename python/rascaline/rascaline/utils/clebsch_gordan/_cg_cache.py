"""
Module that stores the ClebschGordanReal class for computing and caching Clebsch
Gordan coefficients for use in CG combinations.
"""
from typing import Union

import numpy as np
import wigners

from . import _dispatch


try:
    from mops import sparse_accumulation_of_products as sap  # noqa F401

    HAS_MOPS = True
except ImportError:
    HAS_MOPS = False

try:
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


# =================================
# ===== ClebschGordanReal class
# =================================


class ClebschGordanReal:
    """
    Class for computing Clebsch-Gordan coefficients for real spherical
    harmonics.

    Stores the coefficients in a dictionary in the `self.coeffs` attribute,
    which is built at initialization. There are 3 current use cases for the
    format of these coefficients. By default, sparse accumulation of products is
    performed, whether or not Mops is installed.

    Case 1: standard sparse format.

    Each dictionary entry is a dictionary with entries for each (m1, m2, mu)
    combination.

    {
        (l1, l2, lambda): {
            (m1, m2, mu) : cg_{m1, m2, mu}^{l1, l2, lambda}
            for m1 in range(-l1, l1 + 1),
            for m2 in range(-l2, l2 + 1),
        },
        ...
        for l1 in range(0, l1_list)
        for l2 in range(0, l2_list)
        for lambda in range(0, range(|l1 - l2|, ..., |l1 + l2|))
    }

    Case 2: standard dense format.

    Each dictionary entry is a dense array with shape (2 * l1 + 1, 2 * l2 + 1, 2
    * lambda + 1).

    {
        (l1, l2, lambda):
            array(
                cg_{m1, m2, mu}^{l1, l2, lambda}
                ...
                for m1 in range(-l1, l1 + 1),
                for m2 in range(-l2, l2 + 1),
                for mu in range(-lambda, lambda + 1),

                shape=(2 * l1 + 1, 2 * l2 + 1, 2 * lambda + 1),
            )
        ...
        for l1 in range(0, l1_list)
        for l2 in range(0, l2_list)
        for lambda in range(0, range(|l1 - l2|, ..., |l1 + l2|))
    }

    Case 3: MOPS sparse format.

    Each dictionary entry contains a tuple with four 1D arrays, corresponding to
    the CG coeffs and m1, m2, mu indices respectively. All of these arrays are
    sorted according to the mu index. This format is used for Sparse
    Accumulation of Products (SAP) as implemented in MOPS. See
    https://github.com/lab-cosmo/mops .

    {
        (l1, l2, lambda):
            (
                [
                    cg_{m1, m2, mu}^{l1, l2, lambda}
                    ...
                    for m1 in range(-l1, l1 + 1),
                    for m2 in range(-l2, l2 + 1),
                    for mu in range(-lambda, lambda + 1)
                ],
                [
                    m1 for m1 in range(-l1, l1 + 1),
                ],
                [
                    m2 for m2 in range(-l2, l2 + 1),
                ],
                [
                    mu for mu in range(-lambda, lambda + 1),
                ],
            )


    }

    where `cg_{m1, m2, mu}^{l1, l2, lambda}` is the Clebsch-Gordan coefficient
    that describes the combination of the `m1` irreducible component of the `l1`
    angular channel and the `m2` irreducible component of the `l2` angular
    channel into the irreducible tensor of order `lambda`. In all cases, these
    correspond to the non-zero CG coefficients, i.e. those in the range |-l,
    ..., +l| for each angular order l in {l1, l2, lambda}.

    :param lambda_max: maximum lambda value to compute CG coefficients for.
    :param sparse: whether to store the CG coefficients in sparse format.
    :param use_mops: whether to store the CG coefficients in MOPS sparse format.
        This is recommended as the default for sparse accumulation, but can only
        be used if Mops is installed.
    """

    def __init__(self, lambda_max: int, sparse: bool = True, use_mops: bool = HAS_MOPS):
        self._lambda_max = lambda_max
        self._sparse = sparse

        if sparse:
            if not HAS_MOPS:
                # TODO: provide a warning once Mops is fully ready
                # import warnings
                # warnings.warn(
                #     "It is recommended to use MOPS for sparse accumulation. "
                #     " This can be installed with ``pip install"
                #     " git+https://github.com/lab-cosmo/mops`."
                #     " Falling back to numpy for now."
                # )
                self._use_mops = False
            else:
                self._use_mops = True

        else:
            # TODO: provide a warning once Mops is fully ready
            # if HAS_MOPS:
            #     import warnings
            #     warnings.warn(
            #         "Mops is installed, but not being used"
            #         " as dense operations chosen."
            #     )
            self._use_mops = False

        self._coeffs = ClebschGordanReal.build_coeff_dict(
            self._lambda_max,
            self._sparse,
            self._use_mops,
        )

    @property
    def lambda_max(self):
        return self._lambda_max

    @property
    def sparse(self):
        return self._sparse

    @property
    def use_mops(self):
        return self._use_mops

    @property
    def coeffs(self):
        return self._coeffs

    @staticmethod
    def build_coeff_dict(lambda_max: int, sparse: bool, use_mops: bool):
        """
        Builds a dictionary of Clebsch-Gordan coefficients for all possible
        combination of l1 and l2, up to lambda_max.
        """
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        coeff_dict = {}
        for lambda_ in range(0, lambda_max + 1):
            c2r[lambda_] = _complex2real(lambda_)
            r2c[lambda_] = _real2complex(lambda_)

        for l1 in range(lambda_max + 1):
            for l2 in range(lambda_max + 1):
                for lambda_ in range(
                    max(l1, l2) - min(l1, l2), min(lambda_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, lambda_)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[lambda_].T

                    if (l1 + l2 + lambda_) % 2 == 0:
                        cg_l1l2lam = np.real(real_cg)
                    else:
                        cg_l1l2lam = np.imag(real_cg)

                    if sparse:
                        # Find the m1, m2, mu idxs of the nonzero CG coeffs
                        nonzeros_cg_coeffs_idx = np.where(np.abs(cg_l1l2lam) > 1e-15)
                        if use_mops:
                            # Store CG coeffs in a specific format for use in
                            # MOPS. Here we need the m1, m2, mu, and CG coeffs
                            # to be stored as separate 1D arrays.
                            m1_arr, m2_arr, mu_arr, C_arr = [], [], [], []
                            for m1, m2, mu in zip(*nonzeros_cg_coeffs_idx):
                                m1_arr.append(m1)
                                m2_arr.append(m2)
                                mu_arr.append(mu)
                                C_arr.append(cg_l1l2lam[m1, m2, mu])

                            # Reorder the arrays based on sorted mu values
                            mu_idxs = np.argsort(mu_arr)
                            m1_arr = np.array(m1_arr)[mu_idxs]
                            m2_arr = np.array(m2_arr)[mu_idxs]
                            mu_arr = np.array(mu_arr)[mu_idxs]
                            C_arr = np.array(C_arr)[mu_idxs]
                            cg_l1l2lam = (C_arr, m1_arr, m2_arr, mu_arr)
                        else:
                            # Otherwise fall back to torch/numpy and store as
                            # sparse dicts.
                            cg_l1l2lam = {
                                (m1, m2, mu): cg_l1l2lam[m1, m2, mu]
                                for m1, m2, mu in zip(*nonzeros_cg_coeffs_idx)
                            }

                    # Store
                    coeff_dict[(l1, l2, lambda_)] = cg_l1l2lam

        return coeff_dict


# ============================
# ===== Helper functions
# ============================


def _real2complex(lambda_: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order ``lambda_``.

    This is meant to be applied to the left: ``real2complex @ [-lambda_, ...,
    +lambda_]``.

    See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form for details
    on the convention for how these tranformations are defined.
    """
    result = np.zeros((2 * lambda_ + 1, 2 * lambda_ + 1), dtype=np.complex128)
    inv_sqrt_2 = 1.0 / np.sqrt(2)
    i_sqrt_2 = 1j / np.sqrt(2)
    for m in range(-lambda_, lambda_ + 1):
        if m < 0:
            # Positve part
            result[lambda_ + m, lambda_ + m] = +i_sqrt_2
            # Negative part
            result[lambda_ - m, lambda_ + m] = -i_sqrt_2 * ((-1) ** m)

        if m == 0:
            result[lambda_, lambda_] = +1.0

        if m > 0:
            # Negative part
            result[lambda_ - m, lambda_ + m] = +inv_sqrt_2
            # Positive part
            result[lambda_ + m, lambda_ + m] = +inv_sqrt_2 * ((-1) ** m)

    return result


def _complex2real(lambda_: int) -> np.ndarray:
    """
    Converts from complex to real spherical harmonics. This is just given by the
    conjugate tranpose of the real->complex transformation matrices.
    """
    return np.conjugate(_real2complex(lambda_)).T


def _complex_clebsch_gordan_matrix(l1, l2, lambda_):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | lambda_ mu >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over mu. The matrix is real.
    For example, using the relation:
        | l1 l2 lambda_ mu > =
            \sum_{m1, m2}
            <l1 m1 l2 m2 | lambda_ mu > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics lambda_ from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        lambda_: l number For the third set of spherical harmonics
    Returns:
        cg: CG matrix for transforming complex-valued spherical harmonics
    >>> from scipy.special import sph_harm
    >>> import numpy as np
    >>> import wigners
    >>> C_112 = _complex_clebsch_gordan_matrix(1, 1, 2)
    >>> comp_sph_1 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1 + 1)])
    >>> comp_sph_2 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1 + 1)])
    >>> # obtain the (unnormalized) spherical harmonics
    >>> # with l = 2 by contraction over m1 and m2
    >>> comp_sph_2_u = np.einsum("ijk,i,j->k", C_112, comp_sph_1, comp_sph_2)
    >>> # we can check that they differ from the spherical harmonics
    >>> # by a constant factor
    >>> comp_sph_2 = np.array([sph_harm(m, 2, 0.2, 0.2) for m in range(-2, 2 + 1)])
    >>> ratio = comp_sph_2 / comp_sph_2_u
    >>> np.allclose(ratio[0], ratio)
    True
    """
    if np.abs(l1 - l2) > lambda_ or np.abs(l1 + l2) < lambda_:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * lambda_ + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, lambda_)


# =================================================
# ===== Functions for performing CG combinations
# =================================================


def combine_arrays(
    arr_1: Union[np.ndarray, TorchTensor],
    arr_2: Union[np.ndarray, TorchTensor],
    lambda_: int,
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
    the input parameter `lambda_`.

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
    :param lambda_: int value of the resulting coupled channel
    :param cg_cache: either a sparse dictionary with keys (m1, m2, mu) and array
        values being sparse blocks of shape <TODO: fill out>, or a dense array
        of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lambda_ + 1)].

    :returns: array of shape [n_samples, (2*lambda_+1), q_properties * p_properties]
    """
    # If just precomputing metadata, return an empty array
    if return_empty_array:
        return sparse_combine(arr_1, arr_2, lambda_, cg_cache, return_empty_array=True)

    # Otherwise, perform the CG combination
    # Spare CG cache
    if cg_cache.sparse:
        return sparse_combine(arr_1, arr_2, lambda_, cg_cache, return_empty_array=False)

    # Dense CG cache
    return dense_combine(arr_1, arr_2, lambda_, cg_cache)


def sparse_combine(
    arr_1: Union[np.ndarray, TorchTensor],
    arr_2: Union[np.ndarray, TorchTensor],
    lambda_: int,
    cg_cache,
    return_empty_array: bool = False,
) -> Union[np.ndarray, TorchTensor]:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using sparse
    operations. The angular channel of each block is inferred from the size of
    its component axis, and the blocks are combined to the desired output
    angular channel `lambda_` using the appropriate Clebsch-Gordan coefficients.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lambda_: int value of the resulting coupled channel
    :param cg_cache: sparse dictionary with keys (m1, m2, mu) and array values
        being sparse blocks of shape <TODO: fill out>

    :returns: array of shape [n_samples, (2*lambda_+1), q_properties * p_properties]
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
        return _dispatch.zeros_like((n_i, 2 * lambda_ + 1, n_p * n_q), like=arr_1)

    if isinstance(arr_1, np.ndarray) and HAS_MOPS:
        # Reshape
        arr_1 = np.repeat(arr_1[:, :, :, None], n_q, axis=3).reshape(
            n_i, 2 * l1 + 1, n_p * n_q
        )
        arr_2 = np.repeat(arr_2[:, :, None, :], n_p, axis=2).reshape(
            n_i, 2 * l2 + 1, n_p * n_q
        )

        arr_1 = _dispatch.swapaxes(arr_1, 1, 2).reshape(n_i * n_p * n_q, 2 * l1 + 1)
        arr_2 = _dispatch.swapaxes(arr_2, 1, 2).reshape(n_i * n_p * n_q, 2 * l2 + 1)

        # Do SAP
        arr_out = sap(
            arr_1,
            arr_2,
            *cg_cache._coeffs[(l1, l2, lambda_)],
            output_size=2 * lambda_ + 1,
        )
        assert arr_out.shape == (n_i * n_p * n_q, 2 * lambda_ + 1)

        # Reshape back
        arr_out = arr_out.reshape(n_i, n_p * n_q, 2 * lambda_ + 1)
        arr_out = _dispatch.swapaxes(arr_out, 1, 2)

        return arr_out

    if isinstance(arr_1, np.ndarray) or isinstance(arr_1, TorchTensor):
        # Initialise output array
        arr_out = _dispatch.zeros_like((n_i, 2 * lambda_ + 1, n_p * n_q), like=arr_1)

        # Get the corresponding Clebsch-Gordan coefficients
        cg_coeffs = cg_cache.coeffs[(l1, l2, lambda_)]

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
    lambda_: int,
    cg_cache,
) -> Union[np.ndarray, TorchTensor]:
    """
    Performs a Clebsch-Gordan combination step on 2 arrays using a dense
    operation. The angular channel of each block is inferred from the size of
    its component axis, and the blocks are combined to the desired output
    angular channel `lambda_` using the appropriate Clebsch-Gordan coefficients.

    :param arr_1: array with the m values for l1 with shape [n_samples, 2 * l1 +
        1, n_q_properties]
    :param arr_2: array with the m values for l2 with shape [n_samples, 2 * l2 +
        1, n_p_properties]
    :param lambda_: int value of the resulting coupled channel
    :param cg_cache: dense array of shape [(2 * l1 +1) * (2 * l2 +1), (2 * lambda_ +
        1)]

    :returns: array of shape [n_samples, (2*lambda_+1), q_properties * p_properties]
    """
    if isinstance(arr_1, np.ndarray) or isinstance(arr_1, TorchTensor):
        # Infer l1 and l2 from the len of the length of axis 1 of each tensor
        l1 = (arr_1.shape[1] - 1) // 2
        l2 = (arr_2.shape[1] - 1) // 2
        cg_coeffs = cg_cache.coeffs[(l1, l2, lambda_)]

        # (samples None None l1_mu q) * (samples l2_mu p None None)
        # -> (samples l2_mu p l1_mu q) we broadcast it in this way
        # so we only need to do one swapaxes in the next step
        arr_out = arr_1[:, None, None, :, :] * arr_2[:, :, :, None, None]

        # (samples l2_mu p l1_mu q) -> (samples q p l1_mu l2_mu)
        arr_out = _dispatch.swapaxes(arr_out, 1, 4)

        # samples (q p l1_mu l2_mu) -> (samples (q p) (l1_mu l2_mu))
        arr_out = arr_out.reshape(
            -1,
            arr_1.shape[2] * arr_2.shape[2],
            arr_1.shape[1] * arr_2.shape[1],
        )

        # (l1_mu l2_mu lam_mu) -> ((l1_mu l2_mu) lam_mu)
        cg_coeffs = cg_coeffs.reshape(-1, 2 * lambda_ + 1)

        # (samples (q p) (l1_mu l2_mu)) @ ((l1_mu l2_mu) lam_mu)
        # -> samples (q p) lam_mu
        arr_out = arr_out @ cg_coeffs

        # (samples (q p) lam_mu) -> (samples lam_mu (q p))
        return _dispatch.swapaxes(arr_out, 1, 2)

    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
