"""
Module for computing Clebsch-gordan iterations with equistore TensorMaps.
"""
from typing import Sequence
import numpy as np

import wigners

from equistore.core import Labels, TensorBlock, TensorMap


def _clebsch_gordan_combine(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    lamb: int,
    cg_cache,
) -> np.ndarray:
    """
    Couples arrays corresponding to the irreducible spherical components of 2
    angular channels l1 and l2 using the appropriate Clebsch-Gordan
    coefficients. As l1 and l2 can be combined to form multiple lambda channels,
    this function returns the coupling to a single specified channel `lambda`.

    `arr_1` has shape (n_i, 2 * l1 + 1, n_p) and `arr_2` has shape (n_i, 2 * l2
    + 1, n_q). n_i is the number of samples, n_p and n_q are the number of
    properties in each array. The number of samples in each array must be the
    same.

    The ouput array has shape (n_i, 2 * lambda + 1, n_p * n_q), where lambda is
    the input parameter `lamb`.

    The Clebsch-Gordan coefficients are cached in `cg_cache`. Currently, these
    must be produced by the ClebschGordanReal class in this module.
    """
    # Check the first dimension of the arrays are the same (i.e. same samples)
    assert arr_1.shape[0] == arr_2.shape[0]

    # Define useful dimensions
    n_i = arr_1.shape[0]  # number of samples
    n_p = arr_1.shape[2]  # number of properties in arr_1
    n_q = arr_2.shape[2]  # number of properties in arr_2

    # Infer l1 and l2 from the len of the lenght of axis 1 of each tensor
    l1 = int((arr_1.shape[1] - 1) / 2)
    l2 = int((arr_2.shape[1] - 1) / 2)

    # Get the corresponding Clebsch-Gordan coefficients
    cg_coeffs = cg_cache.coeffs[(l1, l2, lamb)]

    # Initialise output array
    arr_out = np.zeros((n_i, 2 * lamb + 1, n_p * n_q))

    # Fill in each mu component of the output array in turn
    for mu in range(2 * lamb + 1):
        # Iterate over the Clebsch-Gordan coefficients for this mu
        for m1, m2, cg_coeff in cg_coeffs[mu]:
            # Broadcast arrays, multiply together and with CG coeff
            out_arr[:, mu, :] = (
                arr_1[:, m1, :, None] * arr_2[:, m2, None, :] * cg_coeff
            ).reshape(n_i, n_p * n_q)

    return arr_out


class ClebschGordanReal:
    """
    Class for computing Clebsch-Gordan coefficients for real spherical
    harmonics.
    """

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.coeffs = ClebschGordanReal.build_coeff_dict(self.l_max)

    @staticmethod
    def build_coeff_dict(l_max: int):
        """
        Builds a dictionary of Clebsch-Gordan coefficients for all possible
        combination of l1 and l2, up to l_max.
        """
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        coeff_dict = {}
        for L in range(0, l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for L in range(max(l1, l2) - min(l1, l2), min(l_max, (l1 + l2)) + 1):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[L].T

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    coeff_dict[(l1, l2, L)] = new_cg

        return coeff_dict


def _real2complex(L: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.
    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | L M >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over M. The matrix is real.
    For example, using the relation:
        | l1 l2 L M > = \sum_{m1, m2} <l1 m1 l2 m2 | L M > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics L from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        L: l number For the third set of spherical harmonics
    Returns:
        cg: CG matrix for transforming complex-valued spherical harmonics
    >>> from scipy.special import sph_harm
    >>> import numpy as np
    >>> import wigners
    ...
    >>> C_112 = _complex_clebsch_gordan_matrix(1, 1, 2)
    >>> comp_sph_1 = np.array([
    ... sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1+1)
    ... ])
    >>> comp_sph_2 = np.array([sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1+1)])
    >>> # obtain the (unnormalized) spherical harmonics
    >>> # with l = 2 by contraction over m1 and m2
    >>> comp_sph_2_u = np.einsum("ijk,i,j->k", C_112, comp_sph_1, comp_sph_2)
    >>> # we can check that they differ from the spherical harmonics
    >>> # by a constant factor
    >>> comp_sph_2 = np.array([sph_harm(m, 2, 0.2, 0.2) for m in range(-2, 2+1)])
    >>> ratio = comp_sph_2 / comp_sph_2_u
    >>> np.allclose(ratio[0], ratio)
    True
    """
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)



# ===== For writing a dense version in the future? =====

# def _clebsch_gordan_combine_dense(
#     l1_mu_values,#: Array[samples, 2 * l1 + 1, q_properties], # mu values for l1
#     l2_mu_values,#: Array[samples, 2 * l2 + 1, p_properties], # mu values for l2
#     lam: int,
#     cg_cache,#: Array[(2 * l1 +1) * (2 * l2 +1), (2 * lam + 1)]
#     ) -> None: #Array[samples, 2 * lam + 1, q_properties * p_properties]:
#     """
#     :param l1_mu_values: mu values for l1
#     :param l2_mu_values: mu values for l2
#     :returns lam_mu_values: of shape [samples, (2 * l1 + 1)* (2 * l2 + 1), q_properties, p_properties]
#     """
#     # 
#     #l1l2_mu_values = l1_mu_values[:,  None,  :] * l2_mu_values[:, :, None, :]
#     #return einops.einsum(l1_mu_values, l2_mu_values, cg_cache, "samples l1_mu q_properties, samples l2_mu p_properties,  l1_mu l2_mu lam_mu -> samples lam_mu (q_properties p_properties)")

#     # more readable subscript
#     #"samples l1_mu q_properties, samples l2_mu p_properties,  l1_mu l2_mu lam_mu -> samples lam_mu (q_properties p_properties)"
#     return np.einsum("slq, skp, lkL -> sLqp", l1_mu_values, l2_mu_values, cg_cache).reshape(l1_mu_values.shape[0], 2*lam+1, -1)