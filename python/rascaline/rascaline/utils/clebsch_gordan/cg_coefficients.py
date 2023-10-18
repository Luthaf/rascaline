"""
Module that stores the ClebschGordanReal class for computing CG coefficients.
"""
import numpy as np
import wigners


class ClebschGordanReal:
    """
    Class for computing Clebsch-Gordan coefficients for real spherical
    harmonics.

    Stores the coefficients in a dictionary in the `self.coeffs` attribute,
    which is built at initialization. This dictionary has the form:

    {
        (l1, l2, lambda): [
            np.ndarray([m1, m2, cg]),
            ...
            for m1 in range(-l1, l1 + 1),
            for m2 in range(-l2, l2 + 1),
        ],
        ...
        for lambda in range(0, l)
    }

    where `cg`, i.e. the third value in each array, is the Clebsch-Gordan
    coefficient that describes the combination of the `m1` irreducible
    component of the `l1` angular channel and the `m2` irreducible component of
    the `l2` angular channel into the irreducible tensor of order `lambda`.
    """

    def __init__(self, lambda_max: int, sparse: bool = True):
        self._lambda_max = lambda_max
        self._sparse = sparse
        self._coeffs = ClebschGordanReal.build_coeff_dict(
            self._lambda_max, self._sparse
        )

    @property
    def lambda_max(self):
        return self._lambda_max

    @property
    def sparse(self):
        return self._sparse

    @property
    def coeffs(self):
        return self._coeffs

    @staticmethod
    def build_coeff_dict(lambda_max: int, sparse: bool):
        """
        Builds a dictionary of Clebsch-Gordan coefficients for all possible
        combination of l1 and l2, up to lambda_max.
        """
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        coeff_dict = {}
        for lam in range(0, lambda_max + 1):
            c2r[lam] = _complex2real(lam)
            r2c[lam] = _real2complex(lam)

        for l1 in range(lambda_max + 1):
            for l2 in range(lambda_max + 1):
                for lam in range(
                    max(l1, l2) - min(l1, l2), min(lambda_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, lam)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[lam].T

                    if (l1 + l2 + lam) % 2 == 0:
                        cg_l1l2lam = np.real(real_cg)
                    else:
                        cg_l1l2lam = np.imag(real_cg)

                    if sparse:
                        # if sparse we make a dictionary out of the matrix
                        nonzeros_cg_coeffs_idx = np.where(np.abs(cg_l1l2lam) > 1e-15)
                        cg_l1l2lam = {
                            (m1, m2, mu): cg_l1l2lam[m1, m2, mu]
                            for m1, m2, mu in zip(*nonzeros_cg_coeffs_idx)
                        }
                    coeff_dict[(l1, l2, lam)] = cg_l1l2lam

        return coeff_dict


def _real2complex(lam: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order ``lam``.

    This is meant to be applied to the left: ``real2complex @ [-lam, ...,
    +lam]``.

    See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form for details
    on the convention for how these tranformations are defined.
    """
    result = np.zeros((2 * lam + 1, 2 * lam + 1), dtype=np.complex128)
    inv_sqrt_2 = 1.0 / np.sqrt(2)
    i_sqrt_2 = 1j / np.sqrt(2)
    for m in range(-lam, lam + 1):
        if m < 0:
            # Positve part
            result[lam + m, lam + m] = +i_sqrt_2
            # Negative part
            result[lam - m, lam + m] = -i_sqrt_2 * ((-1) ** m)

        if m == 0:
            result[lam, lam] = +1.0

        if m > 0:
            # Negative part
            result[lam - m, lam + m] = +inv_sqrt_2
            # Positive part
            result[lam + m, lam + m] = +inv_sqrt_2 * ((-1) ** m)

    return result


def _complex2real(lam: int) -> np.ndarray:
    """
    Converts from complex to real spherical harmonics. This is just given by the
    conjugate tranpose of the real->complex transformation matrices.
    """
    return np.conjugate(_real2complex(lam)).T


def _complex_clebsch_gordan_matrix(l1, l2, lam):
    r"""clebsch-gordan matrix
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
        < l1 m1 l2 m2 | lam mu >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over mu. The matrix is real.
    For example, using the relation:
        | l1 l2 lam mu > = \sum_{m1, m2} <l1 m1 l2 m2 | lam mu > | l1 m1 > | l2 m2 >
    (https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients, section
    "Formal definition of Clebsch-Gordan coefficients", eq 2)
    one can obtain the spherical harmonics lam from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:
    Args:
        l1: l number for the first set of spherical harmonics
        l2: l number for the second set of spherical harmonics
        lam: l number For the third set of spherical harmonics
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
    if np.abs(l1 - l2) > lam or np.abs(l1 + l2) < lam:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * lam + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, lam)
