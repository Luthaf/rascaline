r"""
.. _python-radial-basis:

Radial Basis
============

Radial basis functions :math:`R_{nl}(\boldsymbol{r})` are besides :ref:`atomic densities
<python-atomic-density>` :math:`\rho_i` the central ingredients to compute spherical
expansion coefficients :math:`\langle anlm\vert\rho_i\rangle`. Radial basis functions,
define how which the atomic density is projected. To be more precise, the actual basis
functions are of

.. math::

    B_{nlm}(\boldsymbol{r}) = R_{nl}(r)Y_{lm}(\hat{r}) \,,

where :math:`Y_{lm}(\hat{r})` are the real spherical harmonics evaluated at the point
:math:`\hat{r}`, i.e. at the spherical angles :math:`(\theta, \phi)` that determine the
orientation of the unit vector :math:`\hat{r} = \boldsymbol{r}/r`.

Radial basis are represented as different child class of
:py:class:`rascaline.utils.RadialBasisBase`: :py:class:`rascaline.utils.GtoBasis`,
:py:class:`rascaline.utils.MonomialBasis`, and
:py:class:`rascaline.utils.SphericalBesselBasis` are provided, and you can implement
your own by defining a new class.

.. autoclass:: rascaline.utils.RadialBasisBase
    :members:
    :show-inheritance:

.. autoclass:: rascaline.utils.GtoBasis
    :members:
    :show-inheritance:

.. autoclass:: rascaline.utils.MonomialBasis
    :members:
    :show-inheritance:

.. autoclass:: rascaline.utils.SphericalBesselBasis
    :members:
    :show-inheritance:
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


try:
    import scipy.integrate
    import scipy.optimize
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RadialBasisBase(ABC):
    r"""
    Base class to define radial basis and their evaluation.

    The class provides methods to evaluate the radial basis :math:`R_{nl}(r)` as well as
    its (numerical) derivative with respect to positions :math:`r`.

    :parameter integration_radius: Value up to which the radial integral should be
        performed. The usual value is :math:`\infty`.
    """

    def __init__(self, integration_radius: float):
        self.integration_radius = integration_radius

    @abstractmethod
    def compute(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute the ``n``/``l`` radial basis at all given ``integrand_positions``

        :param n: radial channel
        :param ell: angular channel
        :param integrand_positions: positions to evaluate the radial basis
        :returns: evaluated radial basis
        """

    def compute_derivative(
        self, n: int, ell: int, integrand_positions: np.ndarray
    ) -> np.ndarray:
        """Compute the derivative of the ``n``/``l`` radial basis at all given
        ``integrand_positions``

        This is used for radial integrals with delta-like atomic densities. If not
        defined in a child class, a numerical derivative based on finite differences of
        ``integrand_positions`` will be used instead.

        :param n: radial channel
        :param ell: angular channel
        :param integrand_positions: positions to evaluate the radial basis
        :returns: evaluated derivative of the radial basis
        """
        displacement = 1e-6
        mean_abs_positions = np.abs(integrand_positions).mean()

        if mean_abs_positions < 1.0:
            raise ValueError(
                "Numerically derivative of the radial integral can not be performed "
                "since positions are too small. Mean of the absolute positions is "
                f"{mean_abs_positions:.1e} but should be at least 1."
            )

        radial_basis_pos = self.compute(n, ell, integrand_positions + displacement / 2)
        radial_basis_neg = self.compute(n, ell, integrand_positions - displacement / 2)

        return (radial_basis_pos - radial_basis_neg) / displacement

    def compute_gram_matrix(
        self,
        max_radial: int,
        max_angular: int,
    ) -> np.ndarray:
        """Gram matrix of the current basis.

        :parameter max_radial: number of angular components
        :parameter max_angular: number of radial components
        :returns: orthonormalization matrix of shape
            ``(max_angular + 1, max_radial, max_radial)``
        """

        if not HAS_SCIPY:
            raise ValueError("Orthonormalization requires scipy!")

        # Gram matrix (also called overlap matrix or inner product matrix)
        gram_matrix = np.zeros((max_angular + 1, max_radial, max_radial))

        def integrand(
            integrand_positions: np.ndarray,
            n1: int,
            n2: int,
            ell: int,
        ) -> np.ndarray:
            return (
                integrand_positions**2
                * self.compute(n1, ell, integrand_positions)
                * self.compute(n2, ell, integrand_positions)
            )

        for ell in range(max_angular + 1):
            for n1 in range(max_radial):
                for n2 in range(max_radial):
                    gram_matrix[ell, n1, n2] = scipy.integrate.quad(
                        func=integrand,
                        a=0,
                        b=self.integration_radius,
                        args=(n1, n2, ell),
                    )[0]

        return gram_matrix

    def compute_orthonormalization_matrix(
        self,
        max_radial: int,
        max_angular: int,
    ) -> np.ndarray:
        """Compute orthonormalization matrix

        :parameter max_radial: number of angular components
        :parameter max_angular: number of radial components
        :returns: orthonormalization matrix of shape (max_angular + 1, max_radial,
            max_radial)
        """

        gram_matrix = self.compute_gram_matrix(max_radial, max_angular)

        # Get the normalization constants from the diagonal entries
        normalizations = np.zeros((max_angular + 1, max_radial))

        for ell in range(max_angular + 1):
            for n in range(max_radial):
                normalizations[ell, n] = 1 / np.sqrt(gram_matrix[ell, n, n])

                # Rescale orthonormalization matrix to be defined
                # in terms of the normalized (but not yet orthonormalized)
                # basis functions
                gram_matrix[ell, n, :] *= normalizations[ell, n]
                gram_matrix[ell, :, n] *= normalizations[ell, n]

        orthonormalization_matrix = np.zeros_like(gram_matrix)
        for ell in range(max_angular + 1):
            eigvals, eigvecs = np.linalg.eigh(gram_matrix[ell])
            orthonormalization_matrix[ell] = (
                eigvecs @ np.diag(np.sqrt(1.0 / eigvals)) @ eigvecs.T
            )

        # Rescale the orthonormalization matrix so that it
        # works with respect to the primitive (not yet normalized)
        # radial basis functions
        for ell in range(max_angular + 1):
            for n in range(max_radial):
                orthonormalization_matrix[ell, :, n] *= normalizations[ell, n]

        return orthonormalization_matrix


class GtoBasis(RadialBasisBase):
    r"""Primitive (not normalized nor orthonormalized) GTO radial basis.

    It is defined as

    .. math::

        R_{nl}(r) = R_n(r) = r^n e^{-\frac{r^2}{2\sigma_n^2}},

    where :math:`\sigma_n = \sqrt{n} r_\mathrm{cut}/n_\mathrm{max}` with
    :math:`r_\mathrm{cut}` being the ``cutoff`` and :math:`n_\mathrm{max}` the maximal
    number of radial components.

    :parameter cutoff: spherical cutoff for the radial basis
    :parameter max_radial: number of radial components
    """

    def __init__(self, cutoff, max_radial):
        # choosing infinity leads to problems when calculating the radial integral with
        # `quad`!
        super().__init__(integration_radius=5 * cutoff)
        self.max_radial = max_radial
        self.cutoff = cutoff
        self.sigmas = np.ones(self.max_radial, dtype=float)

        for n in range(1, self.max_radial):
            self.sigmas[n] = np.sqrt(n)
        self.sigmas *= self.cutoff / self.max_radial

    def compute(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return integrand_positions**n * np.exp(
            -0.5 * (integrand_positions / self.sigmas[n]) ** 2
        )

    def compute_derivative(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return n / integrand_positions * self.compute(
            n, ell, integrand_positions
        ) - integrand_positions / self.sigmas[n] ** 2 * self.compute(
            n, ell, integrand_positions
        )


class MonomialBasis(RadialBasisBase):
    r"""Monomial basis.

    Basis is consisting of functions

    .. math::
        R_{nl}(r) = r^{l+2n},

    where :math:`n` runs from :math:`0,1,...,n_\mathrm{max}-1`. These capture precisely
    the radial dependence if we compute the Taylor expansion of a generic function
    defined in 3D space.

    :parameter cutoff: spherical cutoff for the radial basis
    """

    def __init__(self, cutoff):
        super().__init__(integration_radius=cutoff)

    def compute(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return integrand_positions ** (ell + 2 * n)

    def compute_derivative(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return (ell + 2 * n) * integrand_positions ** (ell + 2 * n - 1)


class SphericalBesselBasis(RadialBasisBase):
    """Spherical Bessel functions used in the Laplacian eigenstate (LE) basis.

    :parameter cutoff: spherical cutoff for the radial basis
    :parameter max_radial: number of angular components
    :parameter max_angular: number of radial components
    """

    def __init__(self, cutoff, max_radial, max_angular):
        if not HAS_SCIPY:
            raise ValueError("SphericalBesselBasis requires scipy!")

        super().__init__(integration_radius=cutoff)

        self.max_radial = max_radial
        self.max_angular = max_angular
        self.roots = SphericalBesselBasis.compute_zeros(max_angular, max_radial)

    def compute(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return scipy.special.spherical_jn(
            ell,
            integrand_positions * self.roots[ell, n] / self.integration_radius,
        )

    def compute_derivative(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return (
            self.roots[ell, n]
            / self.integration_radius
            * scipy.special.spherical_jn(
                ell,
                integrand_positions * self.roots[ell, n] / self.integration_radius,
                derivative=True,
            )
        )
    
    @staticmethod
    def compute_zeros(
        max_radial: int, max_angular: int
    ):
        
        # Spherical Bessel zeros from the scipy cookbook
        # https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
        def Jn(r, n):
            return np.sqrt(np.pi / (2 * r)) * scipy.special.jv(n + 0.5, r)
        def Jn_zeros(n, nt):
            zeros_j = np.zeros((n + 1, nt), dtype=np.float64)
            zeros_j[0] = np.arange(1, nt + 1) * np.pi
            points = np.arange(1, nt + n + 1) * np.pi
            roots = np.zeros(nt + n, dtype=np.float64)
            for i in range(1, n + 1):
                for j in range(nt + n - i):
                    roots[j] = scipy.optimize.brentq(Jn, points[j], points[j + 1], (i,))
                points = roots
                zeros_j[i][:nt] = roots[:nt]
            return zeros_j
        
        return Jn_zeros(max_angular, max_radial)
