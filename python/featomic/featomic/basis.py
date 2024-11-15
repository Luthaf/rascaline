import abc
from typing import Dict, List, Optional

import numpy as np


try:
    import scipy.integrate
    import scipy.optimize
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RadialBasis(metaclass=abc.ABCMeta):
    """
    Base class representing a set of radial basis functions, indexed by a radial index
    ``n``.

    You can inherit from this class to define new custom radial basis function, by
    implementing the :py:meth:`compute_primitive` method. If needed,
    :py:meth:`finite_differences_derivative` can be used to compute the derivatives of a
    radial basis.

    Overriding :py:attr:`integration_radius` can be useful to control the integration
    radius when evaluating the radial integral numerically. See this :ref:`explanation
    <radial-integral>` for more information.

    If the new radial basis function has corresponding hyper parameters in the native
    calculators, you should also implement :py:meth:`get_hypers`.
    """

    def __init__(self, *, max_radial: int, radius: float):
        """
        :parameter max_radial: maximal radial basis index to include (there will be
            ``N = max_radial + 1`` basis functions overall)
        :parameter radius: radius of the radial basis. For local spherical expansions,
            this is typically the same as the spherical cutoff radius.
        """
        self.max_radial = int(max_radial)
        self.radius = float(radius)

        assert self.max_radial >= 0
        assert self.radius > 0.0
        self._orthonormalization_matrix = None

    def _featomic_hypers(self):
        return self.get_hypers()

    def get_hypers(self):
        """
        Return the native hyper parameters corresponding to this set of basis functions
        """
        raise NotImplementedError(
            f"This radial basis function ({self.__class__.__name__}) does not have "
            "matching hyper parameters in the native calculators. It should be used "
            "through one of the spliner class instead of directly."
        )

    @property
    def size(self) -> int:
        """Get the size of the basis set (i.e. the total number of basis functions)"""
        return self.max_radial + 1

    @property
    def integration_radius(self) -> float:
        """
        Get the radius to use for numerical evaluation of radial integrals.

        This default to the ``radius`` given as a class parameter, but can be overridden
        by child classes as needed.
        """
        return self.radius

    @abc.abstractmethod
    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative: bool
    ) -> np.ndarray:
        """
        Evaluate the primitive (not normalized not orthogonalized) radial basis for
        index ``n`` on grid points at the given ``positions``.

        :param n: index of the radial basis to evaluate
        :param positions: positions of the grid points where the basis should be
            evaluated
        :param derivative: should this function return the values of the radial basis or
            its derivatives.
        :return: the values (or derivative) of radial basis on the grid points
        """

    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        """
        Evaluate the orthogonalized and normalized radial basis on grid points at the
        given ``positions``. The returned array contains all the radial basis, from
        ``n=0`` to ``n = max_radial``.

        :param positions: positions of the grid points where the basis should be
            evaluated
        :param derivative: should this function return the values of the radial basis or
            its derivatives.
        :return: the values (or derivative) of radial basis on the grid points
        """
        if self._orthonormalization_matrix is None:
            self._orthonormalization_matrix = self._get_orthonormalization_matrix()

        basis = np.vstack(
            [
                self.compute_primitive(positions, n, derivative=derivative)
                for n in range(self.size)
            ]
        )
        return (self._orthonormalization_matrix @ basis).T

    def finite_differences_derivative(
        self,
        positions: np.ndarray,
        n: int,
        *,
        displacement=1e-6,
    ) -> np.ndarray:
        """
        Helper function to compute derivate of the radial function using finite
        differences. This can be used by child classes to implement the
        ``derivative=True`` branch of :py:meth:`compute` function.
        """
        value_pos = self.compute_primitive(
            positions + displacement / 2, n, derivative=False
        )
        value_neg = self.compute_primitive(
            positions - displacement / 2, n, derivative=False
        )

        return (value_pos - value_neg) / displacement

    def _get_orthonormalization_matrix(self) -> np.ndarray:
        """
        Compute the ``(self.size, self.size)`` orthonormalization matrix for this radial
        basis using numerical integration.
        """
        if not HAS_SCIPY:
            raise ValueError("Orthonormalization requires scipy!")

        gram_matrix = self._get_gram_matrix()

        # Get the normalization constants from the diagonal entries
        normalizations = np.zeros(self.size)

        for n in range(self.size):
            normalizations[n] = 1 / np.sqrt(gram_matrix[n, n])
            # Rescale gram matrix to be defined in terms of the normalized
            # (but not yet orthonormalized) basis functions
            gram_matrix[n, :] *= normalizations[n]
            gram_matrix[:, n] *= normalizations[n]

        eigvals, eigvecs = np.linalg.eigh(gram_matrix)
        if np.any(eigvals < 1e-12):
            raise ValueError(
                "Unable to orthonormalize the radial basis, gram matrix is singular. "
                "You can try decreasing the number of radial basis function, or "
                "changing some of the basis function parameters"
            )

        orthonormalization_matrix = (
            eigvecs @ np.diag(np.sqrt(1.0 / eigvals)) @ eigvecs.T
        )

        # Rescale the orthonormalization matrix so that it
        # works with respect to the primitive (i.e. not normalized)
        # radial basis functions
        for n in range(self.size):
            orthonormalization_matrix[:, n] *= normalizations[n]

        return orthonormalization_matrix

    def _get_gram_matrix(self) -> np.ndarray:
        """compute the Gram matrix of the current basis."""
        # Gram matrix (also called overlap matrix or inner product matrix)
        gram_matrix = np.zeros((self.size, self.size))

        def integrand(
            positions: np.ndarray,
            n1: int,
            n2: int,
        ) -> np.ndarray:
            r1 = self.compute_primitive(positions, n1, derivative=False)
            r2 = self.compute_primitive(positions, n2, derivative=False)
            return positions**2 * r1 * r2

        for n1 in range(self.size):
            for n2 in range(self.size):
                gram_matrix[n1, n2] = scipy.integrate.quad(
                    func=integrand,
                    a=0,
                    b=self.integration_radius,
                    args=(n1, n2),
                )[0]

        return gram_matrix


class Gto(RadialBasis):
    r"""
    Gaussian Type Orbital (GTO) radial basis.

    It is defined as

    .. math::

        R_n(r) = r^n e^{-\frac{r^2}{2\sigma_n^2}},

    where :math:`\sigma_n = \sqrt{n} r_0 / N` with :math:`r_0` being the basis
    ``radius`` and :math:`N` the number of radial basis functions.
    """

    def __init__(self, *, max_radial: int, radius: Optional[float] = None):
        """
        :parameter max_radial: maximal radial basis index to include (there will be
            ``N = max_radial + 1`` basis functions overall)
        :parameter radius: radius of the GTO basis functions. This is only required for
            LODE spherical expansion or splining the radial integral.
        """
        if radius is None:
            super().__init__(max_radial=max_radial, radius=float("inf"))
        else:
            super().__init__(max_radial=max_radial, radius=radius)

        self._gto_sigmas = np.ones(self.size, dtype=np.float64)

        if radius is not None:
            self._gto_radius = float(radius)
            for n in range(1, self.size):
                self._gto_sigmas[n] = np.sqrt(n)
            self._gto_sigmas *= self._gto_radius / self.size
        else:
            self._gto_radius = None

    def get_hypers(self):
        hypers = {
            "type": "Gto",
            "max_radial": self.max_radial,
        }

        if self._gto_radius is not None:
            hypers["radius"] = self._gto_radius

        return hypers

    @property
    def integration_radius(self) -> float:
        if self._gto_radius is None:
            raise ValueError(
                "`radius` needs to be specified for numerical evaluation "
                "of the GTO radial basis"
            )

        # We would ideally infinity as the ``integration_radius``, but this leads to
        # problems when calculating the radial integral with `quad`. So we pick
        # something large enough instead
        return 5 * self._gto_radius

    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative: bool
    ) -> np.ndarray:
        if self._gto_radius is None:
            raise ValueError(
                "`radius` needs to be specified for numerical evaluation "
                "of the GTO radial basis"
            )

        values = positions**n * np.exp(-0.5 * (positions / self._gto_sigmas[n]) ** 2)
        if derivative:
            return (
                n / positions * values - positions / self._gto_sigmas[n] ** 2 * values
            )

        return values


class Monomials(RadialBasis):
    r"""
    Monomial radial basis, consisting of functions:

    .. math::
        R_{nl}(r) = r^{l+2n}

    These capture precisely the radial dependence if we compute the Taylor expansion of
    a generic function defined in 3D space.

    :parameter angular_channel: index of the angular channel associated with this radial
        basis, i.e. :math:`l` in the equation above.
    """

    def __init__(self, *, angular_channel: int, max_radial: int, radius: float):
        super().__init__(max_radial=max_radial, radius=radius)
        self.angular_channel = int(angular_channel)
        assert self.angular_channel >= 0

    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative: bool
    ) -> np.ndarray:
        ell = self.angular_channel
        if derivative:
            return (ell + 2 * n) * positions ** (ell + 2 * n - 1)
        else:
            return positions ** (ell + 2 * n)


class SphericalBessel(RadialBasis):
    r"""Spherical Bessel functions as a radial basis.

    This is used among others in the `Laplacian eigenstate
    <https://doi.org/10.1063/5.0124363>`_ basis. The basis functions have the following
    form:

    .. math::

        R_{ln}(r) = j_l \left( \frac{r}{r_0} \text{zero}(J_l, n) \right)

    where :math:`j_l` is the spherical bessel function of the first kind of order
    :math:`l`, :math:`r_0` is the basis function ``radius``, and :math:`\text{zero}(J_l,
    n)` is the :math:`n`-th zero of (non-spherical) bessel function of first kind and
    order :math:`l`.

    :parameter angular_channel: index of the angular channel associated with this radial
        basis, i.e. :math:`l` in the equation above.
    """

    def __init__(self, *, angular_channel: int, max_radial: int, radius: float):
        super().__init__(max_radial=max_radial, radius=radius)
        self.angular_channel = int(angular_channel)
        assert self.angular_channel >= 0

        # this is computing all roots for all `l` up to `angular_channel` to then throw
        # away most of them. Maybe there is a better way to do this
        self._roots = SphericalBessel._compute_zeros(angular_channel + 1, self.size)

    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative: bool
    ) -> np.ndarray:
        ell = self.angular_channel
        values = scipy.special.spherical_jn(
            ell, positions * self._roots[ell, n] / self.radius, derivative=derivative
        )
        if derivative:
            values *= self._roots[ell, n] / self.radius

        return values

    @staticmethod
    def _compute_zeros(angular_size: int, radial_size: int) -> np.ndarray:
        """Zeros of spherical bessel functions.

        Code is taken from the `Scipy Cookbook <spc>`_

        .. _spc: https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html

        :parameter angular_size: number of angular components
        :parameter radial_size: number of radial components
        :returns: computed zeros of the spherical bessel functions
        """

        def Jn(r: float, ell: int) -> float:
            return np.sqrt(np.pi / (2 * r)) * scipy.special.jv(ell + 0.5, r)

        def Jn_zeros(angular_size: int, radial_size: int) -> np.ndarray:
            zeros_j = np.zeros((angular_size, radial_size), dtype=np.float64)
            zeros_j[0] = np.arange(1, radial_size + 1) * np.pi
            points = np.arange(1, radial_size + angular_size + 1) * np.pi
            roots = np.zeros(radial_size + angular_size, dtype=np.float64)
            for ell in range(1, angular_size):
                for j in range(radial_size + angular_size - ell):
                    roots[j] = scipy.optimize.brentq(
                        Jn, points[j], points[j + 1], (ell,)
                    )
                points = roots
                zeros_j[ell][:radial_size] = roots[:radial_size]
            return zeros_j

        return Jn_zeros(angular_size, radial_size)


########################################################################################
########################################################################################


class ExpansionBasis(metaclass=abc.ABCMeta):
    """
    Base class representing a set of basis functions used by spherical expansions.

    A full basis typically uses both a set of radial basis functions, and angular basis
    functions; combined in various ways. The angular basis functions are almost always
    spherical harmonics, while the radial basis function can be freely picked.

    You can inherit from this class to define new sets of basis functions, implementing
    :py:meth:`get_hypers` to create the right hyper parameters for the underlying native
    calculator, as well as :py:meth:`angular_channels` and :py:meth:`radial_basis` to
    define the set of basis functions to use.
    """

    def _featomic_hypers(self):
        return self.get_hypers()

    def get_hypers(self):
        """
        Return the native hyper parameters corresponding to this set of basis functions
        """
        raise NotImplementedError(
            f"This basis functions set ({self.__class__.__name__}) does not have "
            "matching hyper parameters in the native calculators. It should be used "
            "through one of the spliner class instead of directly."
        )

    @abc.abstractmethod
    def angular_channels(self) -> List[int]:
        """Get the list of angular channels that are included in the expansion basis."""

    @abc.abstractmethod
    def radial_basis(self, angular: int) -> RadialBasis:
        """Get the radial basis used for a given ``angular`` channel"""


class TensorProduct(ExpansionBasis):
    r"""
    Basis function set combining spherical harmonics with a radial basis functions set,
    taking all possible combinations of radial and angular basis function.

    Using ``N`` radial basis functions and ``L`` angular basis functions, this will
    create ``N x L`` basis functions (:math:`B_{nlm}`) for the overall expansion:

    .. math::

        B_{nlm}(\boldsymbol{r}) = R_{nl}(r)Y_{lm}(\hat{r}) \,
    """

    def __init__(
        self,
        *,
        max_angular: int,
        radial: RadialBasis,
        spline_accuracy: Optional[float] = 1e-8,
    ):
        """
        :param max_angular: Largest angular channel to include in the basis
        :param radial: radial basis to use for all angular channels
        :param spline_accuracy: requested accuracy of the splined radial integrals,
            defaults to 1e-8
        """
        self.max_angular = int(max_angular)
        self.radial = radial

        if spline_accuracy is None:
            self.spline_accuracy = None
        else:
            self.spline_accuracy = float(spline_accuracy)
            assert self.spline_accuracy > 0

        assert self.max_angular >= 0
        assert isinstance(self.radial, RadialBasis)

    def get_hypers(self):
        return {
            "type": "TensorProduct",
            "max_angular": self.max_angular,
            "radial": self.radial,
            "spline_accuracy": self.spline_accuracy,
        }

    def angular_channels(self) -> List[int]:
        return list(range(self.max_angular + 1))

    def radial_basis(self, angular: int) -> RadialBasis:
        return self.radial


class Explicit(ExpansionBasis):
    """
    An expansion basis where combinations of radial and angular functions is picked
    explicitly.

    The angular basis functions are still spherical harmonics, but only the degrees
    included as keys in ``by_angular`` will be part of the output. Each of these angular
    basis function can then be associated with a set of different radial basis function,
    potentially of different sizes.
    """

    def __init__(
        self,
        *,
        by_angular: Dict[int, RadialBasis],
        spline_accuracy: Optional[float] = 1e-8,
    ):
        """
        :param by_angular: definition of the radial basis for each angular channel to
            include.
        :param spline_accuracy: requested accuracy of the splined radial integrals,
            defaults to 1e-8
        """
        self.by_angular = by_angular

        if spline_accuracy is None:
            self.spline_accuracy = None
        else:
            self.spline_accuracy = float(spline_accuracy)
            assert self.spline_accuracy > 0

        for angular, radial in self.by_angular.items():
            assert angular >= 0
            assert isinstance(radial, RadialBasis)

    def get_hypers(self):
        return {
            "type": "Explicit",
            "by_angular": self.by_angular,
            "spline_accuracy": self.spline_accuracy,
        }

    def angular_channels(self) -> List[int]:
        return list(self.by_angular.keys())

    def radial_basis(self, angular: int) -> RadialBasis:
        return self.by_angular[angular]


class LaplacianEigenstate(ExpansionBasis):
    """
    The Laplacian eigenstate basis, introduced in https://doi.org/10.1063/5.0124363, is
    a set of basis functions for spherical expansion which is both *smooth* (in the same
    sense as the smoothness of a low-pass-truncated Fourier expansion) and *ragged*,
    using a different number of radial function for each angular channel. This is
    intended to obtain a more balanced smoothness level in the radial and angular
    direction for a given total number of basis functions.

    This expansion basis is not directly implemented in the native calculators, but is
    intended to be used with the :py:class:`featomic.splines.SoapSpliner` to create
    splines of the radial integrals.
    """

    def __init__(
        self,
        *,
        radius: float,
        max_radial: int,
        max_angular: Optional[int] = None,
        spline_accuracy: Optional[float] = 1e-8,
    ):
        """
        :param radius: radius of the basis functions
        :param max_radial: number of radial basis function for the ``L=0`` angular
            channel. All other angular channels will have fewer radial basis functions.
        :param max_angular: Truncate the set of radial functions at this angular
            channel. If ``None``, this will be set to a high enough value to include all
            basis functions with an Laplacian eigenvalue below the one for ``l=0,
            n=max_radial``.
        :param spline_accuracy: requested accuracy of the splined radial integrals,
            defaults to 1e-8
        """
        self.radius = float(radius)
        assert self.radius >= 0

        self.max_radial = int(max_radial)
        assert self.max_radial >= 0

        if max_angular is None:
            self.max_angular = self.max_radial
        else:
            self.max_angular = int(max_angular)
            assert self.max_angular >= 0

        if spline_accuracy is None:
            self.spline_accuracy = None
        else:
            self.spline_accuracy = float(spline_accuracy)
            assert self.spline_accuracy > 0

        # compute the zeros of the spherical Bessel functions
        zeros_ln = SphericalBessel._compute_zeros(
            self.max_angular + 1, self.max_radial + 1
        )

        # determine the eigenvalue cutoff
        eigenvalues = zeros_ln**2 / self.radius**2
        max_eigenvalue = eigenvalues[0, max_radial]

        # find the actual `max_angular` if the user did not specify one by repeatedly
        # increasing the size of `eigenvalues`, until we find an angular channel where
        # all eigenvalues are above the cutoff.
        if max_angular is None:
            while eigenvalues[-1, 0] < max_eigenvalue:
                self.max_angular += self.max_radial
                zeros_ln = SphericalBessel._compute_zeros(
                    self.max_angular + 1, self.max_radial + 1
                )
                eigenvalues = zeros_ln**2 / self.radius**2

            self.max_angular = len(np.where(eigenvalues[:, 0] <= max_eigenvalue)[0]) - 1
            assert self.max_angular >= 0

        by_angular = {}
        for angular in range(self.max_angular + 1):
            max_radial = len(np.where(eigenvalues[angular, :] <= max_eigenvalue)[0]) - 1
            by_angular[angular] = SphericalBessel(
                angular_channel=angular,
                max_radial=max_radial,
                radius=self.radius,
            )

        self._explicit = Explicit(
            by_angular=by_angular,
            spline_accuracy=self.spline_accuracy,
        )

    def get_hypers(self):
        return self._explicit.get_hypers()

    def angular_channels(self) -> List[int]:
        return self._explicit.angular_channels()

    def radial_basis(self, angular: int) -> RadialBasis:
        return self._explicit.radial_basis(angular)
