import abc
import warnings
from typing import Optional

import numpy as np


try:
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RadialScaling(metaclass=abc.ABCMeta):
    """
    Base class representing radial scaling of atomic densities.

    You can inherit from this class to define new custom radial scaling, implementing
    :py:meth:`compute` accordingly. If the new radial scaling has corresponding hyper
    parameters in the native calculators, you should also implement
    :py:meth:`get_hypers`.
    """

    def _featomic_hypers(self):
        return self.get_hypers()

    def get_hypers(self):
        """
        Return the native hyper parameters corresponding to this atomic density scaling
        """
        raise NotImplementedError(
            f"this density scaling ({self.__class__.__name__}) does not have matching "
            "hyper parameters in the native calculators"
        )

    @abc.abstractmethod
    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        """
        Compute the scaling function (or it's derivative) on grid points at the given
        ``positions``.

        :param positions: positions of grid point where to evaluate the radial scaling
        :param derivative: should this function return the values or the derivatives of
            the radial scaling?
        :returns: evaluated radial scaling function
        """


class Willatt2018(RadialScaling):
    r"""
    Radial density scaling as proposed in https://doi.org/10.1039/C8CP05921G by Willatt
    et. al.

    .. math::

        \text{scaling}(r) = \frac{c}{c + \left(\frac{r}{r_0}\right) ^ m}
    """

    def __init__(self, *, exponent: int, rate: float, scale: float):
        """
        :param exponent: :math:`m` in the formula above
        :param rate: :math:`c` in the formula above
        :param scale: :math:`r_0` in the formula above
        """
        self.exponent = int(exponent)
        self.rate = float(rate)
        self.scale = float(scale)

        assert self.exponent >= 0
        assert self.scale > 0
        assert self.rate >= 0

    def get_hypers(self):
        return {
            "type": "Willatt2018",
            "exponent": self.exponent,
            "rate": self.rate,
            "scale": self.scale,
        }

    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        if self.rate == 0:
            result = (self.scale / positions) ** self.exponent
            if derivative:
                result *= -self.exponent / positions
            return result
        elif self.exponent == 0:
            if derivative:
                return np.zeros_like(positions)
            else:
                return np.ones_like(positions)
        else:
            if derivative:
                self.rate / (self.rate + (positions / self.scale) ** self.exponent)
            else:
                rs = positions / self.scale
                denominator = (self.rate + rs**self.exponent) ** 2

                factor = -self.rate * self.exponent / self.scale

                return factor * rs ** (self.exponent - 1) / denominator


class AtomicDensity(metaclass=abc.ABCMeta):
    r"""
    Base class representing atomic densities.

    You can inherit from this class to define new custom densities, implementing
    :py:meth:`compute` accordingly. If the new density has corresponding hyper
    parameters in the native calculators, you should also implement
    :py:meth:`get_hypers`.

    All atomic densities are assumed to be invariant under rotation, and as such only
    defined as a function of the distance to the origin.

    The overall density around a central atom is the sum of the central atom's neighbors
    density, with some optional radial scaling.

    .. math::

        \rho_i(r) = \sum_j \text{scaling}(r_{ij}) \; g_j(r - r_{ij})

    where :math:`\text{scaling}(r)` is the scaling function, :math:`g_j` the density
    coming from neighbor :math:`j`, and :math:`r_{ij}` the distance between the center
    :math:`i` and neighbor :math:`j`.
    """

    def __init__(
        self,
        *,
        center_atom_weight: float = 1.0,
        scaling: Optional[RadialScaling] = None,
    ):
        r"""
        :param center_atom_weight: in density expansion, the central atom sees its own
            density, and is in this sense its own neighbor. Setting this weight to ``0``
            allows to disable this behavior and only expand the density of actual
            neighbors.
        :param scaling: optional radial scaling function. If this is left to ``None``,
            no radial scaling is applied.
        """
        self.center_atom_weight = float(center_atom_weight)
        self.scaling = scaling

        assert isinstance(self.scaling, (type(None), RadialScaling))

    @abc.abstractmethod
    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        """
        Compute the density (or it's derivative) around a single atom.

        The atom is located at ``position=0`` and the density is computed on multiple
        grid points at the given ``positions``. The computed density does not include
        any radial scaling or central atom weighting.

        :param positions: positions of grid point where to evaluate the atomic density
        :param derivative: should this function return the values or the derivatives of
            the density?
        :returns: evaluated atomic density on the grid
        """

    def get_hypers(self):
        """
        Return the native hyper parameters corresponding to this atomic density
        """
        raise NotImplementedError(
            f"This density ({self.__class__.__name__}) does not have matching "
            "hyper parameters in the native calculators. It should be used "
            "through one of the spliner class instead of directly."
        )

    def _featomic_hypers(self):
        """
        Return the native hyper parameters corresponding to this atomic density.
        """
        return {
            **self.get_hypers(),
            "center_atom_weight": self.center_atom_weight,
            "scaling": self.scaling,
        }


class DiracDelta(AtomicDensity):
    r"""Delta atomic densities of the form :math:`g(r)=\delta(r)`."""

    def __init__(
        self,
        *,
        center_atom_weight: float = 1.0,
        scaling: Optional[RadialScaling] = None,
    ):
        super().__init__(center_atom_weight=center_atom_weight, scaling=scaling)

    def get_hypers(self):
        return {"type": "DiracDelta"}

    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        if derivative:
            return np.zeros_like(positions)
        else:
            result = np.zeros_like(positions)
            result[result == 0.0] = 1.0
            return result


class Gaussian(AtomicDensity):
    r"""Gaussian atomic density function.

    In featomic, we use the convention

    .. math::

        g(r) = \frac{1}{(\pi \sigma^2)^{3/4}}e^{-\frac{r^2}{2\sigma^2}} \,.

    The prefactor was chosen such that the "L2-norm" of the Gaussian

    .. math::

            \|g\|^2 = \int \mathrm{d}^3\boldsymbol{r} |g(r)|^2 = 1\,,

    The derivatives of the Gaussian atomic density with respect to the position is

    .. math::

        g^\prime(r) =
            \frac{\partial g(r)}{\partial r} = \frac{-r}{\sigma^2(\pi
            \sigma^2)^{3/4}}e^{-\frac{r^2}{2\sigma^2}} \,.

    :param width: Width of the atom-centered gaussian used to create the atomic density
    """

    def __init__(
        self,
        *,
        width: float,
        center_atom_weight: float = 1.0,
        scaling: Optional[RadialScaling] = None,
    ):
        super().__init__(center_atom_weight=center_atom_weight, scaling=scaling)
        self.width = float(width)

    def get_hypers(self):
        return {"type": "Gaussian", "width": self.width}

    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        width_sq = self.width**2
        x = positions**2 / (2 * width_sq)

        density = np.exp(-x) / (np.pi * width_sq) ** (3 / 4)

        if derivative:
            density *= -positions / width_sq

        return density


class SmearedPowerLaw(AtomicDensity):
    r"""Smeared power law density, as used in LODE.

    This is a smooth, differentiable density that behaves like :math:`1 / r^p` as
    :math:`r` goes to infinity.

    It is defined as

    .. math::

        g(r) = \frac{1}{\Gamma\left(\frac{p}{2}\right)}
               \frac{\gamma\left( \frac{p}{2}, \frac{r^2}{2\sigma^2} \right)}
                    {r^p},

    where :math:`p` is the potential exponent, :math:`\Gamma(z)` is the Gamma function
    and :math:`\gamma(a, x)` is the incomplete lower Gamma function.

    For more information about the derivation of this density, see
    https://doi.org/10.1021/acs.jpclett.3c02375 and section D of the supplementary
    information.

    :param smearing: Smearing used to remove the singularity at 0 (:math:`\sigma` above)
    :param exponent: Potential exponent of the decorated atom density (:math:`p` above)
    """

    def __init__(
        self,
        *,
        smearing: float,
        exponent: int,
        center_atom_weight: float = 1.0,
        scaling: Optional[RadialScaling] = None,
    ):
        super().__init__(center_atom_weight=center_atom_weight, scaling=scaling)
        self.smearing = float(smearing)
        self.exponent = int(exponent)

    def get_hypers(self):
        return {
            "type": "SmearedPowerLaw",
            "smearing": self.smearing,
            "exponent": self.exponent,
        }

    def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
        if not HAS_SCIPY:
            raise ValueError("SmearedPowerLaw requires scipy to be installed")

        if self.exponent == 0:
            proxy = Gaussian(width=self.smearing)
            return proxy.compute(positions=positions, derivative=derivative)
        else:
            smearing_sq = self.smearing**2
            a = self.exponent / 2
            x = positions**2 / (2 * smearing_sq)

            # Evaluating the formula above at :math:`r=0` is problematic because
            # :math:`g(r)` is of the form :math:`0/0`. For practical implementations, it
            # is thus more convenient to rewrite the density as
            #
            # .. math::
            #
            #   g(r) = \frac{1}{\Gamma(a)}\frac{1}{\left(2 \sigma^2\right)^a}
            #     \begin{cases}
            #         \frac{1}{a} - \frac{x}{a+1} + \frac{x^2}{2(a+2)}+\mathcal{O}(x^3)
            #             & x < 10^{-5} \\
            #         \frac{\gamma(a,x)}{x^a}
            #             & x \geq 10^{-5}
            #     \end{cases}
            #
            # where :math:`a = p/2`. It is convenient to use the expression for
            # sufficiently small :math:`x` since the relative weight of the first
            # neglected term is on the order of :math:`1/6x^3`. Therefore, the threshold
            # :math:`x = 10^{-5}` leads to relative errors on the order of the machine
            # epsilon.
            with warnings.catch_warnings():
                # Even though we use `np.where` to apply the `_compute_close_zero`
                # method for small `x`, the `_compute_far_zero` method will also
                # evaluated for small `x` and sending RuntimeWarnings. We filter these
                # warnings to avoid that these are presented to the user.
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                density = np.where(
                    x < 1e-5,
                    self._compute_close_zero(a, x, derivative=derivative),
                    self._compute_far_zero(a, x, derivative=derivative),
                )

            density *= 1 / scipy.special.gamma(a) / (2 * smearing_sq) ** a

            # add inner derivative: ∂x/∂r
            if derivative:
                density *= positions / smearing_sq

            return density

    def _compute_close_zero(self, a: float, x: np.ndarray, derivative: bool):
        if derivative:
            return -1 / (a + 1) + x / (a + 2)
        else:
            return 1 / a - x / (a + 1) + x**2 / (2 * (a + 2))

    def _compute_far_zero(self, a: float, x: np.ndarray, derivative: bool):
        if derivative:
            return (
                np.exp(-x)
                - a * scipy.special.gamma(a) * scipy.special.gammainc(a, x) / x**a
            ) / x
        else:
            return scipy.special.gamma(a) * scipy.special.gammainc(a, x) / x**a
