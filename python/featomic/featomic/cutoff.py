import abc
from typing import Optional

import numpy as np


class SmoothingFunction(metaclass=abc.ABCMeta):
    """
    Base class representing radial cutoff smoothing functions.

    You can inherit from this class to define new smoothing functions, implementing
    :py:meth:`compute` accordingly. If the new smoothing function has corresponding
    hyper parameters in the native calculators, you should also implement
    :py:meth:`get_hypers`.
    """

    def _featomic_hypers(self):
        return self.get_hypers()

    def get_hypers(self):
        raise NotImplementedError(
            f"this smoothing function ({self.__class__.__name__}) does not have "
            "matching hyper parameters in the native calculators"
        )

    @abc.abstractmethod
    def compute(
        self, cutoff: float, positions: np.ndarray, *, derivative: bool
    ) -> np.ndarray:
        """
        Compute the smoothing function on grid points at the given ``positions``.

        :param cutoff: spherical cutoff radius
        :param positions: positions of the grid points where the smoothing function
            should be evaluated
        :param derivative: should this function return the values of the smoothing
            function or it's derivatives
        """


class ShiftedCosine(SmoothingFunction):
    r"""
    Shifted cosine smoothing function, with the following form:

    .. math::

        f(r) = \begin{cases}
            1 & \text{for } r \le r_c - \sigma \\
            1/2 \left(1 + \frac{\cos(\pi (r - r_c + \sigma)}{\sigma} \right)
            & \text{for } r_c - \sigma \le r \le r_c \\
            0 & \text{for } r \gt r_c \\
        \end{cases}

    with :math:`r_c` the cutoff radius and :math:`\sigma` is the width of the smoothing
    (roughly how far from the cutoff smoothing should happen).

    :param width: width of the smoothing (:math:`\sigma` in the equation above)
    """

    def __init__(self, *, width: float):
        self.width = float(width)
        assert self.width >= 0

    def get_hypers(self):
        return {"type": "ShiftedCosine", "width": self.width}

    def compute(
        self, cutoff: float, positions: np.ndarray, *, derivative: bool
    ) -> np.ndarray:
        assert cutoff > self.width
        result = np.zeros_like(positions)

        mask = np.logical_and(positions >= cutoff - self.width, positions < cutoff)
        s = np.pi * (positions[mask] - cutoff + self.width) / self.width

        if derivative:
            result[mask] = -0.5 * np.pi * np.sin(s) / self.width
        else:
            result[positions < cutoff - self.width] = 1.0
            result[mask] = 0.5 * (1 + np.cos(s))

        return result


class Step(SmoothingFunction):
    """
    Step smoothing function, i.e. no smoothing.

    This function is equal to 1 inside the cutoff radius and to 0 outside of the cutoff
    radius, with a discontinuity at the cutoff.
    """

    def get_hypers(self):
        return {"type": "Step"}

    def compute(
        self, cutoff: float, positions: np.ndarray, *, derivative: bool
    ) -> np.ndarray:
        if derivative:
            return np.zeros_like(positions)
        else:
            return np.where(positions <= cutoff, 1.0, 0.0)


class Cutoff:
    """
    The ``Cutoff`` class contains the definition of local environments, where an atom
    environment is defined by all its neighbors inside a sphere centered on the atom
    with the given spherical cutoff ``radius``.

    During an atomistic simulation, atoms entering and exiting the sphere will create
    discontinuities. To prevent them, one can use a ``smoothing`` function, smoothing
    introducing new atoms inside the neighborhood.
    """

    def __init__(self, radius: float, smoothing: Optional[SmoothingFunction]):
        self.radius = float(radius)
        assert self.radius >= 0.0

        if smoothing is None:
            self.smoothing = Step()
        else:
            self.smoothing = smoothing

        assert isinstance(self.smoothing, SmoothingFunction)

    def _featomic_hypers(self):
        return {
            "radius": self.radius,
            "smoothing": self.smoothing,
        }
