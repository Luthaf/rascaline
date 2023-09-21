"""
Splined radial integrals
========================

Classes for generating splines which can be used as tabulated radial integrals in the
various SOAP and LODE calculators. For an complete example of how to use these classes
see :ref:`example-splines`.

.. autoclass:: rascaline.utils.RadialIntegralSplinerBase
    :members:
    :show-inheritance:

.. autoclass:: rascaline.utils.RadialIntegralFromFunction
    :members:
    :show-inheritance:
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import numpy as np


class RadialIntegralSplinerBase(ABC):
    """Base class for splining arbitrary radial integrals.

    If ``_radial_integral_derivative`` is not implemented in a child class it will
    computed based on finite differences.

    :parameter max_angular: number of radial components
    :parameter max_radial: number of angular components
    :parameter spline_cutoff: cutoff radius for the spline interpolation. This is also
        the maximal value that can be interpolated.
    :parameter accuracy: accuracy of the numerical integration and the splining.
        Accuracy is reached when either the mean absolute error or the mean relative
        error gets below the ``accuracy`` threshold.
    """

    def __init__(
        self,
        max_radial: int,
        max_angular: int,
        spline_cutoff: float,
        accuracy: float,
    ):
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.spline_cutoff = spline_cutoff
        self.accuracy = accuracy

    @abstractmethod
    def _radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
        """Method calculating the radial integral."""
        ...

    @property
    def _center_contribution(self) -> Union[None, np.ndarray]:
        r"""Contribution of the central atom required for LODE calculations."""

        return None

    def _radial_integral_derivative(
        self, n: int, ell: int, positions: np.ndarray
    ) -> np.ndarray:
        """Method calculating the derivatice of the radial integral."""
        displacement = 1e-6
        mean_abs_positions = np.abs(positions).mean()

        if mean_abs_positions <= 1.0:
            raise ValueError(
                "Numerically derivative of the radial integral can not be performed "
                "since positions are too small. Mean of the absolute positions is "
                f"{mean_abs_positions:.1e} but should be at least 1."
            )

        radial_integral_pos = self._radial_integral(
            n, ell, positions + displacement / 2
        )
        radial_integral_neg = self._radial_integral(
            n, ell, positions - displacement / 2
        )

        return (radial_integral_pos - radial_integral_neg) / displacement

    def _value_evaluator_3D(
        self,
        positions: np.ndarray,
        derivative: bool,
    ):
        values = np.zeros([len(positions), self.max_angular + 1, self.max_radial])
        for ell in range(self.max_angular + 1):
            for n in range(self.max_radial):
                if derivative:
                    values[:, ell, n] = self._radial_integral_derivative(
                        n, ell, positions
                    )
                else:
                    values[:, ell, n] = self._radial_integral(n, ell, positions)

        return values

    def compute(
        self,
        n_spline_points: Optional[int] = None,
    ) -> Dict:
        """Compute the spline for rascaline's tabulated radial integrals.

        :parameter n_spline_points: Use fixed number of spline points instead of find
            the number based on the provided ``accuracy``.
        :returns dict: dictionary for the input as the ``radial_basis`` parameter  of a
            rascaline calculator.
        """

        def value_evaluator_3D(positions):
            return self._value_evaluator_3D(positions, derivative=False)

        def derivative_evaluator_3D(positions):
            return self._value_evaluator_3D(positions, derivative=True)

        if n_spline_points is not None:
            positions = np.linspace(0, self.spline_cutoff, n_spline_points)
            values = value_evaluator_3D(positions)
            derivatives = derivative_evaluator_3D(positions)
        else:
            dynamic_spliner = DynamicSpliner(
                0,
                self.spline_cutoff,
                value_evaluator_3D,
                derivative_evaluator_3D,
                self.accuracy,
            )
            positions, values, derivatives = dynamic_spliner.spline()

        # Convert positions, values, derivatives into the appropriate json formats:
        spline_points = []
        for position, value, derivative in zip(positions, values, derivatives):
            spline_points.append(
                {
                    "position": position,
                    "values": {
                        "v": 1,
                        "dim": value.shape,
                        "data": value.flatten().tolist(),
                    },
                    "derivatives": {
                        "v": 1,
                        "dim": derivative.shape,
                        "data": derivative.flatten().tolist(),
                    },
                }
            )

        parameters = {"points": spline_points}

        center_contribution = self._center_contribution
        if center_contribution is not None:
            parameters["center_contribution"] = center_contribution

        return {"TabulatedRadialIntegral": parameters}


class DynamicSpliner:
    def __init__(
        self,
        start: float,
        stop: float,
        values_fn: Callable[[np.ndarray], np.ndarray],
        derivatives_fn: Callable[[np.ndarray], np.ndarray],
        accuracy: float = 1e-8,
    ) -> None:
        """Dynamic spline generator.

        This class can be used to spline any set of functions defined within the
        start-stop interval. Cubic Hermite splines
        (https://en.wikipedia.org/wiki/Cubic_Hermite_spline) are used. The same spline
        points will be used for all functions, and more will be added until either the
        relative error or the absolute error fall below the requested accuracy on
        average across all functions. The functions are specified via values_fn and
        derivatives_fn. These must be able to take a numpy 1D array of positions as
        their input, and they must output a numpy array where the first dimension
        corresponds to the input positions, while other dimensions are arbitrary and can
        correspond to any way in which the target functions can be classified. The
        splines can be obtained via the spline method.
        """

        self.start = start
        self.stop = stop
        self.values_fn = values_fn
        self.derivatives_fn = derivatives_fn
        self.requested_accuracy = accuracy

        # initialize spline with 11 points
        positions = np.linspace(start, stop, 11)
        self.spline_positions = positions
        self.spline_values = values_fn(positions)
        self.spline_derivatives = derivatives_fn(positions)

        self.number_of_custom_axes = len(self.spline_values.shape) - 1

    def spline(self):
        """Calculates and outputs the splines.

        The outputs of this function are, respectively: - A numpy 1D array containing
        the spline positions. These are equally spaced in the start-stop interval.
        - A numpy ndarray containing the values of the splined functions at the spline
          positions. The first dimension corresponds to the spline positions, while all
          subsequent dimensions are consistent with the values_fn and
          `get_function_derivative` provided during initialization of the class.
        - A numpy ndarray containing the derivatives of the splined functions at the
          spline positions, with the same structure as that of the ndarray of values.
        """

        while True:
            n_intermediate_positions = len(self.spline_positions) - 1

            if n_intermediate_positions >= 50000:
                raise ValueError(
                    "Maximum number of spline points reached. \
                    There might be a problem with the functions to be splined"
                )

            half_step = (self.spline_positions[1] - self.spline_positions[0]) / 2
            intermediate_positions = np.linspace(
                self.start + half_step, self.stop - half_step, n_intermediate_positions
            )

            estimated_values = self._compute_from_spline(intermediate_positions)
            new_values = self.values_fn(intermediate_positions)

            mean_absolute_error = np.mean(np.abs(estimated_values - new_values))
            with np.errstate(divide="ignore"):  # Ignore divide-by-zero warnings
                mean_relative_error = np.mean(
                    np.abs((estimated_values - new_values) / new_values)
                )

            if (
                mean_absolute_error < self.requested_accuracy
                or mean_relative_error < self.requested_accuracy
            ):
                break

            new_derivatives = self.derivatives_fn(intermediate_positions)

            concatenated_positions = np.concatenate(
                [self.spline_positions, intermediate_positions], axis=0
            )
            concatenated_values = np.concatenate(
                [self.spline_values, new_values], axis=0
            )
            concatenated_derivatives = np.concatenate(
                [self.spline_derivatives, new_derivatives], axis=0
            )

            sort_indices = np.argsort(concatenated_positions, axis=0)

            self.spline_positions = concatenated_positions[sort_indices]
            self.spline_values = concatenated_values[sort_indices]
            self.spline_derivatives = concatenated_derivatives[sort_indices]

        return self.spline_positions, self.spline_values, self.spline_derivatives

    def _compute_from_spline(self, positions):
        x = positions
        delta_x = self.spline_positions[1] - self.spline_positions[0]
        n = (np.floor(x / delta_x)).astype(np.int32)

        t = (x - n * delta_x) / delta_x
        t_2 = t**2
        t_3 = t**3

        h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
        h10 = t_3 - 2.0 * t_2 + t
        h01 = -2.0 * t_3 + 3.0 * t_2
        h11 = t_3 - t_2

        p_k = self.spline_values[n]
        p_k_1 = self.spline_values[n + 1]

        m_k = self.spline_derivatives[n]
        m_k_1 = self.spline_derivatives[n + 1]

        new_shape = (-1,) + (1,) * self.number_of_custom_axes
        h00 = h00.reshape(new_shape)
        h10 = h10.reshape(new_shape)
        h01 = h01.reshape(new_shape)
        h11 = h11.reshape(new_shape)

        interpolated_values = (
            h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
        )

        return interpolated_values


class RadialIntegralFromFunction(RadialIntegralSplinerBase):
    r"""Compute the radial integral spline points based on a provided function.

    :parameter radial_integral: Function to compute the radial integral. Function must
        take ``n``, ``l``, and ``positions`` as inputs, where ``n`` and ``l`` are
        integers and ``positions`` is a numpy 1-D array that contains the spline points
        at which the radial integral will be evaluated. The function must return a numpy
        1-D array containing the values of the radial integral.
    :parameter spline_cutoff: cutoff radius for the spline interpolation. This is also
        the maximal value that can be interpolated.
    :parameter max_radial: number of angular componentss
    :parameter max_angular: number of radial components
    :parameter radial_integral_derivative: The derivative of the radial integral taking
        the same paramaters as ``radial_integral``. If it is :py:obj:`None` (default),
        finite differences are used to calculate the derivative of the radial integral.
        It is recommended to provide this parameter if possible. Derivatives from finite
        differences can cause problems when evaluating at the edges of the domain (i.e.,
        at ``0`` and ``spline_cutoff``) because the function might not be defined
        outside of the domain.
    :parameter accuracy: accuracy of the numerical integration and the splining.
        Accuracy is reached when either the mean absolute error or the mean relative
        error gets below the ``accuracy`` threshold.
    :parameter center_contribution: Contribution of the central atom required for LODE
        calculations. The ``center_contribution`` is defined as

        .. math::
           c_n = \sqrt{4π}\int_0^\infty dr r^2 R_n(r) g(r)

        where :math:`g(r)` is the radially symmetric density function, `R_n(r)` the
        radial basis function and :math:`n` the current radial channel. This should be
        pre-computed and provided as a separate parameter.

    Example
    -------
    First define a ``radial_integral`` function

    >>> def radial_integral(n, ell, r):
    ...     return np.sin(r)
    ...

    and provide this as input to the spline generator

    >>> spliner = RadialIntegralFromFunction(
    ...     radial_integral=radial_integral,
    ...     max_radial=12,
    ...     max_angular=9,
    ...     spline_cutoff=8.0,
    ... )

    Finally, we can use the ``spliner`` directly in the ``radial_integral`` section of a
    calculator

    >>> from rascaline import SoapPowerSpectrum
    >>> calculator = SoapPowerSpectrum(
    ...     cutoff=8.0,
    ...     max_radial=12,
    ...     max_angular=9,
    ...     center_atom_weight=1.0,
    ...     radial_basis=spliner.compute(),
    ...     atomic_gaussian_width=1.0,  # ignored
    ...     cutoff_function={"Step": {}},
    ... )

    The ``atomic_gaussian_width`` paramater is required by the calculator but will be
    will be ignored during the feature computation.

    A more in depth example using a "rectangular" Laplacian eigenstate basis
    is provided in the :ref:`example section<example-splines>`.
    """

    def __init__(
        self,
        radial_integral: Callable[[int, int, np.ndarray], np.ndarray],
        spline_cutoff: float,
        max_radial: int,
        max_angular: int,
        radial_integral_derivative: Optional[
            Callable[[int, int, np.ndarray], np.ndarray]
        ] = None,
        center_contribution: Optional[np.ndarray] = None,
        accuracy: float = 1e-8,
    ):
        self.radial_integral_function = radial_integral
        self.radial_integral_derivative_funcion = radial_integral_derivative
        self.center_contribution = center_contribution

        super().__init__(
            max_radial=max_radial,
            max_angular=max_angular,
            spline_cutoff=spline_cutoff,
            accuracy=accuracy,
        )

    def _radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
        return self.radial_integral_function(n, ell, positions)

    @property
    def _center_contribution(self) -> Union[None, np.ndarray]:
        # Test that ``len(self.center_contribution) == max_radial`` is performed by the
        # calculator.
        return self.center_contribution

    def _radial_integral_derivative(
        self, n: int, ell: int, positions: np.ndarray
    ) -> np.ndarray:
        if self.radial_integral_derivative_funcion is None:
            return super()._radial_integral_derivative(n, ell, positions)
        else:
            return self.radial_integral_derivative_funcion(n, ell, positions)
