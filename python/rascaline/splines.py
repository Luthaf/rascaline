# -*- coding: utf-8 -*-
import numpy as np


def generate_splines(
    radial_basis,
    radial_basis_derivatives,
    max_radial,
    max_angular,
    cutoff_radius,
    n_spline_points=None,
    requested_accuracy=1e-8,
):
    """Spline generator for tabulated radial integrals.

    Besides some self-explanatory parameters, this function takes as inputs two
    functions, namely radial_basis and radial_basis_derivatives. These must be
    able to calculate the radial basis functions by taking n, l, and r as their
    inputs, where n and l are integers and r is a numpy 1-D array that contains
    the spline points at which the radial basis function (or its derivative)
    needs to be evaluated. These functions should return a numpy 1-D array
    containing the values of the radial basis function (or its derivative)
    corresponding to the specified n and l, and evaluated at all points in the
    r array. If specified, n_spline_points determines how many spline points
    will be used for each splined radial basis function. Alternatively, the user
    can specify a requested accuracy. Spline points will be added until either
    the relative error or the absolute error fall below the requested accuracy on
    average across all radial basis functions.
    """

    def value_evaluator_3D(positions):
        values = []
        for el in range(max_angular + 1):
            for n in range(max_radial):
                value = radial_basis(n, el, positions)
                values.append(value)
        values = np.array(values).T
        values = values.reshape(len(positions), max_angular + 1, max_radial)
        return values

    def derivative_evaluator_3D(positions):
        derivatives = []
        for el in range(max_angular + 1):
            for n in range(max_radial):
                derivative = radial_basis_derivatives(n, el, positions)
                derivatives.append(derivative)
        derivatives = np.array(derivatives).T
        derivatives = derivatives.reshape(len(positions), max_angular + 1, max_radial)
        return derivatives

    if n_spline_points is not None:  # if user specifies the number of spline points
        positions = np.linspace(0.0, cutoff_radius, n_spline_points)  # spline positions
        values = value_evaluator_3D(positions)
        derivatives = derivative_evaluator_3D(positions)
    else:
        dynamic_spliner = DynamicSpliner(
            0.0,
            cutoff_radius,
            value_evaluator_3D,
            derivative_evaluator_3D,
            requested_accuracy,
        )
        positions, values, derivatives = dynamic_spliner.spline()

    # Convert positions, values, derivatives into the appropriate json formats:
    spline_points = []
    for position, value, derivative in zip(positions, values, derivatives):
        spline_points.append(
            {
                "position": position,
                "values": {
                    # this is the data representation used by ndarray through serde
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

    return spline_points


class DynamicSpliner:
    def __init__(
        self,
        start,
        stop,
        values_fn,
        derivatives_fn,
        requested_accuracy,
    ) -> None:
        """Dynamic spline generator.

        This class can be used to spline any set of functions defined within
        the start-stop interval. Cubic Hermite splines
        (https://en.wikipedia.org/wiki/Cubic_Hermite_spline) are used.
        The same spline points will be used for all functions, and more will
        be added until either the relative error or the absolute error fall below
        the requested accuracy on average across all functions.
        The functions are specified via values_fn and derivatives_fn.
        These must be able to take a numpy 1D array of positions as their input,
        and they must output a numpy array where the first dimension corresponds
        to the input positions, while other dimensions are arbitrary and can
        correspond to any way in which the target functions can be classified.
        The splines can be obtained via the spline method.
        """

        self.start = start
        self.stop = stop
        self.values_fn = values_fn
        self.derivatives_fn = derivatives_fn
        self.requested_accuracy = requested_accuracy

        # initialize spline with 11 points
        positions = np.linspace(start, stop, 11)
        self.spline_positions = positions
        self.spline_values = values_fn(positions)
        self.spline_derivatives = derivatives_fn(positions)

        self.number_of_custom_axes = len(self.spline_values.shape) - 1

    def spline(self):
        """Calculates and outputs the splines.

        The outputs of this function are, respectively:
        - A numpy 1D array containing the spline positions. These are equally
          spaced in the start-stop interval.
        - A numpy ndarray containing the values of the splined functions at the
          spline positions. The first dimension corresponds to the spline
          positions, while all subsequent dimensions are consistent with the
          values_fn and get_function_derivative provided during
          initialization of the class.
        - A numpy ndarray containing the derivatives of the splined functions
          at the spline positions, with the same structure as that of the
          ndarray of values.
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

        return np.array(interpolated_values)
