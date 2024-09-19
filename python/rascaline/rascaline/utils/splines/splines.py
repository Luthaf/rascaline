# """
# .. _python-splined-radial-integral:

# Splined radial integrals
# ========================

# Classes for generating splines which can be used as tabulated radial integrals in the
# various SOAP and LODE calculators.

# All classes are based on :py:class:`rascaline.utils.RadialIntegralSplinerBase`. We
# provides several ways to compute a radial integral: you may chose and initialize a pre
# defined atomic density and radial basis and provide them to
# :py:class:`rascaline.utils.SoapSpliner` or :py:class:`rascaline.utils.LodeSpliner`.
# Both classes require `scipy`_ to be installed in order to perform the numerical
# integrals.

# Alternatively, you can also explicitly provide functions for the radial integral and
# its derivative and passing them to
# :py:class:`rascaline.utils.RadialIntegralFromFunction`.

# .. autoclass:: rascaline.utils.RadialIntegralSplinerBase
#     :members:
#     :show-inheritance:

# .. autoclass:: rascaline.utils.SoapSpliner
#     :members:
#     :show-inheritance:

# .. autoclass:: rascaline.utils.LodeSpliner
#     :members:
#     :show-inheritance:

# .. autoclass:: rascaline.utils.RadialIntegralFromFunction
#     :members:
#     :show-inheritance:


# .. _`scipy`: https://scipy.org
# """

# from abc import ABC, abstractmethod
# from typing import Callable, Dict, Optional, Union

# import numpy as np


# try:
#     from scipy.integrate import dblquad, quad, quad_vec
#     from scipy.special import legendre, spherical_in, spherical_jn

#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# from .atomic_density import AtomicDensityBase, DeltaDensity, GaussianDensity
# from .radial_basis import RadialBasisBase


# class RadialIntegralSplinerBase(ABC):
#     """Base class for splining arbitrary radial integrals.

#     If :py:meth:`RadialIntegralSplinerBase.radial_integral_derivative` is not
#     implemented in a child class it will computed based on finite differences.

#     :parameter max_angular: number of radial components
#     :parameter max_radial: number of angular components
#     :parameter spline_cutoff: cutoff radius for the spline interpolation. This is also
#         the maximal value that can be interpolated.
#     :parameter basis: Provide a :class:`RadialBasisBase` instance to orthonormalize
#         the radial integral.
#     :parameter accuracy: accuracy of the numerical integration and the splining.
#         Accuracy is reached when either the mean absolute error or the mean relative
#         error gets below the ``accuracy`` threshold.
#     """

#     def __init__(
#         self,
#         max_radial: int,
#         max_angular: int,
#         spline_cutoff: float,
#         basis: Optional[RadialBasisBase],
#         accuracy: float,
#     ):
#         self.max_radial = max_radial
#         self.max_angular = max_angular
#         self.spline_cutoff = spline_cutoff
#         self.basis = basis
#         self.accuracy = accuracy

#     def compute(
#         self,
#         n_spline_points: Optional[int] = None,
#     ) -> Dict:
#         """Compute the spline for rascaline's tabulated radial integrals.

#         :parameter n_spline_points: Use fixed number of spline points instead of find
#             the number based on the provided ``accuracy``.
#         :returns dict: dictionary for the input as the ``radial_basis`` parameter  of
#             a rascaline calculator.
#         """

#         if self.basis is not None:
#             orthonormalization_matrix = self.basis.compute_orthonormalization_matrix(
#                 self.max_radial, self.max_angular
#             )
#         else:
#             orthonormalization_matrix = None

#         def value_evaluator_3D(positions):
#             return self._value_evaluator_3D(
#                 positions, orthonormalization_matrix, derivative=False
#             )

#         def derivative_evaluator_3D(positions):
#             return self._value_evaluator_3D(
#                 positions, orthonormalization_matrix, derivative=True
#             )

#         if n_spline_points is not None:
#             positions = np.linspace(0, self.spline_cutoff, n_spline_points)
#             values = value_evaluator_3D(positions)
#             derivatives = derivative_evaluator_3D(positions)
#         else:
#             dynamic_spliner = DynamicSpliner(
#                 0,
#                 self.spline_cutoff,
#                 value_evaluator_3D,
#                 derivative_evaluator_3D,
#                 self.accuracy,
#             )
#             positions, values, derivatives = dynamic_spliner.spline()

#         # Convert positions, values, derivatives into the appropriate json formats:
#         spline_points = []
#         for position, value, derivative in zip(positions, values, derivatives):
#             spline_points.append(
#                 {
#                     "position": position,
#                     "values": {
#                         "v": 1,
#                         "dim": value.shape,
#                         "data": value.flatten().tolist(),
#                     },
#                     "derivatives": {
#                         "v": 1,
#                         "dim": derivative.shape,
#                         "data": derivative.flatten().tolist(),
#                     },
#                 }
#             )

#         parameters = {"points": spline_points}

#         center_contribution = self.center_contribution
#         if center_contribution is not None:
#             if self.basis is not None:
#                 # consider only `l=0` component of the `orthonormalization_matrix`
#                 parameters["center_contribution"] = list(
#                     orthonormalization_matrix[0] @ center_contribution
#                 )
#             else:
#                 parameters["center_contribution"] = center_contribution

#         return {"TabulatedRadialIntegral": parameters}

#     @abstractmethod
#     def radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
#         """evaluate the radial integral"""
#         ...

#     @property
#     def center_contribution(self) -> Union[None, np.ndarray]:
#         r"""Pre-computed value for the contribution of the central atom.

#         Required for LODE calculations. The central atom contribution will be
#         orthonormalized in the same way as the radial integral.
#         """

#         return None

#     def radial_integral_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         """evaluate the derivative of the radial integral"""
#         displacement = 1e-6
#         mean_abs_positions = np.mean(np.abs(positions))

#         if mean_abs_positions < 1.0:
#             raise ValueError(
#                 "Numerically derivative of the radial integral can not be performed "
#                 "since positions are too small. Mean of the absolute positions is "
#                 f"{mean_abs_positions:.1e} but should be at least 1."
#             )

#         radial_integral_pos = self.radial_integral(n, ell, positions+displacement/2)
#         radial_integral_neg = self.radial_integral(n, ell, positions-displacement/2)

#         return (radial_integral_pos - radial_integral_neg) / displacement

#     def _value_evaluator_3D(
#         self,
#         positions: np.ndarray,
#         orthonormalization_matrix: Optional[np.ndarray],
#         derivative: bool,
#     ):
#         values = np.zeros([len(positions), self.max_angular + 1, self.max_radial])
#         for ell in range(self.max_angular + 1):
#             for n in range(self.max_radial):
#                 if derivative:
#                     values[:, ell, n] = self.radial_integral_derivative(
#                         n, ell, positions
#                     )
#                 else:
#                     values[:, ell, n] = self.radial_integral(n, ell, positions)

#         if orthonormalization_matrix is not None:
#             # For each l channel we do a dot product of the orthonormalization_matrix
#             #  of shape (n, n) with the values which should have the shape
#             # (n, n_positions). To achieve the correct broadcasting we have to
#             # transpose twice.
#             for ell in range(self.max_angular + 1):
#                 values[:, ell, :] = (
#                     orthonormalization_matrix[ell] @ values[:, ell, :].T
#                 ).T

#         return values


# class DynamicSpliner:
#     def __init__(
#         self,
#         start: float,
#         stop: float,
#         values_fn: Callable[[np.ndarray], np.ndarray],
#         derivatives_fn: Callable[[np.ndarray], np.ndarray],
#         accuracy: float = 1e-8,
#     ) -> None:
#         """Dynamic spline generator.

#         This class can be used to spline any set of functions defined within the
#         start-stop interval. Cubic Hermite splines
#         (https://en.wikipedia.org/wiki/Cubic_Hermite_spline) are used. The same spline
#         points will be used for all functions, and more will be added until either the
#         relative error or the absolute error fall below the requested accuracy on
#         average across all functions. The functions are specified via values_fn and
#         derivatives_fn. These must be able to take a numpy 1D array of positions as
#         their input, and they must output a numpy array where the first dimension
#         corresponds to the input positions, while other dimensions are arbitrary and
#         can correspond to any way in which the target functions can be classified. The
#         splines can be obtained via the spline method.
#         """

#         self.start = start
#         self.stop = stop
#         self.values_fn = values_fn
#         self.derivatives_fn = derivatives_fn
#         self.requested_accuracy = accuracy

#         # initialize spline with 11 points
#         positions = np.linspace(start, stop, 11)
#         self.spline_positions = positions
#         self.spline_values = values_fn(positions)
#         self.spline_derivatives = derivatives_fn(positions)

#         self.number_of_custom_axes = len(self.spline_values.shape) - 1

#     def spline(self):
#         """Calculates and outputs the splines.

#         The outputs of this function are, respectively:

#         - A numpy 1D array containing the spline positions. These are equally spaced
#           in the start-stop interval.
#         - A numpy ndarray containing the values of the splined functions at the spline
#           positions. The first dimension corresponds to the spline positions, while
#           all subsequent dimensions are consistent with the values_fn and
#           `get_function_derivative` provided during initialization of the class.
#         - A numpy ndarray containing the derivatives of the splined functions at the
#           spline positions, with the same structure as that of the ndarray of values.
#         """

#         while True:
#             n_intermediate_positions = len(self.spline_positions) - 1

#             if n_intermediate_positions >= 50000:
#                 raise ValueError(
#                     "Maximum number of spline points reached. \
#                     There might be a problem with the functions to be splined"
#                 )

#             half_step = (self.spline_positions[1] - self.spline_positions[0]) / 2
#             intermediate_positions = np.linspace(
#                 self.start + half_step, self.stop-half_step,n_intermediate_positions
#             )

#             estimated_values = self._compute_from_spline(intermediate_positions)
#             new_values = self.values_fn(intermediate_positions)

#             mean_absolute_error = np.mean(np.abs(estimated_values - new_values))
#             with np.errstate(divide="ignore"):  # Ignore divide-by-zero warnings
#                 mean_relative_error = np.mean(
#                     np.abs((estimated_values - new_values) / new_values)
#                 )

#             if (
#                 mean_absolute_error < self.requested_accuracy
#                 or mean_relative_error < self.requested_accuracy
#             ):
#                 break

#             new_derivatives = self.derivatives_fn(intermediate_positions)

#             concatenated_positions = np.concatenate(
#                 [self.spline_positions, intermediate_positions], axis=0
#             )
#             concatenated_values = np.concatenate(
#                 [self.spline_values, new_values], axis=0
#             )
#             concatenated_derivatives = np.concatenate(
#                 [self.spline_derivatives, new_derivatives], axis=0
#             )

#             sort_indices = np.argsort(concatenated_positions, axis=0)

#             self.spline_positions = concatenated_positions[sort_indices]
#             self.spline_values = concatenated_values[sort_indices]
#             self.spline_derivatives = concatenated_derivatives[sort_indices]

#         return self.spline_positions, self.spline_values, self.spline_derivatives

#     def _compute_from_spline(self, positions):
#         x = positions
#         delta_x = self.spline_positions[1] - self.spline_positions[0]
#         n = (np.floor(x / delta_x)).astype(np.int32)

#         t = (x - n * delta_x) / delta_x
#         t_2 = t**2
#         t_3 = t**3

#         h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
#         h10 = t_3 - 2.0 * t_2 + t
#         h01 = -2.0 * t_3 + 3.0 * t_2
#         h11 = t_3 - t_2

#         p_k = self.spline_values[n]
#         p_k_1 = self.spline_values[n + 1]

#         m_k = self.spline_derivatives[n]
#         m_k_1 = self.spline_derivatives[n + 1]

#         new_shape = (-1,) + (1,) * self.number_of_custom_axes
#         h00 = h00.reshape(new_shape)
#         h10 = h10.reshape(new_shape)
#         h01 = h01.reshape(new_shape)
#         h11 = h11.reshape(new_shape)

#         interpolated_values = (
#             h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
#         )

#         return interpolated_values


# class RadialIntegralFromFunction(RadialIntegralSplinerBase):
#     r"""Compute radial integral spline points based on a provided function.

#     :parameter radial_integral: Function to compute the radial integral. Function must
#         take ``n``, ``l``, and ``positions`` as inputs, where ``n`` and ``l`` are
#         integers and ``positions`` is a numpy 1-D array that contains the spline
#         points at which the radial integral will be evaluated. The function must
#         return a numpy 1-D array containing the values of the radial integral.
#     :parameter spline_cutoff: cutoff radius for the spline interpolation. This is also
#         the maximal value that can be interpolated.
#     :parameter max_radial: number of angular components
#     :parameter max_angular: number of radial components
#     :parameter radial_integral_derivative: The derivative of the radial integral
#         taking the same parameters as ``radial_integral``. If it is ``None``
#         (default), finite differences are used to calculate the derivative of the
#         radial integral. It is recommended to provide this parameter if possible.
#         Derivatives from finite differences can cause problems when evaluating at the
#         edges of the domain (i.e., at ``0`` and ``spline_cutoff``) because the
#         function might not be defined outside of the domain.
#     :parameter accuracy: accuracy of the numerical integration and the splining.
#         Accuracy is reached when either the mean absolute error or the mean relative
#         error gets below the ``accuracy`` threshold.
#     :parameter center_contribution: Contribution of the central atom required for LODE
#         calculations. The ``center_contribution`` is defined as

#         .. math::
#            c_n = \sqrt{4π}\int_0^\infty dr r^2 R_n(r) g(r)

#         where :math:`g(r)` is the radially symmetric density function, `R_n(r)` the
#         radial basis function and :math:`n` the current radial channel. This should be
#         pre-computed and provided as a separate parameter.

#     Example
#     -------
#     First define a ``radial_integral`` function

#     >>> def radial_integral(n, ell, r):
#     ...     return np.sin(r)

#     and provide this as input to the spline generator

#     >>> spliner = RadialIntegralFromFunction(
#     ...     radial_integral=radial_integral,
#     ...     max_radial=12,
#     ...     max_angular=9,
#     ...     spline_cutoff=8.0,
#     ... )

#     Finally, we can use the ``spliner`` directly in the ``radial_integral`` section
#     of a calculator

#     >>> from rascaline import SoapPowerSpectrum
#     >>> calculator = SoapPowerSpectrum(
#     ...     cutoff=8.0,
#     ...     max_radial=12,
#     ...     max_angular=9,
#     ...     center_atom_weight=1.0,
#     ...     radial_basis=spliner.compute(),
#     ...     atomic_gaussian_width=1.0,  # ignored
#     ...     cutoff_function={"Step": {}},
#     ... )

#     The ``atomic_gaussian_width`` parameter is required by the calculator but will be
#     will be ignored during the feature computation.

#     A more in depth example using a "rectangular" Laplacian eigenstate (LE) basis is
#     provided in the :ref:`userdoc-how-to-splined-radial-integral` how-to guide.
#     """

#     def __init__(
#         self,
#         radial_integral: Callable[[int, int, np.ndarray], np.ndarray],
#         spline_cutoff: float,
#         max_radial: int,
#         max_angular: int,
#         radial_integral_derivative: Optional[
#             Callable[[int, int, np.ndarray], np.ndarray]
#         ] = None,
#         center_contribution: Optional[np.ndarray] = None,
#         accuracy: float = 1e-8,
#     ):
#         self.radial_integral_function = radial_integral
#         self.radial_integral_derivative_function = radial_integral_derivative

#         if center_contribution is not None and len(center_contribution) != max_radial:
#             raise ValueError(
#                 f"center contribution has {len(center_contribution)} entries but "
#                 f"should be the same as max_radial ({max_radial})"
#             )

#         self._center_contribution = center_contribution

#         super().__init__(
#             max_radial=max_radial,
#             max_angular=max_angular,
#             spline_cutoff=spline_cutoff,
#             basis=None,  # do no orthonormalize the radial integral
#             accuracy=accuracy,
#         )

#     def radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
#         return self.radial_integral_function(n, ell, positions)

#     @property
#     def center_contribution(self) -> Union[None, np.ndarray]:
#         return self._center_contribution

#     def radial_integral_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         if self.radial_integral_derivative_function is None:
#             return super().radial_integral_derivative(n, ell, positions)
#         else:
#             return self.radial_integral_derivative_function(n, ell, positions)


# class SoapSpliner(RadialIntegralSplinerBase):
#     """Compute radial integral spline points for real space calculators.

#     Use only in combination with a real space calculators like
#     :class:`rascaline.SphericalExpansion` or :class:`rascaline.SoapPowerSpectrum`. For
#     k-space spherical expansions use :class:`LodeSpliner`.

#     If ``density`` is either :class:`rascaline.utils.DeltaDensity` or
#     :class:`rascaline.utils.GaussianDensity` the radial integral will be partly solved
#     analytical. These simpler expressions result in a faster and more stable
#     evaluation.

#     :parameter cutoff: spherical cutoff for the radial basis
#     :parameter max_radial: number of angular components
#     :parameter max_angular: number of radial components
#     :parameter basis: definition of the radial basis
#     :parameter density: definition of the atomic density
#     :parameter accuracy: accuracy of the numerical integration and the splining.
#         Accuracy is reached when either the mean absolute error or the mean relative
#         error gets below the ``accuracy`` threshold.
#     :raise ValueError: if `scipy`_ is not available

#     Example
#     -------

#     First import the necessary classed and define hyper parameters for the spherical
#     expansions.

#     >>> from rascaline import SphericalExpansion
#     >>> from rascaline.utils import GaussianDensity, GtoBasis

#     >>> cutoff = 2
#     >>> max_radial = 6
#     >>> max_angular = 4
#     >>> atomic_gaussian_width = 1.0

#     Next we initialize our radial basis and the density

#     >>> basis = GtoBasis(cutoff=cutoff, max_radial=max_radial)
#     >>> density = GaussianDensity(atomic_gaussian_width=atomic_gaussian_width)

#     And finally the actual spliner instance

#     >>> spliner = SoapSpliner(
#     ...     cutoff=cutoff,
#     ...     max_radial=max_radial,
#     ...     max_angular=max_angular,
#     ...     basis=basis,
#     ...     density=density,
#     ...     accuracy=1e-3,
#     ... )

#     Above we reduced ``accuracy`` from the default value of ``1e-8`` to ``1e-3`` to
#     speed up calculations.

#     As for all spliner classes you can use the output
#     :meth:`RadialIntegralSplinerBase.compute` method directly as the ``radial_basis``
#     parameter.

#     >>> calculator = SphericalExpansion(
#     ...     cutoff=cutoff,
#     ...     max_radial=max_radial,
#     ...     max_angular=max_angular,
#     ...     center_atom_weight=1.0,
#     ...     atomic_gaussian_width=atomic_gaussian_width,
#     ...     radial_basis=spliner.compute(),
#     ...     cutoff_function={"Step": {}},
#     ... )

#     You can now use ``calculator`` to obtain the spherical expansion coefficients of
#     your systems. Note that the the spliner based used here will produce the same
#     coefficients as if ``radial_basis={"Gto": {}}`` would be used.

#     An additional example using a "rectangular" Laplacian eigenstate (LE) basis is
#     provided in the :ref:`userdoc-how-to-le-basis`.

#     .. seealso::
#         :class:`LodeSpliner` for a spliner class that works with
#         :class:`rascaline.LodeSphericalExpansion`
#     """

#     def __init__(
#         self,
#         cutoff: float,
#         max_radial: int,
#         max_angular: int,
#         basis: RadialBasisBase,
#         density: AtomicDensityBase,
#         accuracy: float = 1e-8,
#     ):
#         if not HAS_SCIPY:
#             raise ValueError("Spliner class requires scipy!")

#         self.density = density

#         super().__init__(
#             max_radial=max_radial,
#             max_angular=max_angular,
#             spline_cutoff=cutoff,
#             basis=basis,
#             accuracy=accuracy,
#         )

#     def radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
#         if type(self.density) is DeltaDensity:
#             return self._radial_integral_delta(n, ell, positions)
#         elif type(self.density) is GaussianDensity:
#             return self._radial_integral_gaussian(n, ell, positions)
#         else:
#             return self._radial_integral_custom(n, ell, positions)

#     def radial_integral_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         if type(self.density) is DeltaDensity:
#             return self._radial_integral_delta_derivative(n, ell, positions)
#         elif type(self.density) is GaussianDensity:
#             return self._radial_integral_gaussian_derivative(n, ell, positions)
#         else:
#             return self._radial_integral_custom(n, ell, positions, derivative=True)

#     def _radial_integral_delta(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         return self.basis.compute(n, ell, positions)

#     def _radial_integral_delta_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         return self.basis.compute_derivative(n, ell, positions)

#     def _radial_integral_gaussian(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         atomic_gaussian_width_sq = self.density.atomic_gaussian_width**2

#         prefactor = (
#             (4 * np.pi)
#             / (np.pi * atomic_gaussian_width_sq) ** (3 / 4)
#             * np.exp(-0.5 * positions**2 / atomic_gaussian_width_sq)
#         )

#         def integrand(
#             integrand_position: float, n: int, ell: int, positions: np.array
#         ) -> np.ndarray:
#             return (
#                 integrand_position**2
#                 * self.basis.compute(n, ell, integrand_position)
#                 * np.exp(-0.5 * integrand_position**2 / atomic_gaussian_width_sq)
#                 * spherical_in(
#                     ell,
#                     integrand_position * positions / atomic_gaussian_width_sq,
#                 )
#             )

#         return (
#             prefactor
#             * quad_vec(
#                 f=integrand,
#                 a=0,
#                 b=self.basis.integration_radius,
#                 args=(n, ell, positions),
#             )[0]
#         )

#     def _radial_integral_gaussian_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         # The derivative here for `positions=0`, any `n` and `ell=1` are wrong due to
#         # a bug in Scipy: https://github.com/scipy/scipy/issues/20506
#         #
#         # However, this is not problematic because the derivative at zero is only
#         # required if two atoms are VERY close and we have checks that should prevent
#         # very small distances between atoms. The center contribution is also not
#         # affected becuase it only needs the values but not derivatives of the radial
#         # integral.
#         atomic_gaussian_width_sq = self.density.atomic_gaussian_width**2

#         prefactor = (
#             (4 * np.pi)
#             / (np.pi * atomic_gaussian_width_sq) ** (3 / 4)
#             * np.exp(-0.5 * positions**2 / atomic_gaussian_width_sq)
#         )

#         def integrand(
#             integrand_position: float, n: int, ell: int, positions: np.array
#         ) -> np.ndarray:
#             return (
#                 integrand_position**3
#                 * self.basis.compute(n, ell, integrand_position)
#                 * np.exp(-(integrand_position**2) / (2 * atomic_gaussian_width_sq))
#                 * spherical_in(
#                     ell,
#                     integrand_position * positions / atomic_gaussian_width_sq,
#                     derivative=True,
#                 )
#             )

#         return atomic_gaussian_width_sq**-1 * (
#             prefactor
#             * quad_vec(
#                 f=integrand,
#                 a=0,
#                 b=self.basis.integration_radius,
#                 args=(n, ell, positions),
#             )[0]
#             - positions * self._radial_integral_gaussian(n, ell, positions)
#         )

#     def _radial_integral_custom(
#         self, n: int, ell: int, positions: np.ndarray, derivative: bool = False
#     ) -> np.ndarray:
#         P_ell = legendre(ell)

#         if derivative:

#             def integrand(
#                 u: float, integrand_position: float, n: int, ell: int, position: float
#             ) -> float:
#                 arg = np.sqrt(
#                     integrand_position**2
#                     + position**2
#                     - 2 * integrand_position * position * u
#                 )

#                 return (
#                     integrand_position**2
#                     * self.basis.compute(n, ell, integrand_position)
#                     * P_ell(u)
#                     * (position - u * integrand_position)
#                     * self.density.compute_derivative(arg)
#                     / arg
#                 )

#         else:

#             def integrand(
#                 u: float, integrand_position: float, n: int, ell: int, position: float
#             ) -> float:
#                 arg = np.sqrt(
#                     integrand_position**2
#                     + position**2
#                     - 2 * integrand_position * position * u
#                 )

#                 return (
#                     integrand_position**2
#                     * self.basis.compute(n, ell, integrand_position)
#                     * P_ell(u)
#                     * self.density.compute(arg)
#                 )

#         radial_integral = np.zeros(len(positions))

#         for i, position in enumerate(positions):
#             radial_integral[i], _ = dblquad(
#                 func=integrand,
#                 a=0,
#                 b=self.basis.integration_radius,
#                 gfun=-1,
#                 hfun=1,
#                 args=(n, ell, position),
#             )

#         return 2 * np.pi * radial_integral


# class LodeSpliner(RadialIntegralSplinerBase):
#     r"""Compute radial integral spline points for k-space calculators.

#     Use only in combination with a k-space/Fourier-space calculators like
#     :class:`rascaline.LodeSphericalExpansion`. For real space spherical expansions use
#     :class:`SoapSpliner`.

#     :parameter k_cutoff: spherical reciprocal cutoff
#     :parameter max_radial: number of angular components
#     :parameter max_angular: number of radial components
#     :parameter basis: definition of the radial basis
#     :parameter density: definition of the atomic density
#     :parameter accuracy: accuracy of the numerical integration and the splining.
#         Accuracy is reached when either the mean absolute error or the mean relative
#         error gets below the ``accuracy`` threshold.
#     :raise ValueError: if `scipy`_ is not available

#     Example
#     -------

#     First import the necessary classed and define hyper parameters for the spherical
#     expansions.

#     >>> from rascaline import LodeSphericalExpansion
#     >>> from rascaline.utils import GaussianDensity, GtoBasis

#     Note that ``cutoff`` defined below denotes the maximal distance for the projection
#     of the density. In contrast to SOAP, LODE also takes atoms outside of this
#     ``cutoff`` into account for the density.

#     >>> cutoff = 2
#     >>> max_radial = 6
#     >>> max_angular = 4
#     >>> atomic_gaussian_width = 1.0

#     :math:`1.2 \, \pi \, \sigma` where :math:`\sigma` is the ``atomic_gaussian_width``
#     which is a reasonable value for most systems.

#     >>> k_cutoff = 1.2 * np.pi / atomic_gaussian_width

#     Next we initialize our radial basis and the density

#     >>> basis = GtoBasis(cutoff=cutoff, max_radial=max_radial)
#     >>> density = GaussianDensity(atomic_gaussian_width=atomic_gaussian_width)

#     And finally the actual spliner instance

#     >>> spliner = LodeSpliner(
#     ...     k_cutoff=k_cutoff,
#     ...     max_radial=max_radial,
#     ...     max_angular=max_angular,
#     ...     basis=basis,
#     ...     density=density,
#     ... )

#     As for all spliner classes you can use the output
#     :meth:`RadialIntegralSplinerBase.compute` method directly as the ``radial_basis``
#     parameter.

#     >>> calculator = LodeSphericalExpansion(
#     ...     cutoff=cutoff,
#     ...     max_radial=max_radial,
#     ...     max_angular=max_angular,
#     ...     center_atom_weight=1.0,
#     ...     atomic_gaussian_width=atomic_gaussian_width,
#     ...     potential_exponent=1,
#     ...     radial_basis=spliner.compute(),
#     ... )

#     You can now use ``calculator`` to obtain the spherical expansion coefficients of
#     your systems. Note that the the spliner based used here will produce the same
#     coefficients as if ``radial_basis={"Gto": {}}`` would be used.

#     .. seealso::
#         :class:`SoapSpliner` for a spliner class that works with
#         :class:`rascaline.SphericalExpansion`
#     """

#     def __init__(
#         self,
#         k_cutoff: float,
#         max_radial: int,
#         max_angular: int,
#         basis: RadialBasisBase,
#         density: AtomicDensityBase,
#         accuracy: float = 1e-8,
#     ):
#         if not HAS_SCIPY:
#             raise ValueError("Spliner class requires scipy!")

#         self.density = density

#         super().__init__(
#             max_radial=max_radial,
#             max_angular=max_angular,
#             basis=basis,
#             spline_cutoff=k_cutoff,  # use k_cutoff here because we spline in k-space
#             accuracy=accuracy,
#         )

#     def radial_integral(self, n: int, ell: int, positions: np.ndarray) -> np.ndarray:
#         def integrand(
#             integrand_position: float, n: int, ell: int, positions: np.ndarray
#         ) -> np.ndarray:
#             return (
#                 integrand_position**2
#                 * self.basis.compute(n, ell, integrand_position)
#                 * spherical_jn(ell, integrand_position * positions)
#             )

#         return quad_vec(
#             f=integrand,
#             a=0,
#             b=self.basis.integration_radius,
#             args=(n, ell, positions),
#         )[0]

#     def radial_integral_derivative(
#         self, n: int, ell: int, positions: np.ndarray
#     ) -> np.ndarray:
#         def integrand(
#             integrand_position: float, n: int, ell: int, positions: np.ndarray
#         ) -> np.ndarray:
#             return (
#                 integrand_position**3
#                 * self.basis.compute(n, ell, integrand_position)
#                 * spherical_jn(ell, integrand_position * positions, derivative=True)
#             )

#         return quad_vec(
#             f=integrand,
#             a=0,
#             b=self.basis.integration_radius,
#             args=(n, ell, positions),
#         )[0]

#     @property
#     def center_contribution(self) -> np.ndarray:
#         if type(self.density) is DeltaDensity:
#             center_contrib = self._center_contribution_delta
#         else:
#             center_contrib = self._center_contribution_custom

#        return [np.sqrt(4 * np.pi) * center_contrib(n) for n in range(self.max_radial)]

#     def _center_contribution_delta(self, n: int):
#         raise NotImplementedError(
#             "center contribution for delta distributions is not implemented yet."
#         )

#     def _center_contribution_custom(self, n: int):
#         def integrand(integrand_position: float, n: int) -> np.ndarray:
#             return (
#                 integrand_position**2
#                 * self.basis.compute(n, 0, integrand_position)
#                 * self.density.compute(integrand_position)
#             )

#         return quad(
#             func=integrand,
#             a=0,
#             b=self.basis.integration_radius,
#             args=(n,),
#         )[0]
