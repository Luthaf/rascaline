import functools
from typing import Callable, Optional

import numpy as np


try:
    import scipy.integrate
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .basis import ExpansionBasis, Explicit, RadialBasis
from .cutoff import Cutoff
from .density import AtomicDensity, DiracDelta, Gaussian, SmearedPowerLaw


class Spline:
    """
    `Cubic Hermite splines`_ implementation

    .. _Cubic Hermit splines: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    """

    def __init__(self, start, stop):
        self.start = float(start)
        self.stop = float(stop)

        self.positions = None
        self.values = None
        self.derivatives = None

    def __len__(self):
        if self.positions is None:
            return 0
        else:
            return len(self.positions)

    def compute(
        self,
        positions: np.ndarray,
        n: Optional[int] = None,
        *,
        derivative=False,
    ) -> np.ndarray:
        """
        Evaluate the spline at the given ``positions``, optionally evaluating only
        the value for the ``n``'th splined function.
        """

        if self.positions is None:
            raise ValueError("you must add points to the spline before evaluating it")

        if n is None:
            n = ...

        x = positions
        delta_x = self.positions[1] - self.positions[0]
        k = (np.floor(x / delta_x)).astype(np.int64)

        t = (x - k * delta_x) / delta_x
        t_2 = t**2
        t_3 = t**3

        h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
        h10 = t_3 - 2.0 * t_2 + t
        h01 = -2.0 * t_3 + 3.0 * t_2
        h11 = t_3 - t_2

        p_k = self.values[k, n]
        p_k_1 = self.values[k + 1, n]

        m_k = self.derivatives[k, n]
        m_k_1 = self.derivatives[k + 1, n]

        new_shape = (-1,) + (1,) * (len(p_k.shape) - 1)
        h00 = h00.reshape(new_shape)
        h10 = h10.reshape(new_shape)
        h01 = h01.reshape(new_shape)
        h11 = h11.reshape(new_shape)

        if derivative:
            d_h00_dt = 6.0 * (t_2 - t)
            d_h10_dt = 3.0 * t_2 - 4.0 * t + 1.0
            d_h01_dt = -d_h00_dt
            d_h11_dt = 3.0 * t_2 - 2.0 * t

            dx_dt = 1.0 / delta_x

            return (
                d_h00_dt * p_k * dx_dt
                + d_h10_dt * m_k
                + d_h01_dt * p_k_1 * dx_dt
                + d_h11_dt * m_k_1
            )

        else:
            return h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1

    def add_points(self, positions, values, derivatives):
        """Add points to the spline

        :param positions: positions of all points
        :param values: values of the splined functions at the points positions
        :param derivatives: derivative of the splined functions at the points positions
        """
        positions = np.asarray(positions)
        values = np.asarray(values)
        derivatives = np.asarray(derivatives)

        if not np.all(np.isfinite(positions)):
            raise ValueError(
                "new spline points `positions` contains NaN/infinity, "
                "numerical integrals are not converging"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "new spline points `values` contains NaN/infinity, "
                "numerical integrals are not converging"
            )

        if not np.all(np.isfinite(derivatives)):
            raise ValueError(
                "new spline points `derivatives` contains NaN/infinity, "
                "numerical integrals are not converging"
            )

        assert len(positions.shape) == 1
        assert np.min(positions) >= self.start
        assert np.max(positions) <= self.stop

        assert values.shape == derivatives.shape
        assert values.shape[0] == len(positions)
        assert derivatives.shape[0] == len(positions)

        if self.values is not None:
            assert values.shape[1:] == self.values.shape[1:]
            assert derivatives.shape[1:] == self.derivatives.shape[1:]

            positions = np.concatenate([self.positions, positions], axis=0)
            values = np.concatenate([self.values, values], axis=0)
            derivatives = np.concatenate([self.derivatives, derivatives], axis=0)

        sort_indices = np.argsort(positions, axis=0)
        self.positions = positions[sort_indices]
        self.values = values[sort_indices]
        self.derivatives = derivatives[sort_indices]

        assert len(self.values.shape) == 2

    @staticmethod
    def with_accuracy(
        start: float,
        stop: float,
        values_fn: Callable[[np.ndarray], np.ndarray],
        derivatives_fn: Callable[[np.ndarray], np.ndarray],
        accuracy: float,
    ) -> "Spline":
        """
        Create a :py:class:`Spline` using any set of functions defined within the
        ``start, stop`` interval. The same spline points will be used for all functions,
        and more will be added until either the relative error or the absolute error
        fall below the requested accuracy on average across all functions.

        The functions are specified via ``values_fn`` and ``derivatives_fn``. These must
        take the positions of new spline points as their input, and they must output a
        numpy array containing the full set of function evaluated at these points.
        """
        # initialize spline with 11 points
        spline = Spline(start=start, stop=stop)
        positions = np.linspace(start, stop, 11)
        spline.add_points(positions, values_fn(positions), derivatives_fn(positions))

        while True:
            n_intermediate_positions = len(spline) - 1

            if n_intermediate_positions >= 50000:
                raise ValueError(
                    "Maximum number of spline points reached. \
                    There might be a problem with the functions to be splined"
                )

            half_step = (spline.positions[1] - spline.positions[0]) / 2
            intermediate_positions = np.linspace(
                start + half_step, stop - half_step, n_intermediate_positions
            )

            estimated_values = spline.compute(intermediate_positions)
            new_values = values_fn(intermediate_positions)

            mean_absolute_error = np.mean(np.abs(estimated_values - new_values))
            with np.errstate(divide="ignore"):  # Ignore divide-by-zero warnings
                relative_error = np.abs((estimated_values - new_values) / (new_values))
                mean_relative_error = np.mean(
                    # exclude points where the denominator is (almost) zero
                    relative_error[np.where(np.abs(new_values) > 1e-16)]
                )

            if mean_absolute_error < accuracy or mean_relative_error < accuracy:
                break

            new_derivatives = derivatives_fn(intermediate_positions)

            spline.add_points(intermediate_positions, new_values, new_derivatives)

        return spline


class SplinedRadialBasis(RadialBasis):
    """
    Radial basis based on a spline. This is mainly intended to be used to transfer hyper
    parameters to the native code, but can also be used to check the exact shape of the
    splined radial basis.
    """

    def __init__(
        self,
        *,
        spline: Spline,
        max_radial: int,
        radius: float,
        lode_center_contribution: Optional[np.ndarray] = None,
    ):
        super().__init__(max_radial=max_radial, radius=radius)
        self.spline = spline
        self.lode_center_contribution = lode_center_contribution

    def get_hypers(self):
        hypers = {
            "type": "Tabulated",
            "points": [
                {
                    "position": float(p),
                    "values": v.tolist(),
                    "derivatives": d.tolist(),
                }
                for p, v, d in zip(
                    self.spline.positions,
                    self.spline.values,
                    self.spline.derivatives,
                )
            ],
        }

        if self.lode_center_contribution is not None:
            hypers["center_contribution"] = self.lode_center_contribution.tolist()

        return hypers

    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative: bool
    ) -> np.ndarray:
        return self.spline.compute(positions, n, derivative=derivative)


def _spherical_bessel_scaled(ell, z):
    """Compute ``exp(-z) * spherical_in(ell, z)``, avoiding overflow in
    ``spherical_in`` for large values of ``z``"""
    one_over_z = np.divide(1.0, z, out=np.zeros_like(z), where=z != 0)

    result = np.sqrt(0.5 * np.pi * one_over_z) * scipy.special.ive(ell + 0.5, z)

    if ell == 0:
        return np.where(z == 0, 1.0, result)
    else:
        return result


class SoapSpliner:
    """Compute an explicit spline of the radial integral for SOAP calculators.

    This allows a great deal of customization in the radial basis function used (any
    child class of :py:class:`RadialBasis`) and atomic density (any child class of
    :py:class:`AtomicDensity`). This way, users can define custom densities and/or
    basis, and use them with any of the SOAP calculators. For more information about the
    radial integral, you can refer to :ref:`this document <radial-integral>`.

    This class should be used only in combination with SOAP calculators like
    :class:`featomic.SphericalExpansion` or :class:`featomic.SoapPowerSpectrum`. For
    k-space spherical expansions use :class:`LodeSpliner`.

    If ``density`` is either :class:`featomic.density.Delta` or
    :class:`featomic.density.Gaussian` the radial integral will be partly solved
    analytically, for faster and more stable evaluation.

    Example
    -------

    First let's define the hyper parameters for the spherical expansions. It is
    important to note that only class-based hyper parameters are supported (in
    opposition to ``dict`` based hyper parameters).

    >>> import featomic

    >>> cutoff = featomic.cutoff.Cutoff(radius=2, smoothing=None)
    >>> density = featomic.density.Gaussian(width=1.0)
    >>> basis = featomic.basis.TensorProduct(
    ...     max_angular=4,
    ...     radial=featomic.basis.Gto(max_radial=5, radius=2),
    ...     # use a reduced ``accuracy`` of ``1e-3`` (the default is ``1e-8``)
    ...     # to speed up calculations.
    ...     spline_accuracy=1e-3,
    ... )

    From here we can initialize the spliner instance

    >>> spliner = SoapSpliner(cutoff=cutoff, density=density, basis=basis)

    You can then give the result of :meth:`SoapSpliner.get_hypers()` directly the
    calculator:

    >>> calculator = featomic.SphericalExpansion(**spliner.get_hypers())

    .. seealso::
        :class:`LodeSpliner` for a spliner class that works with
        :class:`featomic.LodeSphericalExpansion`
    """

    def __init__(
        self,
        *,
        cutoff: Cutoff,
        density: AtomicDensity,
        basis: ExpansionBasis,
        n_spline_points: Optional[int] = None,
    ):
        """
        :param cutoff: description of the local atomic environment, defined by a
            spherical cutoff
        :param density: atomic density that should be expanded for all neighbors in the
            local atomic environment
        :param basis: basis function to use to expand the neighbors' atomic density.
        :param n_spline_points: number of spline points to use. If ``None``, points will
            be added to the spline until the accuracy is at least the requested
            ``basis.spline_accuracy``.
        """
        if not HAS_SCIPY:
            raise ImportError("SoapSpliner class requires scipy")

        if not isinstance(cutoff, Cutoff):
            raise TypeError(f"`cutoff` should be a `Cutoff` object, got {type(cutoff)}")

        if not isinstance(density, AtomicDensity):
            raise TypeError(
                f"`density` should be an `AtomicDensity` object, got {type(density)}"
            )

        if not isinstance(basis, ExpansionBasis):
            raise TypeError(
                f"`basis` should be an `ExpansionBasis object, got {type(basis)}"
            )

        self.cutoff = cutoff
        self.density = density
        self.basis = basis

        self.n_spline_points = n_spline_points

    def get_hypers(self):
        """Get the SOAP hyper-parameters for the splined basis and density."""
        # This class works by computing a spline for the radial integral, and then using
        # this spline as a basis in combination with a DiracDelta density.
        hypers = {
            "cutoff": self.cutoff,
            "density": DiracDelta(
                center_atom_weight=self.density.center_atom_weight,
                scaling=self.density.scaling,
            ),
        }

        def generic_values_fn(positions, radial_size, angular, orthonormalization):
            integrals = np.vstack(
                [
                    self._radial_integral(radial, angular, positions, derivative=False)
                    for radial in range(radial_size)
                ]
            )
            return (orthonormalization @ integrals).T

        def generic_derivatives_fn(positions, radial_size, angular, orthonormalization):
            integrals = np.vstack(
                [
                    self._radial_integral(radial, angular, positions, derivative=True)
                    for radial in range(radial_size)
                ]
            )
            return (orthonormalization @ integrals).T

        # We transform whatever basis the user provided to an "explicit" basis, since
        # the radial integral will be different for different angular channels.
        by_angular = {}
        for angular in self.basis.angular_channels():
            radial_basis = self.basis.radial_basis(angular)
            orthonormalization = radial_basis._get_orthonormalization_matrix()

            values_fn = functools.partial(
                generic_values_fn,
                radial_size=radial_basis.size,
                angular=angular,
                orthonormalization=orthonormalization,
            )
            derivatives_fn = functools.partial(
                generic_derivatives_fn,
                radial_size=radial_basis.size,
                angular=angular,
                orthonormalization=orthonormalization,
            )

            if self.n_spline_points is not None:
                positions = np.linspace(0, self.cutoff.radius, self.n_spline_points)
                spline = Spline(0, self.cutoff.radius)
                spline.add_points(
                    positions=positions,
                    values=values_fn(positions),
                    derivatives=derivatives_fn(positions),
                )
            else:
                spline = Spline.with_accuracy(
                    start=0,
                    stop=self.cutoff.radius,
                    values_fn=values_fn,
                    derivatives_fn=derivatives_fn,
                    accuracy=(
                        1e-8
                        if self.basis.spline_accuracy is None
                        else self.basis.spline_accuracy
                    ),
                )

            by_angular[angular] = SplinedRadialBasis(
                spline=spline,
                max_radial=radial_basis.max_radial,
                radius=self.cutoff.radius,
            )

        hypers["basis"] = Explicit(by_angular=by_angular, spline_accuracy=None)
        return hypers

    def _radial_integral(
        self,
        radial: int,
        angular: int,
        distances: np.ndarray,
        derivative: bool,
    ) -> np.ndarray:
        """Compute the SOAP radial integral with a given density and radial basis"""
        if isinstance(self.density, DiracDelta):
            radial_basis = self.basis.radial_basis(angular)
            return radial_basis.compute_primitive(
                distances, radial, derivative=derivative
            )
        elif isinstance(self.density, Gaussian):
            if derivative:
                return self._ri_gaussian_density_derivative(radial, angular, distances)
            else:
                return self._ri_gaussian_density(radial, angular, distances)
        else:
            return self._ri_custom_density(
                radial, angular, distances, derivative=derivative
            )

    def _ri_gaussian_density(
        self,
        radial: int,
        angular: int,
        distances: np.ndarray,
    ) -> np.ndarray:
        # This code is derived in
        # https://metatensor.github.io/featomic/latest/devdoc/explanations/radial-integral.html,
        # and follows the same naming convention as the documentation.
        sigma_sq = self.density.width**2

        prefactor = (
            (4 * np.pi)
            / (np.pi * sigma_sq) ** (3 / 4)
            * np.exp(-0.5 * distances**2 / sigma_sq)
        )

        radial_basis = self.basis.radial_basis(angular)

        def integrand(r: float, n: int, ell: int, rij: np.array) -> np.ndarray:
            Rnl = radial_basis.compute_primitive(r, n, derivative=False)

            z = r * rij / sigma_sq
            bessel = _spherical_bessel_scaled(ell, z)

            return r**2 * Rnl * np.exp(z - 0.5 * r**2 / sigma_sq) * bessel

        integral = scipy.integrate.quad_vec(
            f=integrand,
            a=0,
            b=radial_basis.integration_radius,
            args=(radial, angular, distances),
        )[0]
        return prefactor * integral

    def _ri_gaussian_density_derivative(
        self,
        radial: int,
        angular: int,
        distances: np.ndarray,
    ) -> np.ndarray:
        # This code is derived in
        # https://metatensor.github.io/featomic/latest/devdoc/explanations/radial-integral.html,
        # and follows the same naming convention as the documentation.
        sigma_sq = self.density.width**2

        prefactor = (
            (4 * np.pi)
            / (np.pi * sigma_sq) ** (3 / 4)
            * np.exp(-0.5 * distances**2 / sigma_sq)
        )

        radial_basis = self.basis.radial_basis(angular)

        def integrand(r: float, n: int, ell: int, rij: np.array) -> np.ndarray:
            Rnl = radial_basis.compute_primitive(r, n, derivative=False)

            z = r * rij / sigma_sq
            one_over_z = np.divide(1.0, z, out=np.zeros_like(z), where=z != 0)

            # using recurrence relation from https://dlmf.nist.gov/10.51#E5 for the
            # derivative
            bessel_ell = _spherical_bessel_scaled(ell, z)
            bessel_ell_p1 = _spherical_bessel_scaled(ell + 1, z)
            bessel_grad = bessel_ell_p1 + ell * one_over_z * bessel_ell

            # The formula above is wrong for z=0, so let's replace it manually
            if ell == 1:
                bessel_grad = np.where(z == 0, 1.0 / 3.0, bessel_grad)

            return r**3 * Rnl * np.exp(z - 0.5 * r**2 / sigma_sq) * bessel_grad

        radial_integral = self._ri_gaussian_density(radial, angular, distances)
        grad_integral = scipy.integrate.quad_vec(
            f=integrand,
            a=0,
            b=radial_basis.integration_radius,
            args=(radial, angular, distances),
        )[0]
        return (prefactor * grad_integral - distances * radial_integral) / sigma_sq

    def _ri_custom_density(
        self,
        radial: int,
        angular: int,
        distances: np.ndarray,
        derivative: bool,
    ) -> np.ndarray:
        # This code is derived in
        # https://metatensor.github.io/featomic/latest/devdoc/explanations/radial-integral.html,
        # and follows the same naming convention as the documentation.

        P_ell = scipy.special.legendre(angular)
        radial_basis = self.basis.radial_basis(angular)

        if derivative:

            def integrand(u: float, r: float, n: int, ell: int, rij: float) -> float:
                arg = np.sqrt(r**2 + rij**2 - 2 * r * rij * u)

                Rnl = radial_basis.compute_primitive(r, n, derivative=False)
                density_grad = self.density.compute(arg, derivative=True)

                return r**2 * Rnl * P_ell(u) * (rij - u * r) * density_grad / arg

        else:

            def integrand(u: float, r: float, n: int, ell: int, rij: float) -> float:
                arg = np.sqrt(r**2 + rij**2 - 2 * r * rij * u)

                Rnl = radial_basis.compute_primitive(r, n, derivative=False)
                density = self.density.compute(arg, derivative=False)

                return r**2 * Rnl * P_ell(u) * density

        radial_integral = np.zeros(len(distances))

        for i, rij in enumerate(distances):
            radial_integral[i], _ = scipy.integrate.dblquad(
                func=integrand,
                # integration bounds for `r``
                a=0,
                b=radial_basis.integration_radius,
                # integration bounds for `u`
                gfun=-1,
                hfun=1,
                args=(radial, angular, rij),
            )

        return 2 * np.pi * radial_integral


class LodeSpliner:
    r"""Compute an explicit spline of the radial integral for LODE/k-space calculators.

    This allows a great deal of customization in the radial basis function used (any
    child class of :py:class:`RadialBasis`). This way, users can define custom basis,
    and use them with the LODE calculators. For more information about the radial
    integral, you can refer to :ref:`this document <radial-integral>`.

    This class should be used only in combination with k-space/LODE calculators like
    :class:`featomic.LodeSphericalExpansion`. For real space spherical expansions you
    should use :class:`SoapSpliner`.

    Example
    -------

    First let's define the hyper parameters for the LODE spherical expansions. It is
    important to note that only class-based hyper parameters are supported (in
    opposition to ``dict`` based hyper parameters).

    >>> import featomic

    >>> density = featomic.density.SmearedPowerLaw(smearing=1.0, exponent=1)
    >>> basis = featomic.basis.TensorProduct(
    ...     max_angular=4,
    ...     radial=featomic.basis.Gto(max_radial=5, radius=2),
    ... )

    From here we can initialize the spliner instance

    >>> spliner = LodeSpliner(density=density, basis=basis)

    You can then give the result of :meth:`LodeSpliner.get_hypers()` directly the
    calculator:

    >>> calculator = featomic.LodeSphericalExpansion(**spliner.get_hypers())

    .. seealso::
        :class:`SoapSpliner` for a spliner class that works with
        :class:`featomic.SphericalExpansion`
    """

    def __init__(
        self,
        density: AtomicDensity,
        basis: ExpansionBasis,
        k_cutoff: Optional[float] = None,
        n_spline_points: Optional[int] = None,
    ):
        """
        :param density: atomic density that should be expanded for all neighbors in the
            local atomic environment
        :param basis: basis function to use to expand the neighbors' atomic density.
            Currently only :py:class:`TensorProduct` expansion basis are supported.
        :param k_cutoff: Spherical cutoff in reciprocal space.
        :param n_spline_points: number of spline points to use. If ``None``, points will
            be added to the spline until the accuracy is at least the requested
            ``basis.spline_accuracy``.
        """
        if not HAS_SCIPY:
            raise ImportError("LodeSpliner class requires scipy")

        if not isinstance(density, SmearedPowerLaw):
            raise TypeError(
                "only `SmearedPowerLaw` `density` is supported by LODE, "
                f"got {type(density)}"
            )

        if not isinstance(basis, ExpansionBasis):
            raise TypeError(
                f"`basis` should be an `ExpansionBasis object, got {type(basis)}"
            )

        if k_cutoff is None:
            self.k_cutoff = 1.2 * np.pi / density.smearing
        else:
            self.k_cutoff = float(k_cutoff)

        self.density = density
        self.basis = basis

        self.n_spline_points = n_spline_points

    def get_hypers(self):
        """Get the LODE hyper-parameters for the splined basis and density."""
        # Contrary to the SOAP spliner, we don't transform the density to a delta
        # density, since the splines are applied directly to k-vectors (and not to the
        # density).
        hypers = {
            "k_cutoff": self.k_cutoff,
            "density": self.density,
        }

        def generic_values_fn(positions, radial_size, angular, orthonormalization):
            integrals = np.vstack(
                [
                    self._radial_integral(radial, angular, positions, derivative=False)
                    for radial in range(radial_size)
                ]
            )
            return (orthonormalization @ integrals).T

        def generic_derivatives_fn(positions, radial_size, angular, orthonormalization):
            integrals = np.vstack(
                [
                    self._radial_integral(radial, angular, positions, derivative=True)
                    for radial in range(radial_size)
                ]
            )
            return (orthonormalization @ integrals).T

        # We still transform whatever basis the user provided to an "explicit" basis,
        # since the radial integral will be different for different angular channels.
        by_angular = {}
        for angular in self.basis.angular_channels():
            radial_basis = self.basis.radial_basis(angular)
            orthonormalization = radial_basis._get_orthonormalization_matrix()

            values_fn = functools.partial(
                generic_values_fn,
                radial_size=radial_basis.size,
                angular=angular,
                orthonormalization=orthonormalization,
            )
            derivatives_fn = functools.partial(
                generic_derivatives_fn,
                radial_size=radial_basis.size,
                angular=angular,
                orthonormalization=orthonormalization,
            )

            if self.n_spline_points is not None:
                positions = np.linspace(0, self.k_cutoff, self.n_spline_points)
                spline = Spline(0, self.k_cutoff)
                spline.add_points(
                    positions=positions,
                    values=values_fn(positions),
                    derivatives=derivatives_fn(positions),
                )
            else:
                spline = Spline.with_accuracy(
                    start=0,
                    stop=self.k_cutoff,
                    values_fn=values_fn,
                    derivatives_fn=derivatives_fn,
                    accuracy=(
                        1e-8
                        if self.basis.spline_accuracy is None
                        else self.basis.spline_accuracy
                    ),
                )

            if angular == 0:
                center_contribution = self._center_contribution()
            else:
                center_contribution = None

            by_angular[angular] = SplinedRadialBasis(
                spline=spline,
                max_radial=radial_basis.max_radial,
                radius=self.k_cutoff,
                lode_center_contribution=center_contribution,
            )

        hypers["basis"] = Explicit(by_angular=by_angular, spline_accuracy=None)
        return hypers

    def _radial_integral(
        self,
        radial: int,
        angular: int,
        distances: np.ndarray,
        derivative: bool,
    ) -> np.ndarray:
        radial_basis = self.basis.radial_basis(angular)
        if derivative:

            def integrand(r: float, n: int, ell: int, rij: np.ndarray) -> np.ndarray:
                Rnl = radial_basis.compute_primitive(r, n, derivative=False)
                spherical_jn = scipy.special.spherical_jn(ell, r * rij, derivative=True)
                return r**3 * Rnl * spherical_jn

        else:

            def integrand(r: float, n: int, ell: int, rij: np.ndarray) -> np.ndarray:
                Rnl = radial_basis.compute_primitive(r, n, derivative=False)
                spherical_jn = scipy.special.spherical_jn(
                    ell, r * rij, derivative=False
                )
                return r**2 * Rnl * spherical_jn

        return scipy.integrate.quad_vec(
            f=integrand,
            a=0,
            b=radial_basis.integration_radius,
            args=(radial, angular, distances),
        )[0]

    def _center_contribution(self) -> np.ndarray:
        assert isinstance(self.density, SmearedPowerLaw)

        radial_basis = self.basis.radial_basis(angular=0)

        def integrand(r: float, n: int) -> np.ndarray:
            return (
                r**2
                * radial_basis.compute_primitive(r, n, derivative=False)
                * self.density.compute(r, derivative=False)
            )

        integrals = []
        for radial in range(radial_basis.size):
            integrals.append(
                scipy.integrate.quad(
                    func=integrand,
                    a=0,
                    b=radial_basis.integration_radius,
                    args=(radial,),
                )[0]
            )

        return np.sqrt(4 * np.pi) * np.array(integrals)
