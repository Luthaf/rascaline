import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from rascaline import LodeSphericalExpansion, SphericalExpansion
from rascaline.utils import LodeSpliner, RadialIntegralFromFunction, SoapSpliner
from rascaline.utils.atomic_density import DeltaDensity, GaussianDensity, LodeDensity
from rascaline.utils.radial_basis import GtoBasis

from ..test_systems import SystemForTests


pytest.importorskip("scipy")
from scipy.special import gamma, hyp1f1  # noqa


def sine(n: int, ell: int, positions: np.ndarray) -> np.ndarray:
    return np.sin(positions)


def cosine(n: int, ell: int, positions: np.ndarray) -> np.ndarray:
    return np.cos(positions)


@pytest.mark.parametrize("n_spline_points", [None, 1234])
def test_splines_with_n_spline_points(n_spline_points):
    spline_cutoff = 8.0

    spliner = RadialIntegralFromFunction(
        radial_integral=sine,
        max_radial=12,
        max_angular=9,
        spline_cutoff=spline_cutoff,
        radial_integral_derivative=cosine,
    )

    radial_integral = spliner.compute(n_spline_points=n_spline_points)[
        "TabulatedRadialIntegral"
    ]

    # check central contribution is not added
    with pytest.raises(KeyError):
        radial_integral["center_contribution"]

    spline_points = radial_integral["points"]

    # check that the first spline point is at 0
    assert spline_points[0]["position"] == 0.0

    # check that the last spline point is at the cutoff radius
    assert spline_points[-1]["position"] == 8.0

    # ensure correct length for values representation
    assert len(spline_points[52]["values"]["data"]) == (9 + 1) * 12

    # ensure correct length for derivatives representation
    assert len(spline_points[23]["derivatives"]["data"]) == (9 + 1) * 12

    # check values at r = 0.0
    assert np.allclose(
        np.array(spline_points[0]["values"]["data"]), np.zeros((9 + 1) * 12)
    )

    # check derivatives at r = 0.0
    assert np.allclose(
        np.array(spline_points[0]["derivatives"]["data"]), np.ones((9 + 1) * 12)
    )

    n_spline_points = len(spline_points)
    random_spline_point = 123
    random_x = random_spline_point * spline_cutoff / (n_spline_points - 1)

    # check value of a random spline point
    assert np.allclose(
        np.array(spline_points[random_spline_point]["values"]["data"]),
        np.sin(random_x) * np.ones((9 + 1) * 12),
    )


def test_splines_numerical_derivative():
    kwargs = {
        "radial_integral": sine,
        "max_radial": 12,
        "max_angular": 9,
        "spline_cutoff": 8.0,
    }

    spliner = RadialIntegralFromFunction(**kwargs, radial_integral_derivative=cosine)
    spliner_numerical = RadialIntegralFromFunction(**kwargs)

    spline_points = spliner.compute()["TabulatedRadialIntegral"]["points"]
    spline_points_numerical = spliner_numerical.compute()["TabulatedRadialIntegral"][
        "points"
    ]

    for s, s_num in zip(spline_points, spline_points_numerical):
        assert_equal(s["values"]["data"], s_num["values"]["data"])
        assert_allclose(
            s["derivatives"]["data"], s_num["derivatives"]["data"], rtol=1e-7
        )


def test_splines_numerical_derivative_error():
    kwargs = {
        "radial_integral": sine,
        "max_radial": 12,
        "max_angular": 9,
        "spline_cutoff": 1e-3,
    }

    match = "Numerically derivative of the radial integral can not be performed"
    with pytest.raises(ValueError, match=match):
        RadialIntegralFromFunction(**kwargs).compute()


def test_kspace_radial_integral():
    """Test against anayliycal integral with Gaussian densities and GTOs"""

    cutoff = 2
    max_radial = 6
    max_angular = 3
    atomic_gaussian_width = 1.0
    k_cutoff = 1.2 * np.pi / atomic_gaussian_width

    basis = GtoBasis(cutoff=cutoff, max_radial=max_radial)

    spliner = LodeSpliner(
        max_radial=max_radial,
        max_angular=max_angular,
        k_cutoff=k_cutoff,
        basis=basis,
        density=DeltaDensity(),  # density does not enter in a Kspace radial integral
        accuracy=1e-8,
    )

    Neval = 100
    kk = np.linspace(0, k_cutoff, Neval)

    sigma = np.ones(max_radial, dtype=float)
    for i in range(1, max_radial):
        sigma[i] = np.sqrt(i)
    sigma *= cutoff / max_radial

    factors = np.sqrt(np.pi) * np.ones((max_radial, max_angular + 1))

    coeffs_num = np.zeros([max_radial, max_angular + 1, Neval])
    coeffs_exact = np.zeros_like(coeffs_num)

    for ell in range(max_angular + 1):
        for n in range(max_radial):
            i1 = 0.5 * (3 + n + ell)
            i2 = 1.5 + ell
            factors[n, ell] *= (
                2 ** (0.5 * (n - ell - 1))
                * gamma(i1)
                / gamma(i2)
                * sigma[n] ** (2 * i1)
            )
            coeffs_exact[n, ell] = (
                factors[n, ell]
                * kk**ell
                * hyp1f1(i1, i2, -0.5 * (kk * sigma[n]) ** 2)
            )

            coeffs_num[n, ell] = spliner._radial_integral(n, ell, kk)

    assert_allclose(coeffs_num, coeffs_exact)


def test_rspace_delta():
    cutoff = 2
    max_radial = 6
    max_angular = 3

    basis = GtoBasis(cutoff=cutoff, max_radial=max_radial)
    density = DeltaDensity()

    spliner = SoapSpliner(
        max_radial=max_radial,
        max_angular=max_angular,
        cutoff=cutoff,
        basis=basis,
        density=density,
        accuracy=1e-8,
    )

    positions = np.linspace(0, cutoff)

    for ell in range(max_angular + 1):
        for n in range(max_radial):
            assert_equal(
                spliner._radial_integral(n, ell, positions),
                basis.compute(n, ell, positions),
            )
            assert_equal(
                spliner._radial_integral_derivative(n, ell, positions),
                basis.compute_derivative(n, ell, positions),
            )


def test_real_space_spliner():
    """Compare splined spherical expansion with GTOs and a Gaussian density to
    analytical implementation."""
    cutoff = 8.0
    max_radial = 12
    max_angular = 9
    atomic_gaussian_width = 1.2

    # We choose an accuracy that is larger then the default one (1e-8) to limit the time
    # consumption of the test.
    accuracy = 1e-4

    spliner = SoapSpliner(
        cutoff=cutoff,
        max_radial=max_radial,
        max_angular=max_angular,
        basis=GtoBasis(cutoff=cutoff, max_radial=max_radial),
        density=GaussianDensity(atomic_gaussian_width=atomic_gaussian_width),
        accuracy=accuracy,
    )

    hypers_spherical_expansion = {
        "cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "center_atom_weight": 1.0,
        "atomic_gaussian_width": atomic_gaussian_width,
        "cutoff_function": {"Step": {}},
    }

    analytic = SphericalExpansion(
        radial_basis={"Gto": {}}, **hypers_spherical_expansion
    ).compute(SystemForTests())
    splined = SphericalExpansion(
        radial_basis=spliner.compute(), **hypers_spherical_expansion
    ).compute(SystemForTests())

    for key, block_analytic in analytic.items():
        block_splined = splined.block(key)
        assert_allclose(
            block_splined.values, block_analytic.values, rtol=5e-4, atol=2e-5
        )


@pytest.mark.parametrize("center_atom_weight", [1.0, 0.0])
@pytest.mark.parametrize("potential_exponent", [0, 1])
def test_fourier_space_spliner(center_atom_weight, potential_exponent):
    """Compare splined LODE spherical expansion with GTOs and a Gaussian density to
    analytical implementation."""

    cutoff = 2
    max_radial = 6
    max_angular = 4
    atomic_gaussian_width = 0.8
    k_cutoff = 1.2 * np.pi / atomic_gaussian_width

    spliner = LodeSpliner(
        k_cutoff=k_cutoff,
        max_radial=max_radial,
        max_angular=max_angular,
        basis=GtoBasis(cutoff=cutoff, max_radial=max_radial),
        density=LodeDensity(
            atomic_gaussian_width=atomic_gaussian_width,
            potential_exponent=potential_exponent,
        ),
    )

    hypers_spherical_expansion = {
        "cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "center_atom_weight": center_atom_weight,
        "atomic_gaussian_width": atomic_gaussian_width,
        "potential_exponent": potential_exponent,
    }

    analytic = LodeSphericalExpansion(
        radial_basis={"Gto": {}}, **hypers_spherical_expansion
    ).compute(SystemForTests())
    splined = LodeSphericalExpansion(
        radial_basis=spliner.compute(), **hypers_spherical_expansion
    ).compute(SystemForTests())

    for key, block_analytic in analytic.items():
        block_splined = splined.block(key)
        assert_allclose(block_splined.values, block_analytic.values, atol=1e-14)


def test_center_contribution_gto_gaussian():
    cutoff = 2.0
    max_radial = 6
    max_angular = 4
    atomic_gaussian_width = 0.8
    k_cutoff = 1.2 * np.pi / atomic_gaussian_width

    # Numerical evaluation of center contributions
    spliner = LodeSpliner(
        k_cutoff=k_cutoff,
        max_radial=max_radial,
        max_angular=max_angular,
        basis=GtoBasis(cutoff=cutoff, max_radial=max_radial),
        density=GaussianDensity(atomic_gaussian_width=atomic_gaussian_width),
    )

    # Analytical evaluation of center contributions
    center_contr_analytical = np.zeros((max_radial))

    normalization = 1.0 / (np.pi * atomic_gaussian_width**2) ** (3 / 4)
    sigma_radial = np.ones(max_radial, dtype=float)

    for n in range(1, max_radial):
        sigma_radial[n] = np.sqrt(n)
    sigma_radial *= cutoff / max_radial

    for n in range(max_radial):
        sigmatemp_sq = 1.0 / (
            1.0 / atomic_gaussian_width**2 + 1.0 / sigma_radial[n] ** 2
        )
        neff = 0.5 * (3 + n)
        center_contr_analytical[n] = (2 * sigmatemp_sq) ** neff * gamma(neff)

    center_contr_analytical *= normalization * 2 * np.pi / np.sqrt(4 * np.pi)

    assert_allclose(spliner._center_contribution, center_contr_analytical, rtol=1e-14)
