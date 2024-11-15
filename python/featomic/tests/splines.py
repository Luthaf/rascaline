import numpy as np
import pytest

import featomic
from featomic import LodeSphericalExpansion, SphericalExpansion

from .test_systems import SystemForTests


pytest.importorskip("scipy")
from scipy.special import gamma, hyp1f1  # noqa


def test_soap_spliner():
    """Compare splined spherical expansion with GTOs and a Gaussian density to
    analytical implementation."""
    cutoff = 8.0

    hypers = {
        "cutoff": featomic.cutoff.Cutoff(
            radius=cutoff,
            smoothing=featomic.cutoff.Step(),
        ),
        "density": featomic.density.Gaussian(width=0.6),
        "basis": featomic.basis.TensorProduct(
            max_angular=6,
            radial=featomic.basis.Gto(max_radial=6, radius=cutoff),
            # We choose an accuracy that is lower then the default one (1e-8)
            # to limit the time taken by this test.
            spline_accuracy=1e-3,
        ),
    }

    spliner = featomic.splines.SoapSpliner(**hypers)

    analytic = SphericalExpansion(**hypers).compute(SystemForTests())
    splined = SphericalExpansion(**spliner.get_hypers()).compute(SystemForTests())

    for key, block_analytic in analytic.items():
        block_splined = splined.block(key)
        np.testing.assert_allclose(
            block_splined.values, block_analytic.values, rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("exponent", [0, 1, 4])
def test_lode_spliner(exponent):
    """Compare splined LODE spherical expansion with GTOs and a Gaussian density to
    analytical implementation."""

    hypers = {
        "density": featomic.density.SmearedPowerLaw(smearing=0.8, exponent=exponent),
        "basis": featomic.basis.TensorProduct(
            max_angular=4,
            radial=featomic.basis.Gto(max_radial=4, radius=2),
        ),
    }

    spliner = featomic.splines.LodeSpliner(**hypers)

    analytic = LodeSphericalExpansion(**hypers).compute(SystemForTests())
    splined = LodeSphericalExpansion(**spliner.get_hypers()).compute(SystemForTests())

    for key, block_analytic in analytic.items():
        block_splined = splined.block(key)
        np.testing.assert_allclose(
            block_splined.values, block_analytic.values, rtol=1e-6, atol=1e-6
        )


def test_lode_center_contribution():
    """Compare the numerical center_contribution calculation with the analytical
    formula"""

    def center_contribution_analytical(radial_size, smearing, gto_radius):
        result = np.zeros((radial_size))

        normalization = 1.0 / (np.pi * smearing**2) ** (3 / 4)
        sigma_radial = np.ones(radial_size, dtype=float)

        for n in range(1, radial_size):
            sigma_radial[n] = np.sqrt(n)
        sigma_radial *= gto_radius / radial_size

        for n in range(radial_size):
            tmp = 1.0 / (1.0 / smearing**2 + 1.0 / sigma_radial[n] ** 2)
            n_eff = 0.5 * (3 + n)
            result[n] = (2 * tmp) ** n_eff * gamma(n_eff)

        result *= normalization * 2 * np.pi / np.sqrt(4 * np.pi)
        return result

    gto_radius = 2.0
    max_radial = 5
    smearing = 1.2
    hypers = {
        "density": featomic.density.SmearedPowerLaw(smearing=smearing, exponent=0),
        "basis": featomic.basis.TensorProduct(
            max_angular=4,
            radial=featomic.basis.Gto(max_radial=max_radial, radius=gto_radius),
            # We choose an accuracy that is lower then the default one (1e-8)
            # to limit the time taken by this test.
            spline_accuracy=1e-3,
        ),
    }

    spliner = featomic.splines.LodeSpliner(**hypers)

    np.testing.assert_allclose(
        # Numerical evaluation of center contributions
        spliner._center_contribution(),
        # Analytical evaluation of center contributions
        center_contribution_analytical(max_radial + 1, smearing, gto_radius),
        rtol=1e-5,
    )


def test_custom_radial_integral():
    hypers = {
        "cutoff": featomic.cutoff.Cutoff(
            radius=0.2,
            smoothing=featomic.cutoff.Step(),
        ),
        "density": featomic.density.Gaussian(width=1.2),
        "basis": featomic.basis.TensorProduct(
            max_angular=2,
            radial=featomic.basis.Gto(max_radial=3, radius=2.0),
        ),
    }

    n_spline_points = 3
    spliner_gaussian = featomic.splines.SoapSpliner(
        **hypers, n_spline_points=n_spline_points
    )

    # Create a custom density that has not type "Gaussian" to trigger full
    # numerical evaluation of the radial integral.
    class NotAGaussian(featomic.density.AtomicDensity):
        def __init__(self, *, width: float):
            super().__init__(center_atom_weight=1.0, scaling=None)
            self.width = float(width)

        def compute(self, positions: np.ndarray, *, derivative: bool) -> np.ndarray:
            width_sq = self.width**2
            x = positions**2 / (2 * width_sq)

            density = np.exp(-x) / (np.pi * width_sq) ** (3 / 4)

            if derivative:
                density *= -positions / width_sq

            return density

    hypers["density"] = NotAGaussian(width=1.2)

    spliner_custom = featomic.splines.SoapSpliner(
        **hypers, n_spline_points=n_spline_points
    )

    splines_gaussian = spliner_gaussian.get_hypers()["basis"].by_angular
    splines_custom = spliner_custom.get_hypers()["basis"].by_angular

    for ell in [0, 1, 2]:
        spline_gaussian = splines_gaussian[ell].spline
        spline_custom = splines_custom[ell].spline

        assert np.all(spline_gaussian.positions == spline_custom.positions)

        np.testing.assert_allclose(
            spline_gaussian.values,
            spline_custom.values,
            atol=1e-9,
        )

        np.testing.assert_allclose(
            spline_gaussian.derivatives,
            spline_custom.derivatives,
            atol=1e-9,
        )
