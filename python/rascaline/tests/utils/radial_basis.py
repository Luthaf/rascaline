from typing import Union

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rascaline.utils import (
    GtoBasis,
    MonomialBasis,
    RadialBasisBase,
    SphericalBesselBasis,
)


pytest.importorskip("scipy")


class RtoNRadialBasis(RadialBasisBase):
    def compute(
        self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return integrand_positions**n


def test_radial_basis_gram():
    """Test that quad integration of the gram matrix is the same as an analytical."""

    integration_radius = 1
    max_radial = 4
    max_angular = 2

    test_basis = RtoNRadialBasis(integration_radius=integration_radius)

    numerical_gram = test_basis.compute_gram_matrix(max_radial, max_angular)
    analytical_gram = np.zeros_like(numerical_gram)

    for ell in range(max_angular + 1):
        for n1 in range(max_radial):
            for n2 in range(max_radial):
                exp = 3 + n1 + n2
                analytical_gram[ell, n1, n2] = integration_radius**exp / exp

    assert_allclose(numerical_gram, analytical_gram)


def test_radial_basis_orthornormalization():
    integration_radius = 1
    max_radial = 4
    max_angular = 2

    test_basis = RtoNRadialBasis(integration_radius=integration_radius)

    gram = test_basis.compute_gram_matrix(max_radial, max_angular)
    ortho = test_basis.compute_orthonormalization_matrix(max_radial, max_angular)

    for ell in range(max_angular):
        eye = ortho[ell] @ gram[ell] @ ortho[ell].T
        assert_allclose(eye, np.eye(max_radial, max_radial), atol=1e-11)


@pytest.mark.parametrize(
    "analytical_basis",
    [
        GtoBasis(cutoff=4, max_radial=6),
        MonomialBasis(cutoff=4),
        SphericalBesselBasis(cutoff=4, max_radial=6, max_angular=4),
    ],
)
def test_derivative(analytical_basis: RadialBasisBase):
    """Finite difference test for testing the derivative of a radial basis"""

    class NumericalRadialBasis(RadialBasisBase):
        def compute(
            self, n: int, ell: int, integrand_positions: Union[float, np.ndarray]
        ) -> Union[float, np.ndarray]:
            return analytical_basis.compute(n, ell, integrand_positions)

    numerical_basis = NumericalRadialBasis(integration_radius=np.inf)

    cutoff = 4
    max_radial = 6
    max_angular = 4
    positions = np.linspace(2, cutoff)

    for n in range(max_radial):
        for ell in range(max_angular):
            assert_allclose(
                numerical_basis.compute_derivative(n, ell, positions),
                analytical_basis.compute_derivative(n, ell, positions),
                atol=1e-9,
            )
