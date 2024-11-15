import numpy as np  # noqa: I001
import pytest

from featomic.utils import hypers_to_json
from featomic.basis import (
    Gto,
    LaplacianEigenstate,
    Monomials,
    RadialBasis,
    SphericalBessel,
)


pytest.importorskip("scipy")


class RtoNRadialBasis(RadialBasis):
    def __init__(self, max_radial, radius):
        super().__init__(max_radial=max_radial, radius=radius)

    def compute_primitive(
        self, positions: np.ndarray, n: int, *, derivative=False
    ) -> np.ndarray:
        assert not derivative
        return positions**n


def test_radial_basis_gram():
    """Test that quad integration of the gram matrix is the same as an analytical."""

    radius = 1.5
    max_radial = 4

    test_basis = RtoNRadialBasis(max_radial=max_radial, radius=radius)
    numerical_gram = test_basis._get_gram_matrix()

    analytical_gram = np.zeros_like(numerical_gram)
    for n1 in range(max_radial + 1):
        for n2 in range(max_radial + 1):
            exponent = 3 + n1 + n2
            analytical_gram[n1, n2] = radius**exponent / exponent

    np.testing.assert_allclose(numerical_gram, analytical_gram, atol=1e-12)


def test_radial_basis_orthornormalization():
    radius = 1.5
    max_radial = 4

    test_basis = RtoNRadialBasis(max_radial=max_radial, radius=radius)

    gram = test_basis._get_gram_matrix()
    ortho = test_basis._get_orthonormalization_matrix()

    np.testing.assert_allclose(
        ortho @ gram @ ortho.T, np.eye(max_radial + 1), atol=1e-9
    )


# Define a helper class that used the numerical derivatives from `RadialBasis`
# instead of the explicitly implemented analytical ones in the child classes.
class NumericalRadialBasis(RadialBasis):
    def __init__(self, basis):
        super().__init__(max_radial=basis.max_radial, radius=basis.radius)
        self.basis = basis

    def compute_primitive(
        self, n: int, positions: np.ndarray, *, derivative: bool
    ) -> np.ndarray:
        if derivative:
            return self.basis.finite_differences_derivative(n, positions)
        else:
            return self.basis.compute_primitive(n, positions, derivative=False)


@pytest.mark.parametrize(
    "analytical_basis",
    [
        Gto(radius=4, max_radial=6),
        Monomials(radius=4, max_radial=6, angular_channel=3),
        SphericalBessel(radius=4, max_radial=6, angular_channel=3),
    ],
)
def test_derivative(analytical_basis):
    """Finite difference test for testing the derivative of a radial basis"""

    numerical_basis = NumericalRadialBasis(analytical_basis)

    positions = np.linspace(0.5, analytical_basis.radius)
    for n in range(analytical_basis.size):
        np.testing.assert_allclose(
            numerical_basis.compute_primitive(positions, n, derivative=True),
            analytical_basis.compute_primitive(positions, n, derivative=True),
            atol=1e-6,
        )


def test_le_basis():
    basis = LaplacianEigenstate(max_radial=4, radius=3.4)
    assert basis.max_angular == 10
    assert basis.angular_channels() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for angular in basis.angular_channels():
        radial = basis.radial_basis(angular)
        assert radial.max_radial <= 4

    basis = LaplacianEigenstate(max_radial=4, max_angular=4, radius=3.4)
    assert basis.max_angular == 4
    assert basis.angular_channels() == [0, 1, 2, 3, 4]

    message = (
        "This radial basis function \\(SphericalBessel\\) does not have matching "
        "hyper parameters in the native calculators. It should be used through one of "
        "the spliner class instead of directly"
    )
    with pytest.raises(NotImplementedError, match=message):
        hypers_to_json(basis.get_hypers())
