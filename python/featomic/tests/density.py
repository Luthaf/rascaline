import numpy as np
import pytest

from featomic.density import Gaussian, SmearedPowerLaw


pytest.importorskip("scipy")


@pytest.mark.parametrize(
    "density",
    [Gaussian(width=1.2), SmearedPowerLaw(smearing=1.2, exponent=1)],
)
def test_derivative(density):
    positions = np.linspace(0, 5, num=int(1e6))
    rho = density.compute(positions, derivative=False)
    analytical_grad = density.compute(positions, derivative=True)
    numerical_grad = np.gradient(rho, positions)

    np.testing.assert_allclose(numerical_grad, analytical_grad, atol=1e-6)
