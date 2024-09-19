# import numpy as np
# import pytest
# from numpy.testing import assert_allclose

# from rascaline.utils import GaussianDensity, LodeDensity


# pytest.importorskip("scipy")


# @pytest.mark.parametrize(
#     "density",
#     [
#         GaussianDensity(atomic_gaussian_width=1.2),
#         LodeDensity(atomic_gaussian_width=1.2, potential_exponent=1),
#     ],
# )
# def test_derivative(density):
#     positions = np.linspace(0, 5, num=int(1e6))
#     dens = density.compute(positions)
#     grad_ana = density.compute_derivative(positions)
#     grad_numerical = np.gradient(dens, positions)

#     assert_allclose(grad_numerical, grad_ana, atol=1e-6)
