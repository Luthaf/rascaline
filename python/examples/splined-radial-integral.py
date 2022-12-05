"""
Splined radial integrals
========================

.. start-body
"""

# %%
#
# This script illustrates how to generate splined radial basis
# functions/integrals, using a "rectangular" Laplacian eigenstate (LE,
# https://doi.org/10.1063/5.0124363) basis as the example, i.e, a LE basis
# truncated with l_max, n_max hyper-parameters.


import ase
import numpy as np
import scipy as sp
import scipy.optimize
from scipy.special import jv
from scipy.special import spherical_jn as j_l
from scipy.integrate import quadrature

from rascaline import SphericalExpansion, generate_splines


# %%
#
# Set some hyper-parameters:
max_angular = 6
max_radial = 8
cutoff = 5.0  # This is also the radius of the LE sphere

# %%
#
# Spherical Bessel zeros (from the scipy cookbook)


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r)


def Jn_zeros(n, nt):
    zeros_j = np.zeros((n + 1, nt), dtype=np.float64)
    zeros_j[0] = np.arange(1, nt + 1) * np.pi
    points = np.arange(1, nt + n + 1) * np.pi
    roots = np.zeros(nt + n, dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(nt + n - i):
            roots[j] = scipy.optimize.brentq(Jn, points[j], points[j + 1], (i,))
        points = roots
        zeros_j[i][:nt] = roots[:nt]
    return zeros_j


z_ln = Jn_zeros(max_angular, max_radial)
z_nl = z_ln.T


# %%
#
# Define the radial basis functions:


def R_nl(n, el, r):
    # Un-normalized LE radial basis functions
    return j_l(el, z_nl[n, el] * r / cutoff)


def N_nl(n, el):
    # Normalization factor for LE basis functions, excluding the a**(-1.5) factor
    def function_to_integrate_to_get_normalization_factor(x):
        return j_l(el, x) ** 2 * x**2

    integral, _ = sp.integrate.quadrature(
        function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, el]
    )
    return (1.0 / z_nl[n, el] ** 3 * integral) ** (-0.5)


def laplacian_eigenstate_basis(n, el, r):
    R = np.zeros_like(r)
    for i in range(r.shape[0]):
        R[i] = R_nl(n, el, r[i])
    return N_nl(n, el) * R * cutoff ** (-1.5)


# %%
#
# Quick normalization check:
normalization_check_integral, _ = sp.integrate.quadrature(
    lambda x: laplacian_eigenstate_basis(1, 1, x) ** 2 * x**2,
    0.0,
    cutoff,
)
print(f"Normalization check (needs to be close to 1): {normalization_check_integral}")


# %%
#
# Now the derivatives (by finite differences):
def laplacian_eigenstate_basis_derivative(n, el, r):
    delta = 1e-6
    all_derivatives_except_at_zero = (
        laplacian_eigenstate_basis(n, el, r[1:] + delta)
        - laplacian_eigenstate_basis(n, el, r[1:] - delta)
    ) / (2.0 * delta)
    derivative_at_zero = (
        laplacian_eigenstate_basis(n, el, np.array([delta / 10.0]))
        - laplacian_eigenstate_basis(n, el, np.array([0.0]))
    ) / (delta / 10.0)
    return np.concatenate([derivative_at_zero, all_derivatives_except_at_zero])


# %%
#
# The radial basis functions and their derivatives can be input into a spline
# generator. This will output the positions of the spline points, the
# values of the basis functions evaluated at the spline points, and the
# corresponding derivatives.
spline_points = generate_splines(
    laplacian_eigenstate_basis,
    laplacian_eigenstate_basis_derivative,
    max_radial,
    max_angular,
    cutoff,
    requested_accuracy=1e-5,
)

# %%
#
# The, we feed the splines to the Rust calculator:
# (IMPORTANT: "atomic_gaussian_width" will be ignored)

hypers_spherical_expansion = {
    "cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "TabulatedRadialIntegral": {
            "spline_points": spline_points,
        }
    },
    "atomic_gaussian_width": 1.0,  # ignored
    "cutoff_function": {"Step": {}},
}
calculator = SphericalExpansion(**hypers_spherical_expansion)

# %%
#
# Create dummy structures to test if the calculator outputs correct radial functions:


def get_dummy_structures(r_array):
    dummy_structures = []
    for r in r_array:
        dummy_structures.append(ase.Atoms("CH", positions=[(0, 0, 0), (0, 0, r)]))
    return dummy_structures


r = np.linspace(0.1, 4.9, 20)
structures = get_dummy_structures(r)
spherical_expansion_coefficients = calculator.compute(structures)

# Extract l = 0 features and check that the n = 2 predictions are the same:
block_C_l0 = spherical_expansion_coefficients.block(
    species_center=6, spherical_harmonics_l=0, species_neighbor=1
)
block_C_l0_n2 = block_C_l0.values[:, :, 2].flatten()
spherical_harmonics_0 = 1.0 / np.sqrt(4.0 * np.pi)

# radial function = feature / spherical harmonics function
rascaline_output_radial_function = block_C_l0_n2 / spherical_harmonics_0

assert np.allclose(
    rascaline_output_radial_function,
    laplacian_eigenstate_basis(2, 0, r),
    atol=1e-5,
)
print("Assertion passed successfully!")


# %%
#
# .. end-body
