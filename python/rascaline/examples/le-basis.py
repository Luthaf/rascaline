"""
.. _example-le-basis:

Laplacian eigenstate basis
==========================

.. start-body
"""

# %%
#
# This script illustrates how to generate a spherical expansion using
# the Laplacian eigenstate basis (https://doi.org/10.1063/5.0124363),
# both using truncation with l_max, n_max hyper-parameters and with an
# eigenvalue threshold.

import ase.io
import numpy as np
import scipy.optimize
from scipy.special import jv

from metatensor import Labels, TensorBlock, TensorMap
import rascaline

try:
    import torch
    torch.set_default_dtype(torch.float64)
    import torch_spex
    HAS_TORCH_SPEX = True
except ImportError:
    HAS_TORCH_SPEX = False

# %%
#
# First using a truncation with l_max, n_max hyper-parameters (easy):
cutoff = 4.4
l_max = 6
n_max = 8

structures = ase.io.read("dataset.xyz", ":10")

spliner = rascaline.utils.SoapSpliner(
    cutoff=cutoff,
    max_radial=n_max,
    max_angular=l_max,
    basis=rascaline.utils.SphericalBesselBasis(cutoff=cutoff, max_radial=n_max, max_angular=l_max),
    density=rascaline.utils.DeltaDensity(),
    accuracy=1e-8,
)

calculator = rascaline.SphericalExpansion(
    cutoff=cutoff,
    max_radial=n_max,
    max_angular=l_max,
    center_atom_weight=1.0,
    radial_basis=spliner.compute(),
    atomic_gaussian_width=-1.0,  # will not be used due to the delta density above
    cutoff_function={"ShiftedCosine": {"width": 0.5}},
)

spherical_expansion = calculator.compute(structures)

# %%
#
# Now we will calculate the same basis with an eigenvalue threshold
# (more involved), which affords a better accuracy/cost ratio, using
# property selection.

E_max = 400  # eigenvalue threshold

# %%
#
# Spherical Bessel zeros (from the scipy cookbook) and
# Laplacian eigenvalues

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

l_max_large = 50  # just used to get the eigenvalues
n_max_large = 50  # just used to get the eigenvalues
z_ln = Jn_zeros(l_max_large, n_max_large)

E_ln = z_ln**2  # proportional to the Laplacian eigenvalues, which would be z_ln**2 / cutoff**2

# %%
#
# Determine the l_max, n_max parameters that will certainly
# contain all the desired terms

n_max_l = []
for l in range(l_max_large+1):
    # for each l, calculate how many basis functions
    n_max_l.append(
        len(np.where(E_ln[l]<E_max)[0])
    )
    if n_max_l[-1] == 0:
        # all eigenvalues for this l are over the threshold
        n_max_l.pop()
        l_max = l-1
        break

n_max = n_max_l[0]

# %%
#
# Comparing this l-dependent threshold with the one based on l_max
# and n_max, we see that the eigenvalue thresholding leads to a gradual
# decrease of n_max for high l values:

import matplotlib.pyplot as plt
plt.plot(list(range(15+1)), [4 if l<11 else 0 for l in range(15+1)], ".", label="l_max, n_max threshold")
plt.plot(list(range(l_max+1)), n_max_l, ".", label="Eigenvalue threshold")
plt.xlabel("l")
plt.ylabel("n_max")
plt.ylim(-0.5, n_max_l[0]+0.5)
plt.legend()
plt.show()

# %%
#
# Set up a TensorMap for property selection

all_species = list(np.unique(np.concatenate([structure.numbers for structure in structures])))  # extract all the species from the small dataset

keys = []
blocks = []
for center_species in all_species:
    for neighbor_species in all_species:
        for l in range(l_max+1):
            keys.append([l, center_species, neighbor_species])
            blocks.append(
                TensorBlock(
                    values=np.empty((1, n_max_l[l])),
                    samples=Labels.single(),
                    components=[],
                    properties=Labels(
                        names=["n"],
                        values=np.arange(n_max_l[l]).reshape(n_max_l[l], 1)
                    )
                )
            )

selected_properties = TensorMap(
    keys=Labels(
        names=["spherical_harmonics_l", "species_center", "species_neighbor"],
        values=np.array(keys)
    ),
    blocks=blocks
)

# %%
#
# Build a calculator and calculate the spherical expansion

spliner = rascaline.utils.SoapSpliner(
    cutoff=cutoff,
    max_radial=n_max,
    max_angular=l_max,
    basis=rascaline.utils.SphericalBesselBasis(cutoff=cutoff, max_radial=n_max, max_angular=l_max),
    density=rascaline.utils.DeltaDensity(),
    accuracy=1e-8,
)

calculator = rascaline.SphericalExpansion(
    cutoff=cutoff,
    max_radial=n_max,
    max_angular=l_max,
    center_atom_weight=1.0,
    radial_basis=spliner.compute(),
    atomic_gaussian_width=-1.0,  # will not be used due to the delta density above
    cutoff_function={"ShiftedCosine": {"width": 0.5}},
)

spherical_expansion = calculator.compute(
    structures,
    selected_properties=selected_properties  # we tell the calculator to only compute the selected properties
)
