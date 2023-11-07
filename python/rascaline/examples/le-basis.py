"""
LE basis
========

.. start-body
"""

# %%
#
# This script illustrates how to generate a spherical expansion using
# the Laplacian eigenstate basis (https://doi.org/10.1063/5.0124363),
# both using truncation with l_max, n_max hyper-parameters and with an
# eigenvalue threshold.

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

import rascaline


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
    basis=rascaline.utils.SphericalBesselBasis(
        cutoff=cutoff, max_radial=n_max, max_angular=l_max
    ),
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
# Spherical Bessel zeros and Laplacian eigenvalues

l_max_large = 50  # just used to get the eigenvalues
n_max_large = 50  # just used to get the eigenvalues

# spherical Bessel zeros:
z_ln = rascaline.utils.SphericalBesselBasis.compute_zeros(l_max_large, n_max_large)

E_ln = (
    z_ln**2
)  # proportional to the Laplacian eigenvalues, which would be z_ln**2 / cutoff**2

# %%
#
# Determine the l_max, n_max parameters that will certainly
# contain all the desired terms

n_max_l = []
for ell in range(l_max_large + 1):
    # for each l, calculate how many basis functions
    n_max_l.append(len(np.where(E_ln[ell] < E_max)[0]))
    if n_max_l[-1] == 0:
        # all eigenvalues for this l are over the threshold
        n_max_l.pop()
        l_max = ell - 1
        break

n_max = n_max_l[0]

# %%
#
# Comparing this l-dependent threshold with the one based on l_max
# and n_max, we see that the eigenvalue thresholding leads to a gradual
# decrease of n_max for high l values:

plt.plot(
    list(range(15 + 1)),
    [4 if ell < 11 else 0 for ell in range(15 + 1)],
    ".",
    label="l_max, n_max threshold",
)
plt.plot(list(range(l_max + 1)), n_max_l, ".", label="Eigenvalue threshold")
plt.xlabel("l")
plt.ylabel("n_max")
plt.ylim(-0.5, n_max_l[0] + 0.5)
plt.legend()
plt.show()

# %%
#
# Set up a TensorMap for property selection

all_species = list(
    np.unique(np.concatenate([structure.numbers for structure in structures]))
)  # extract all the species from the small dataset

keys = []
blocks = []
for center_species in all_species:
    for neighbor_species in all_species:
        for ell in range(l_max + 1):
            keys.append([ell, center_species, neighbor_species])
            blocks.append(
                TensorBlock(
                    values=np.empty((1, n_max_l[ell])),
                    samples=Labels.single(),
                    components=[],
                    properties=Labels(
                        names=["n"],
                        values=np.arange(n_max_l[ell]).reshape(n_max_l[ell], 1),
                    ),
                )
            )

selected_properties = TensorMap(
    keys=Labels(
        names=["spherical_harmonics_l", "species_center", "species_neighbor"],
        values=np.array(keys),
    ),
    blocks=blocks,
)

# %%
#
# Build a calculator and calculate the spherical expansion

spliner = rascaline.utils.SoapSpliner(
    cutoff=cutoff,
    max_radial=n_max,
    max_angular=l_max,
    basis=rascaline.utils.SphericalBesselBasis(
        cutoff=cutoff, max_radial=n_max, max_angular=l_max
    ),
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
    selected_properties=selected_properties,
    # we tell the calculator to only compute the selected properties
)

# %%
#
# .. end-body
