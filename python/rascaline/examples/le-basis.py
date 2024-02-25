"""
.. _userdoc-tutorials-le-basis:

LE basis
========

.. start-body

This example illustrates how to generate a spherical expansion using the Laplacian
eigenstate (LE) basis (https://doi.org/10.1063/5.0124363), both using truncation with
``l_max``, ``n_max`` hyper-parameters and with an eigenvalue threshold.
The main ideas behind this approach are:

1. use a basis of controllable _smoothness_ (intended in the same sense as the 
   smoothness of a low-pass-truncated Fourier expansion) 
2. apply a "ragged truncation" strategy in which different ``l`` channels are 
   truncated at a different ``n_max``, so as to obtain more balanced smoothness
   level in the radial and angular direction, for a given number of basis functions.


Here we use :class:`rascaline.utils.SphericalBesselBasis` class. An detailed how-to
guide how to construct radial integrals for the LE basis from scratch is given in
:ref:`userdoc-tutorials-splined-radial-integrals`.
"""

# %%
#

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

import rascaline


# %%
#
# First using a truncation with ``l_max`` and ``n_max`` hyper-parameters. This uses
# basis functions that are the solution of a radial Laplacian eigenvalue problem
# (spherical Bessel functions) but apply a "traditional" basis selection strategy
# that retains all functions with ``l<l_max`` and ``n<n_max``

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

# %%
#
# Plot the splines for a couple of functions. This gives an idea of the
# smoothness of the different components
#

splined_basis = spliner.compute()
xgrid = [p["position"] for p in splined_basis["TabulatedRadialIntegral"]["points"]]
values = np.array(
    [
        np.array(p["values"]["data"]).reshape(p["values"]["dim"])
        for p in splined_basis["TabulatedRadialIntegral"]["points"]
    ]
)

plt.plot(xgrid, values[:, 1, 1], "b-", label="l=1, n=1")
plt.plot(xgrid, values[:, 4, 1], "r-", label="l=4, n=1")
plt.plot(xgrid, values[:, 1, 4], "g-", label="l=1, n=4")
plt.plot(xgrid, values[:, 4, 4], "y-", label="l=4, n=4")
plt.xlabel("$r$")
plt.ylabel(r"$R_{nl}$")
plt.legend()
plt.show()

# %%
#
# Now off to using the splines to evaluate spherical expansion coefficients

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
# Now we will calculate the same basis with an eigenvalue threshold (more involved),
# which affords a better accuracy/cost ratio, using property selection.
# The idea is to treat on the same footings the radial and angular dimension, and
# select all functions with a mean Laplacian below a certain threshold. This is similar
# to the common practice in plane-wave electronic-structure methods to use a
# kinetic energy cutoff where ``k_x**2+k_y**2+k_z**2<k_max**2``

E_max = 400  # eigenvalue threshold

# %%
#
# Computation of the spherical Bessel zeros and Laplacian eigenvalues

l_max_large = 50  # just used to get the eigenvalues
n_max_large = 50  # just used to get the eigenvalues

# compute the zeroth of the spherical Bessel functions
z_ln = rascaline.utils.SphericalBesselBasis.compute_zeros(l_max_large, n_max_large)

# calculate the Laplacian eigenvalues, up to a constant factor
# (the Laplacian eigenvalues are z_ln**2 / cutoff**2). These are directly
# related to the "resolution" of the corresponding basis functions
E_ln = z_ln**2

# %%
#
# Determine the l_max, n_max parameters that will certainly contain all the desired
# terms

n_max_l = []
for ell in range(l_max_large + 1):
    # for each l, calculate how many basis functions
    n_max_l.append(len(np.where(E_ln[ell] < E_max)[0]))
    if n_max_l[-1] == 0:
        # all eigenvalues for this `l` are over the threshold
        n_max_l.pop()
        l_max = ell - 1
        break

n_max = n_max_l[0]

# %%
#
# Comparing this l-dependent threshold with the one based on ``l_max`` and ``n_max``, we
# see that the eigenvalue thresholding leads to a gradual decrease of ``n_max`` for high
# ``l`` values

plt.fill_between(
    [0, 10], [4, 4], label=r"$l_\mathrm{max}$, $n_\mathrm{max}$ threshold", color="gray"
)
plt.fill_between(np.arange(l_max + 1), n_max_l, label="Eigenvalue threshold", alpha=0.5)
plt.xlabel("$l$")
plt.ylabel(r"$n_\mathrm{max}$")
plt.ylim(-0.5, n_max_l[0] + 0.5)
plt.legend()
plt.show()

# %%
#
# Set up a TensorMap for property selection. This allows to compute
# only a subset of the ``properties`` axis of the descriptors
# (see :ref:`userdoc-how-to-property-selection` for more details.).

# extract all the species from the small dataset
all_species = list(
    np.unique(np.concatenate([structure.numbers for structure in structures]))
)

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
# Build a calculator and calculate the spherical expansion coefficents

# set up a spliner object for the spherical Bessel functions
# this radial basis will be used to compute the spherical expansion
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
    # we tell the calculator to only compute the selected properties (the
    # desired set of (l,n) expansion coefficients
)

# %%
#
# .. end-body
