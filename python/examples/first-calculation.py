"""
.. _userdoc-tutorials-get-started:

First SOAP calculation
======================

This is an introduction to the rascaline interface using a molecular crystals
dataset using the Python interface. If you are interested in another
programming language we recommend you first follow this tutorial and afterward
take a look at the how-to guide on :ref:`userdoc-how-to-computing-soap`.

The dataset
-----------

The atomic configurations used in our documentation are a small subset of the
`ShiftML2 dataset <https://pubs.acs.org/doi/pdf/10.1021/acs.jpcc.2c03854>`_
containing molecular crystals. There are four crystals - one with each of the
elements [hydrogen, carbon], [hydrogen, carbon, nitrogen, oxygen], [hydrogen,
carbon, nitrogen], or [hydrogen, carbon, oxygen]. Each crystal has 10 structures,
also denoted by frames, attributed to it. The first frame of each crystal structure
is the geometry-optimized frame. The following 9 frames contain atoms that are
slightly displaced from the geometry-optimized frame. You can obtain the dataset
from our :download:`website <../../static/dataset.xyz>`.
"""

# %%
#
# We will start by importing all the required packages: the classic numpy;
# chemfiles to load data, and rascaline to compute representations. Afterward
# we will load the dataset using chemfiles.

import chemfiles
import numpy as np

from rascaline import SphericalExpansion


with chemfiles.Trajectory("dataset.xyz") as trajectory:
    frames = [f for f in trajectory]

print(f"The dataset contains {len(frames)} frames.")

# %%
#
# We will not explain here how to use chemfiles in detail, as we only use a few
# functions. Briefly, :class:`chemfiles.Trajectory` loads structure data in a
# format rascaline can use. If you want to learn more about the possibilities
# take a look at the `chemfiles documentation <https://chemfiles.org>`_.
#
# Let us now take a look at the first frame of the dataset.

frame0 = frames[0]

print(frame0)

# %%
#
# With ``frame0.atoms`` we get a list of the atoms that make up frame zero.
# The ``name`` attribute gives us the name of the specified atom.

elements, counts = np.unique([atom.name for atom in frame0.atoms], return_counts=True)

print(
    f"The first frame contains "
    f"{counts[0]} {elements[0]}-atoms, "
    f"{counts[1]} {elements[1]}-atoms, "
    f"{counts[2]} {elements[2]}-atoms and "
    f"{counts[3]} {elements[3]}-atoms."
)

# %%
#
# Calculate a descriptor
# ----------------------
#
# We will now calculate an atomic descriptor for this structure using the SOAP
# spherical expansion as introduced by `Bartók, Kondor, and Csányi
# <http://dx.doi.org/10.1103/PhysRevB.87.184115>`_.
#
# To do so we define below a set of parameters telling rascaline how the
# spherical expansion should be calculated. These parameters are also called
# hyper parameters since they are parameters of the representation, in
# opposition to parameters of machine learning models. Hyper parameters are a
# crucial part of calculating descriptors. Poorly selected hyper parameters will
# lead to a poor description of your dataset as discussed in the `literature
# <https://arxiv.org/abs/1502.02127>`_. The effect of
# changing some hyper parameters is discussed in a :ref:`second tutorial
# <userdoc-tutorials-understanding-hypers>`.

HYPER_PARAMETERS = {
    "cutoff": 4.5,
    "max_radial": 9,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling": {"Willatt2018": {"scale": 2.0, "rate": 1.0, "exponent": 4}},
}

# %%
#
# After we set the hyper parameters we initialize a
# :class:`rascaline.calculators.SphericalExpansion`
# object with hyper parameters defined above.

calculator = SphericalExpansion(**HYPER_PARAMETERS)
descriptor0 = calculator.compute(frame0)
print(type(descriptor0))

# %%
#
# The descriptor format is a :class:`equistore.tensor.TensorMap` object.
# Equistore is like numpy for storing representations of atomistic ML data.
# Extensive details on the equistore are covered in the `corresponding
# documentation <https://lab-cosmo.github.io/equistore/>`_.
#
# We will now have a look at how the data is stored inside
# :class:`equistore.tensor.TensorMap` objects.


print(descriptor0)

# %%
#
# The :class:`equistore.tensor.TensorMap` is structured in ``blocks``, each
# associated with a key. Here we have one block for each angular channel
# labeled by ``spherical_harmonics_l``,
# the central atom species ``species_center`` and neighbor atom species
# labeled by ``species_neighbor``. Different atomic species are
# represented using their atomic number, e.g. 1 for hydrogen, 6 for carbon, etc.
# Our descriptor contains 112 blocks covering all combinations of the angular
# channels and the elements.
#
# Let us take a look at the first block in detail. The 0th block contains the
# descriptor for the 0th angular channel between the hydrogen-hydrogen pairs.

print(descriptor0.keys[0])

# %%
#
# The values are stored in an array. Let us take a look at its dimensions given
# by the shape attribute.


print(descriptor0.block(1).values.shape)

# %%
# The first dimension is 8 because we have eight hydrogen atoms in our frame.
# The second dimension
# is associated with the selected angular channel and has a size of
# :math:$2l + 1$, where :math:`l` is the current
# ``spherical_harmonics_l`` channel.
# Here it dimension is 1 because we are looking at the ``spherical_harmonics_l=0``
# channel. The last value represents the number
# of radial channels. We choose ``max_radial=9`` in the hyper parameters above
# so there are 9 values in the last dimension.
#
# Let's take a look at the values of the representation (also called features)
# associated with the first hydrogen:

print(descriptor0.block(0).values[0])

# %%
#
# As you see the values are floating point numbers between -1 and 1. Values in
# this range are reasonable and can be directly used as an input for a machine
# learning algorithm.
#
# Rascaline is also able to process more than one structure within one function
# call. You can process a whole dataset with

descriptor_full = calculator.compute(frames)
print(descriptor_full.block(0).values.shape)

# %%
#
# Now, the 0th block of the :class:`equistore.tensor.TensorMap` contains not 8
# but 420 entries in the first dimensions. This reflects the fact that in total
# we have 420 hydrogen atoms in the whole dataset.
#
#
# If you want to use another calculator instead of
# :class:`rascaline.calculators.SphericalExpansion` shown here
# check out the :ref:`userdoc-references` section.
#
# Computing gradients
# -------------------
#
# Additionally, rascaline is also able to calculate the gradients with respect
# to positions, which is useful for constructing an ML potential and running
# simulations. ``gradients``
# of the representation with respect to atomic positions can be calculated by
# setting the ``gradients`` parameter of the compute method to
# ``["positions"]``.

descriptor = calculator.compute(frame0, gradients=["positions"])
print(descriptor.block(0).gradient("positions").data.shape)

# %%
#
# The gradients are stored in the data attribute. The shape of gradients of the
# first frame is calculated as follows: `8*8=64` gradients for all
# hydrogen-hydrogen pairs. Rascaline can also calculate gradients with respect
# to the cell by setting ``gradients=["cell"]``. Cell gradients are useful when
# computing stress and pressure.
#
# If you want to know about the effect of changing hypers take a look at the
# next tutorial. If you want to solve an explicit problem our
# :ref:`userdoc-how-to` might help you.
