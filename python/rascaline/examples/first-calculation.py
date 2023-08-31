"""
.. _userdoc-tutorials-get-started:

First descriptor computation
============================

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
# <https://arxiv.org/abs/1502.02127>`_. The effect of changing some hyper
# parameters is discussed in a :ref:`second tutorial
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
# :class:`rascaline.calculators.SphericalExpansion` object with hyper parameters
# defined above and run the
# :py:func:`rascaline.calculators.CalculatorBase.compute()` method.

calculator = SphericalExpansion(**HYPER_PARAMETERS)
descriptor0 = calculator.compute(frame0)
print(type(descriptor0))

# %%
#
# The descriptor format is a :class:`metatensor.TensorMap` object. Metatensor is
# like numpy for storing representations of atomistic ML data. Extensive details
# on the metatensor are covered in the `corresponding documentation
# <https://lab-cosmo.github.io/metatensor/>`_.
#
# We will now have a look at how the data is stored inside
# :class:`metatensor.TensorMap` objects.


print(descriptor0)

# %%
#
# The :class:`metatensor.TensorMap` is structured in several instances of an
# :class:`metatensor.TensorBlock`. To distinguish the block each block is
# associated with a unique key. For the current example, we have one block for
# each angular channel labeled by ``spherical_harmonics_l``, the central atom
# species ``species_center`` and neighbor atom species labeled by
# ``species_neighbor``. Different atomic species are represented using their
# atomic number, e.g. 1 for hydrogen, 6 for carbon, etc. To summarize, this
# descriptor contains 112 blocks covering all combinations of the angular
# channels of the central and neighbor atom species in our dataset.
#
# Let us take a look at the second block (at index 1) in detail. This block
# contains the descriptor for the :math:`l=1` angular channel for
# hydrogen-hydrogen pairs.

block = descriptor0.block(1)
print(descriptor0.keys[1])


# %%
#
# The descriptor values
# ---------------------
#
# The values of the representation are stored as an array. Each entry in this
# array also has associated unique metadata as each block. For the spherical
# expansion calculator used in this tutorial the values have three dimensions
# which we can verify from the ``.shape`` attribute.


print(block.values.shape)

# %%
#
# The descriptor values
# ---------------------
#
# The first dimension is denoted by the `samples`, the intermediate dimension by
# `components`, and the last dimension by `properties`. The "sample dimension"
# has a length of eight because we have eight hydrogen atoms in the first frame.
# We can reveal more detailed metadata information about the sample-dimension
# printing of the :py:attr:`metatensor.TensorBlock.samples` attribute of the
# block

print(block.samples)

# %%
#
# The result is an :class:`metatensor.TensorMap` instance. It contains in total
# eight tuples each with two values. The tuple values are named as follows

print(block.samples.names)

# %%
#
# Meaning that the first entry of each tuple indicates the _structure_, which is
# 0 for all because we only computed the representation of a single frame. The
# second entry of each tuple refers to the index of the _center_ atom.
#
# We can do a similar investigation for the second dimension: the
# :py:attr:`metatensor.TensorBlock.components`.

print(block.components)

# %%
#
# Here, the components are associated with the angular channels of the
# representation. The size of ``spherical_harmonics_m`` is :math:`2l + 1`, where
# :math:`l` is the current ``spherical_harmonics_l`` of the block. Here, its
# dimension is three because we are looking at the ``spherical_harmonics_l=1``
# block. You may have noticed that the return value of the last call is a
# :class:`list` of :class:`metatensor.Labels` and not a single ``Labels``
# instance. The reason is that a block can have several component dimensions as
# we will see below for the gradients.
#
# The last value represents the number of radial channels. For the
# :py:attr:`metatensor.TensorBlock.properties` dimension we find an object

print(block.properties)

# %%
#
# containing a tuple of only one value ranging from 0 to 8. The name of this entry is

print(block.properties.names)

# %%
#
# and denoting the radial channels. The range results from our choice of
# ``max_radial = 9`` in the hyper parameters above.
#
# After looking at the metadata we can investigate the actual data of the
# representation in more details

print(block.values[0, 0, :])

# %%
#
# By using ``[0, 0, :]`` we selected the first hydrogen and the first ``m``
# channel. As you the output shows the values are floating point numbers between
# ``-1.0`` and ``1.0``. Values in this range are reasonable and can be directly
# used as input for a machine learning algorithm.
#
# Rascaline is also able to process more than one structure within one function
# call. You can process a whole dataset with

descriptor_full = calculator.compute(frames)

block_full = descriptor_full.block(0)
print(block_full.values.shape)

# %%
#
# Now, the 0th block of the :class:`metatensor.TensorMap` contains not eight but
# 420 entries in the first dimensions. This reflects the fact that in total we
# have 420 hydrogen atoms in the whole dataset.
#
# If you want to use another calculator instead of
# :class:`rascaline.calculators.SphericalExpansion` shown here check out the
# :ref:`userdoc-references` section.
#
# Computing gradients
# -------------------
#
# Additionally, rascaline is also able to calculate gradients on top of the
# values. Gradients are useful for constructing an ML potential and running
# simulations. For example ``gradients`` of the representation with respect to
# atomic positions can be calculated by setting the ``gradients`` parameter of
# the :py:func:`rascaline.calculators.CalculatorBase.compute()` method to
# ``["positions"]``.

descriptor_gradients = calculator.compute(frame0, gradients=["positions"])

block_gradients = descriptor_gradients.block(0)
gradient_position = block_gradients.gradient("positions")

print(gradient_position.values.shape)

# %%
#
# The calculated descriptor contains the values and in each block the associated
# position gradients as an :class:`metatensor.block.Gradient` instance. The
# actual values are stored in the ``data`` attribute. Similar to the features
# the gradient data also has associated metadata. But, compared to the values
# were we found three dimensions, and gradients have four. Again the first is
# called `samples` and the `properties`. The dimensions between the sample and
# property dimensions are denoted by `components`.
#
# Looking at the shape in more detail we find that we have 52 samples, which is
# much more compared to features where we only have eight samples. This arises
# from the fact that we calculate the position gradient for each pair in the
# structure. For our selected block these are all hydrogen-hydrogen pairs.
# Naively one would come up with ``8 * 8 = 64`` samples, but rascaline already
# ignores pairs that are outside of the cutoff radius. Their position gradient
# is always zero. The :attr:`metatensor.block.Gradient.samples` attribute shows
# this in detail.

print(gradient_position.samples)

# %%
#
# Note that we have a tuple of three with the names

print(gradient_position.samples.names)

# %%
#
# In the above output of the Labels instance for example the `(2, 0, 17)` entry
# is missing indicating that this pair is outside of the cutoff.
#
# Now looking at the :attr:`metatensor.block.Gradient.components`

print(gradient_position.components)

# %%
#
# we find two of them. Besides the `spherical_harmonics_m` component that is
# also present in the features position gradients also have a component
# indicating the direction of the gradient vector.
#
# Finally, the :attr:`metatensor.block.Gradient.properties` dimension is the same
# as for the values

print(gradient_position.properties)

# %%
#
# Rascaline can also calculate gradients with respect to the cell. For this, you
# have to add ``"cell"`` to the list parsed to the ``gradients`` parameter of
# the :py:func:`rascaline.calculators.CalculatorBase.compute()` method. Cell
# gradients are useful when computing the stress and the pressure.
#
# If you want to know about the effect of changing hypers take a look at the
# next tutorial. If you want to solve an explicit problem our
# :ref:`userdoc-how-to` might help you.
