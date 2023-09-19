"""
Property Selection
==================

.. start-body
"""
import chemfiles
import numpy as np
from metatensor import Labels, MetatensorError, TensorBlock, TensorMap
from skmatter.feature_selection import FPS

from rascaline import SoapPowerSpectrum


# %%
#
# First we load the dataset with chemfiles

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    frames = [f for f in trajectory]

# %%
#
# and define the hyper parameters of the representation

HYPER_PARAMETERS = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.5},
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

descriptor = calculator.compute(frames)

# %%
#
# The selections for feature can be a set of ``Labels``, in which case the names
# of the labels must be a subset of the names of the properties produced by the
# calculator. You can see the default set of names with:

print("properties names:", descriptor.properties_names)

# %%
#
# We can use a subset of these names to define a selection. In this case, only
# properties matching the labels in this selection will be used by rascaline
# (here, only properties with ``l = 0`` will be used)

selection = Labels(
    names=["l"],
    values=np.array([[0]]),
)
selected_descriptor = calculator.compute(frames, selected_properties=selection)

selected_descriptor = selected_descriptor.keys_to_samples("species_center")
selected_descriptor = selected_descriptor.keys_to_properties(
    ["species_neighbor_1", "species_neighbor_2"]
)

properties = selected_descriptor.block().properties

# %%
#
# We expect to get `[0]` as the list of `l` properties

print(f"we have the following angular components: {np.unique(properties['l'])}")

# %%
#
# The previous selection method uses the same selection for all blocks. If you
# can to use different selection for different blocks, you should use a
# ``TensorMap`` to create your selection

selected_descriptor = calculator.compute(frames, selected_properties=selection)
descriptor_for_comparison = calculator.compute(
    frames, selected_properties=selected_descriptor
)

# %%
#
# The descriptor had 180 properties stored in the first block, the
# selected_descriptor had 36. So ``descriptor_for_comparison`` will also have 36
# properties.
print("shape of first block initially:", descriptor.block(0).values.shape)
print("shape of first block of reference:", selected_descriptor.block(0).values.shape)
print(
    "shape of first block after selection:",
    descriptor_for_comparison.block(0).values.shape,
)

# %%
#
# The ``TensorMap`` format allows us to select different features within each
# block, and then construct a general matrix of features. We can select the most
# significant features using FPS, which selects features based on the distance
# between them. The following code snippet selects the 10 most important
# features in each block, then constructs a TensorMap containing this selection,
# and calculates the final matrix of features for it.


def fps_feature_selection(descriptor, n_to_select):
    """
    Select ``n_to_select`` features block by block in the ``descriptor``, using
    Farthest Point Sampling to do the selection; and return a ``TensorMap`` with
    the right structure to be used as properties selection with rascaline calculators
    """
    blocks = []
    for block in descriptor:
        # create a separate FPS selector for each block
        fps = FPS(n_to_select=n_to_select)
        mask = fps.fit(block.values).get_support()
        selected_properties = Labels(
            names=block.properties.names,
            values=block.properties.values[mask],
        )
        # The only important data here is the properties, so we create empty
        # sets of samples and components.
        blocks.append(
            TensorBlock(
                values=np.empty((1, len(selected_properties))),
                samples=Labels.single(),
                components=[],
                properties=selected_properties,
            )
        )

    return TensorMap(descriptor.keys, blocks)


# %%
#
# We can then apply this function to subselect according to the data contained
# in a descriptor

selection = fps_feature_selection(descriptor, n_to_select=10)

# %%
#
# and use the selection with rascaline, potentially running the calculation on a
# different set of systems

selected_descriptor = calculator.compute(frames, selected_properties=selection)

# %%
#
# Note that in this case it is no longer possible to have a single feature
# matrix, because each block will have its own properties.

try:
    selected_descriptor.keys_to_samples("species_center")
except MetatensorError as err:
    print(err)

# %%
#
# .. end-body
