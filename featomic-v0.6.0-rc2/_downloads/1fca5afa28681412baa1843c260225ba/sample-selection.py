"""
Sample Selection
================

.. start-body
"""

import chemfiles
import numpy as np
from metatensor import Labels

from featomic import SoapPowerSpectrum


# %%
#
# First we load the dataset with chemfiles

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    frames = [f for f in trajectory]

# %%
#
# and define the hyper parameters of the representation

HYPER_PARAMETERS = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 6},
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

descriptor = calculator.compute(frames)

# %%
#
# The selections for sample can be a set of ``Labels``, in which case the names
# of the labels must be a subset of the names of the samples produced by the
# calculator. You can see the default set of names with:

print("sample names:", descriptor.sample_names)

# %%
#
# We can use a subset of these names to define a selection. In this case, only
# samples matching the labels in this selection will be used by featomic (here,
# only atoms from system 0, 2, and 3)

selection = Labels(
    names=["system"],
    values=np.array([[0], [2], [3]]),
)

descriptor_selected = calculator.compute(frames, selected_samples=selection)

descriptor_selected = descriptor_selected.keys_to_samples("center_type")
descriptor_selected = descriptor_selected.keys_to_properties(
    ["neighbor_1_type", "neighbor_2_type"]
)

samples = descriptor_selected.block().samples

# %%
#
# The first block should have ``[0, 2, 3]`` as ``samples["system"]``

print(f"we have the following systems: {np.unique(samples['system'])}")

# %%
#
# If we want to select not only based on the system indexes but also atomic
# indexes, we can do the following (here we select atom 0 in the first system
# and atom 1 in the third system):

selection = Labels(
    names=["system", "atom"],
    values=np.array([[0, 0], [2, 1]]),
)

descriptor_selected = calculator.compute(frames, selected_samples=selection)
descriptor_selected = descriptor_selected.keys_to_samples("center_type")
descriptor_selected = descriptor_selected.keys_to_properties(
    ["neighbor_1_type", "neighbor_2_type"]
)

# %%
#
# The values will have 2 rows, since we have two samples:

print(
    "shape of first block of descriptor:",
    descriptor_selected.block(0).values.shape,
)

# %%
#
# The previous selection method uses the same selection for all blocks. If you
# can to use different selection for different blocks, you should use a
# `TensorMap` to create your selection

descriptor = calculator.compute(frames)
descriptor_selected = calculator.compute(frames, selected_samples=selection)

# %%
#
# notice how we are passing a TensorMap as the ``selected_samples`` argument:

print(type(descriptor_selected))
descriptor_for_comparison = calculator.compute(
    frames, selected_samples=descriptor_selected
)

# %%
#
# The descriptor had 420 samples stored in the first block,
# the ``descriptor_selected`` had 0. So ``descriptor_for_comparison``
# will also have 0 samples.

print("shape of first block initially:", descriptor.block(0).values.shape)
print(
    "shape of first block of reference:",
    descriptor_selected.block(0).values.shape,
)
print(
    "shape of first block after selection:",
    descriptor_for_comparison.block(0).values.shape,
)

# %%
#
# .. end-body
