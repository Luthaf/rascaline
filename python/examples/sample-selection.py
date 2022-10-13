import sys

import chemfiles
import numpy as np
from equistore import Labels

from rascaline import SoapPowerSpectrum


with chemfiles.Trajectory(sys.argv[1]) as trajectory:
    frames = [f for f in trajectory]

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

# The selections for sample can be a set of `Labels`, in which case
# the names of the labels must be a subset of the names of the
# samples produced by the calculator. You can see the default set
# of names with:
print("samples names:", descriptor.sample_names)

# We can use a subset of these names to define a selection.
# In this case, only samples matching the labels in this selection
# will be used by rascaline (here, only atoms from
# structures 0, 2, and 3)
selection = Labels(
    names=["structure"],
    values=np.array([[0], [2], [3]]),
)

descriptor_partial_structures = calculator.compute(frames, selected_samples=selection)

descriptor_partial_structures.keys_to_samples("species_center")
descriptor_partial_structures.keys_to_properties(
    ["species_neighbor_1", "species_neighbor_2"]
)

samples = descriptor_partial_structures.block().samples
# The first block should have `[0, 2, 3]` as `samples["structure"]`
print(f"we have the following structures: {np.unique(samples['structure'])}")

# If we want to select not only based on the structure indexes but also
# atomic indexes, we can do the following (here we select atom 0 in the
# first structure and atom 1 in the third structure):
selection = Labels(
    names=["structure", "center"],
    values=np.array([[0, 0], [2, 1]]),
)

descriptor_partial_structures = calculator.compute(frames, selected_samples=selection)
descriptor_partial_structures.keys_to_samples("species_center")
descriptor_partial_structures.keys_to_properties(
    ["species_neighbor_1", "species_neighbor_2"]
)

# The values will have 2 rows, since we have two samples:
print(
    "shape of first block of descriptor:",
    {descriptor_partial_structures.block(0).values.shape},
)

# The previous selection method uses the same selection for all blocks.
# If you can to use different selection for different blocks, you should
# use a `TensorMap` to create your selection

descriptor = calculator.compute(frames)
descriptor_partial_structures = calculator.compute(frames, selected_samples=selection)

print(type(descriptor_partial_structures))
# notice how we are passing a TensorMap as the `selected_samples` argument:
descriptor_for_comparison = calculator.compute(
    frames, selected_samples=descriptor_partial_structures
)
# The descriptor had 420 samples stored in the first block,
# the descriptor_partial_structures had 0. So descriptor_for_comparison
# will also have 0 samples.
print("shape of first block initially:", {descriptor.block(0).values.shape})
print(
    "shape of first block of reference:",
    {descriptor_partial_structures.block(0).values.shape},
)
print(
    "shape of first block after selection:",
    {descriptor_for_comparison.block(0).values.shape},
)
