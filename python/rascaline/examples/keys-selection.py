"""
Keys Selection
==============

.. start-body
"""
import chemfiles
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

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

# %%
#
# The selections for keys should be a set of ``Labels``, with the names of the
# keys being a subset of the names of the keys produced by the calculator.

descriptor = calculator.compute(frames)
print("keys names:", descriptor.keys.names)

# %%
#
# We can use these names to define a selection, and only blocks matching the
# labels in this selection will be used by rascaline. Here, only blocks with
# keys ``[1,1,1]`` and ``[4,4,4]`` will be calculated.

selection = Labels(
    names=["species_center", "species_neighbor_1", "species_neighbor_2"],
    values=np.array([[1, 1, 1], [4, 4, 4]], dtype=np.int32),
)
selected_descriptor = calculator.compute(frames, selected_keys=selection)

# %%
#
# We get a TensorMap with 2 blocks, corresponding to the requested keys

print(selected_descriptor.keys)

# %%
#
# The block for ``[1, 1, 1]`` will be exactly the same as the one in the full
# ``TensorMap``
answer = np.array_equal(descriptor.block(0).values, selected_descriptor.block(0).values)
print(f"Are the blocks 0 in the descriptor and selected_descriptor equal? {answer}")

# %%
#
# Since there is no block for ``[4, 4, 4]`` in the full ``TensorMap``, an empty
# block with no samples and the default set of properties is generated

print(selected_descriptor.block(1).values.shape)

# %%
#
# ``selected_keys`` can be used simultaneously with samples and properties
# selection. Here we define a selection for properties as a ``TensorMap`` to
# select different properties for each block:

selection = [
    Labels(names=["l", "n1", "n2"], values=np.array([[0, 0, 0]])),
    Labels(names=["l", "n1", "n2"], values=np.array([[1, 1, 1]])),
]
blocks = []
for entries in selection:
    blocks.append(
        TensorBlock(
            values=np.empty((len(entries), 1)),
            samples=Labels.single(),
            components=[],
            properties=entries,
        )
    )

keys = Labels(
    names=["species_center", "species_neighbor_1", "species_neighbor_2"],
    values=np.array([[1, 1, 1], [8, 8, 8]], dtype=np.int32),
)

selected_properties = TensorMap(keys, blocks)

# %%
#
# Only one of the key from our ``selected_properties`` will be used in the
# ``selected_keys``, meaning the output will only contain this one key/block.

selected_keys = Labels(
    names=["species_center", "species_neighbor_1", "species_neighbor_2"],
    values=np.array([[1, 1, 1]], dtype=np.int32),
)

descriptor = calculator.compute(
    frames,
    selected_properties=selected_properties,
    selected_keys=selected_keys,
)

# %%
#
# As expected, we get 1 block with values of the form (420, 1), i.e. with only 1
# property.

print(f"list of keys: {descriptor.keys}")
print(descriptor.block(0).values.shape)

# %%
#
# .. end-body
