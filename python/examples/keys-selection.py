"""
Keys Selection
==============

.. start-body
"""
import chemfiles
import numpy as np
from equistore import Labels, TensorBlock, TensorMap

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
# The selections for keys can be a set of ``Labels``, in which case the names
# of the keys must be a set of the names of the keys produced by the
# calculator. You can see the default set of names with:

print("properties names:", descriptor.keys.names)

# %%
#
# We can use a set of these names to define a selection. In this case, only
# blocks matching the labels in this selection will be used by rascaline
# (here, only blocks with keys ``[1,1,1]`` and ``[4,4,4]`` will be calculated)

selection = Labels(
    names=["species_center", "species_neighbor_1", "species_neighbor_2"],
    values=np.array([[1, 1, 1], [4, 4, 4]], dtype=np.int32),
)
selected_descriptor = calculator.compute(frames, selected_keys=selection)

# %%
#
# We get a TensorMap with 2 blocks:

print(selected_descriptor.keys)

# %%
#
# In this case, the block ``[1,1,1]`` contained in the original ``TensorMap``
# will be exactly the same:
answer = np.array_equal(descriptor.block(0).values, selected_descriptor.block(0).values)
print(f"Are the blocks 0 in the descriptor and selected_descriptor equal? {answer}")

# %%
#
# And the form of the block ``[4,4,4]``, which is not created by default,
# will be 0,180, that is, an empty block with the required key and set of
# properties was generated

print(selected_descriptor.block(1).values.shape)

# %%
#
# In addition, ``selected_keys`` can be used simultaneously with other
# selectors. To do this, create ``TensorMap`` below, which we will pass
# as ``selected_properties``:

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
# As ``selected_keys`` we will take only 1 of the keys that were in
# ``selected_properties``

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
# As a result, we get 1 block with values of the form (420, 1), that is, with
# only 1 property

print(f"list of keys: {descriptor.keys}")
print(descriptor.block(0).values.shape)

# %%
#
# .. end-body
