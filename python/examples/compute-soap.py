"""
Computing SOAP features
=======================

.. start-body
"""
import chemfiles

from rascaline import SoapPowerSpectrum


# %%
#
# Read structures using chemfiles. You can obtain the dataset used in this
# example from our :download:`website <../../static/dataset.xyz>`.

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    frames = [f for f in trajectory]

# %%
#
# Rascaline can also handles structures read by `ASE
# <https://wiki.fysik.dtu.dk/ase/>`_ using
#
# ``frames = ase.io.read("dataset.xyz", ":")``.
#
# We can now define hyper parameters for the calculation

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
# And then run the actual calculation, including gradients with respect to positions

descriptor = calculator.compute(frames, gradients=["positions"])

# %%
#
# The descriptor is an equistore ``TensorMap``, containing multiple blocks. We
# can transform it to a single block containing a dense representation, with one
# sample for each atom-centered environment by using ``keys_to_samples`` and
# ``keys_to_properties``

print("before: ", len(descriptor.keys))

descriptor.keys_to_samples("species_center")
descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])
print("after: ", len(descriptor.keys))

# %%
#
# you can now use ``descriptor.block().values`` as the input of a machine
# learning algorithm

print(descriptor.block().values.shape)


# %%
#
# .. end-body
