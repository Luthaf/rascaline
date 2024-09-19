"""
Computing SOAP features
=======================

.. start-body
"""

import chemfiles

from rascaline import SoapPowerSpectrum


# %%
#
# Read systems using chemfiles. You can obtain the dataset used in this
# example from our :download:`website <../../static/dataset.xyz>`.

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    systems = [s for s in trajectory]

# %%
#
# Rascaline can also handles systems read by `ASE
# <https://wiki.fysik.dtu.dk/ase/>`_ using
#
# ``systems = ase.io.read("dataset.xyz", ":")``.
#
# We can now define hyper parameters for the calculation

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

# %%
#
# And then run the actual calculation, including gradients with respect to positions

descriptor = calculator.compute(systems, gradients=["positions"])

# %%
#
# The descriptor is a metatensor ``TensorMap``, containing multiple blocks. We
# can transform it to a single block containing a dense representation, with one
# sample for each atom-centered environment by using ``keys_to_samples`` and
# ``keys_to_properties``

print("before: ", len(descriptor.keys))

descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
print("after: ", len(descriptor.keys))

# %%
#
# you can now use ``descriptor.block().values`` as the input of a machine
# learning algorithm

print(descriptor.block().values.shape)


# %%
#
# .. end-body
