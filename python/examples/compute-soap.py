import sys

import chemfiles

from rascaline import SoapPowerSpectrum


# read structures using chemfiles
with chemfiles.Trajectory(sys.argv[1]) as trajectory:
    frames = [f for f in trajectory]
# or using ASE, at your own convenience
# frames = ase.io.read(sys.argv[1], ":")

# define hyper parameters for the calculation
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

# run the actual calculation
descriptor = calculator.compute(frames)

# The descriptor is an equistore `TensorMap`, containing multiple blocks.
# We can transform it to a single block containing a dense representation, with
# one sample for each atom-centered environment.
descriptor.keys_to_samples("species_center")
descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

# you can now use descriptor.block(0).values as the
# input of a machine learning algorithm
print(descriptor.block(0).values.shape)
