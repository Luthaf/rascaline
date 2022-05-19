import os

import ase
import numpy as np
from ase import io  # noqa

from rascaline import SphericalExpansion

from save_data import save_calculator_input, save_numpy_array


# structure without periodic boundary condition
frame = ase.Atoms(
    "CHOC4H2",
    positions=[
        [3.88997, 5.11396, 1.9859],
        [1.60538, 5.74085, 3.48071],
        [5.15178, 5.59335, 5.55114],
        [2.22548, 2.03678, 4.16896],
        [2.33853, 2.79487, 2.3533],
        [3.54073, 3.59016, 2.34664],
        [1.34344, 2.94555, 2.4665],
        [1.74165, 3.03466, 0.921584],
        [0.474942, 3.34246, 2.73754],
    ],
    pbc=[False, False, False],
)

hyperparameters = {
    "cutoff": 3.5,
    "max_radial": 8,
    "max_angular": 8,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "gradients": False,
    "cutoff_function": {
        "ShiftedCosine": {
            "width": 0.5,
        }
    },
}

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor.densify("species_neighbor")

save_calculator_input("spherical-expansion-values", frame, hyperparameters)
save_numpy_array("spherical-expansion-values", descriptor.values)

# Use less values for gradients to keep the file size low
hyperparameters["max_radial"] = 4
hyperparameters["max_angular"] = 4
hyperparameters["gradients"] = True


def sum_gradient(descriptor):
    """compute the gradient w.r.t. each atom of the sum of all rows"""
    result = np.zeros((len(frame), 3, descriptor.gradients.shape[1]))
    for sample, gradient in zip(descriptor.gradients_samples, descriptor.gradients):
        result[sample["atom"], sample["spatial"], :] += gradient[:]

    return result


calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor.densify("species_neighbor")

save_calculator_input("spherical-expansion-gradients", frame, hyperparameters)
save_numpy_array("spherical-expansion-gradients", sum_gradient(descriptor))

# structure with periodic boundary condition. Some atoms in this structure are
# neighbors twice with a cutoff of 4.5 (pairs 0-19, 1-18, 9-16, 12-13 after the
# pop below).
datadir = os.path.join(os.path.dirname(__file__), "..", "..", "benches", "data")
frame = ase.io.read(os.path.join(datadir, "molecular_crystals.xyz"), "4")

# remove most atoms to have smaller reference data while keeping duplicated
# pairs
for _ in range(60):
    frame.pop()

for _ in range(20):
    frame.pop(5)

hyperparameters = {
    "cutoff": 4.5,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.5,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "gradients": False,
    "cutoff_function": {
        "ShiftedCosine": {
            "width": 0.2,
        }
    },
}

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor.densify("species_neighbor")

save_calculator_input("spherical-expansion-pbc-values", frame, hyperparameters)
save_numpy_array("spherical-expansion-pbc-values", descriptor.values)
