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
    "cutoff": 5.5,
    "max_radial": 8,
    "max_angular": 8,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {
            "splined_radial_integral": False,
        },
    },
    "cutoff_function": {
        "ShiftedCosine": {
            "width": 0.5,
        }
    },
}

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)

descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties("neighbor_type")
descriptor = descriptor.components_to_properties("o3_mu")
descriptor = descriptor.keys_to_properties("o3_lambda")

save_calculator_input("spherical-expansion-values", frame, hyperparameters)
save_numpy_array("spherical-expansion-values", descriptor.block().values)


def sum_gradient(descriptor):
    """compute the gradient w.r.t. each atom of the sum of all rows"""
    gradient = descriptor.block().gradient("positions")

    result = np.zeros((len(frame), 3, len(gradient.properties)))
    for sample, row in zip(gradient.samples, gradient.values):
        result[sample["atom"], :, :] += row[:, :]

    return result


# Use smaller hypers for gradients to keep the file size low
hyperparameters["max_radial"] = 4
hyperparameters["max_angular"] = 4
frame.cell = [6.0, 6.0, 6.0]
frame.pbc = [True, True, True]

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(
    frame,
    use_native_system=True,
    gradients=["positions", "strain", "cell"],
)

descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties("neighbor_type")
descriptor = descriptor.components_to_properties("o3_mu")
descriptor = descriptor.keys_to_properties("o3_lambda")

save_calculator_input("spherical-expansion-gradients", frame, hyperparameters)
save_numpy_array("spherical-expansion-positions-gradient", sum_gradient(descriptor))

strain_gradient = descriptor.block().gradient("strain").values
save_numpy_array("spherical-expansion-strain-gradient", strain_gradient)

cell_gradient = descriptor.block().gradient("cell").values
save_numpy_array("spherical-expansion-cell-gradient", cell_gradient)

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
        "Gto": {
            "splined_radial_integral": False,
        },
    },
    "cutoff_function": {
        "ShiftedCosine": {
            "width": 0.2,
        }
    },
}

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)

descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties("neighbor_type")
descriptor = descriptor.components_to_properties("o3_mu")
descriptor = descriptor.keys_to_properties("o3_lambda")

save_calculator_input("spherical-expansion-pbc-values", frame, hyperparameters)
save_numpy_array("spherical-expansion-pbc-values", descriptor.block().values)
