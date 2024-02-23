import os

import ase
import numpy as np
from ase import io  # noqa

from rascaline import LodeSphericalExpansion

from save_data import save_calculator_input, save_numpy_array


ROOT = os.path.abspath(os.path.dirname(__file__))

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
    pbc=[True, True, True],
    cell=[6, 6, 6],
)


def sum_gradient(descriptor):
    """compute the gradient w.r.t. each atom of the sum of all rows"""
    gradient = descriptor.block().gradient("positions")

    result = np.zeros((len(frame), 3, len(gradient.properties)))
    for sample, row in zip(gradient.samples, gradient.data):
        result[sample["atom"], :, :] += row[:, :]

    return result


try:
    os.mkdir(os.path.join(ROOT, "generated", "lode-spherical-expansion"))
except OSError:
    pass


for potential_exponent in [1, 2, 3, 4, 5, 6]:
    path = os.path.join(
        ROOT,
        "generated",
        "lode-spherical-expansion",
        f"potential_exponent-{potential_exponent}",
    )
    try:
        os.mkdir(path)
    except OSError:
        pass

    hyperparameters = {
        "cutoff": 2.5,
        "max_radial": 4,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {
                "splined_radial_integral": False,
            },
        },
        "potential_exponent": potential_exponent,
    }

    calculator = LodeSphericalExpansion(**hyperparameters)
    descriptor = calculator.compute(frame, use_native_system=True)

    descriptor.keys_to_samples("center_type")
    descriptor.keys_to_properties("neighbor_type")
    descriptor.components_to_properties("o3_mu")
    descriptor.keys_to_properties("o3_lambda")

    save_calculator_input(os.path.join(path, "values"), frame, hyperparameters)
    save_numpy_array(os.path.join(path, "values"), descriptor.block().values)

    # Use smaller hypers for gradients to keep the file size low
    hyperparameters["max_radial"] = 3
    hyperparameters["max_angular"] = 3

    calculator = LodeSphericalExpansion(**hyperparameters)
    descriptor = calculator.compute(
        frame,
        use_native_system=True,
        gradients=["positions"],
    )

    descriptor.keys_to_samples("center_type")
    descriptor.keys_to_properties("neighbor_type")
    descriptor.components_to_properties("o3_mu")
    descriptor.keys_to_properties("o3_lambda")

    save_calculator_input(os.path.join(path, "gradients"), frame, hyperparameters)
    save_numpy_array(os.path.join(path, "positions-gradient"), sum_gradient(descriptor))
