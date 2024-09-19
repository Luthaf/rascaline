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
    for sample, row in zip(gradient.samples, gradient.values):
        result[sample["atom"], :, :] += row[:, :]

    return result


try:
    os.mkdir(os.path.join(ROOT, "generated", "lode-spherical-expansion"))
except OSError:
    pass


for exponent in [1, 2, 3, 4, 5, 6]:
    path = os.path.join(
        ROOT,
        "generated",
        "lode-spherical-expansion",
        f"exponent-{exponent}",
    )
    try:
        os.mkdir(path)
    except OSError:
        pass

    hyperparameters = {
        "density": {
            "type": "SmearedPowerLaw",
            "smearing": 0.3,
            "exponent": exponent,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"max_radial": 3, "type": "Gto", "radius": 2.5},
            "spline_accuracy": None,
        },
    }

    calculator = LodeSphericalExpansion(**hyperparameters)
    descriptor = calculator.compute(frame, use_native_system=True)

    descriptor = descriptor.keys_to_samples("center_type")
    descriptor = descriptor.keys_to_properties("neighbor_type")
    descriptor = descriptor.components_to_properties("o3_mu")
    descriptor = descriptor.keys_to_properties("o3_lambda")

    save_calculator_input(os.path.join(path, "values"), frame, hyperparameters)
    save_numpy_array(os.path.join(path, "values"), descriptor.block().values)

    # Use smaller hypers for gradients to keep the file size low
    hyperparameters["basis"]["radial"]["max_radial"] = 2
    hyperparameters["basis"]["max_angular"] = 3

    calculator = LodeSphericalExpansion(**hyperparameters)
    descriptor = calculator.compute(
        frame,
        use_native_system=True,
        gradients=["positions"],
    )

    descriptor = descriptor.keys_to_samples("center_type")
    descriptor = descriptor.keys_to_properties("neighbor_type")
    descriptor = descriptor.components_to_properties("o3_mu")
    descriptor = descriptor.keys_to_properties("o3_lambda")

    save_calculator_input(os.path.join(path, "gradients"), frame, hyperparameters)
    save_numpy_array(os.path.join(path, "positions-gradient"), sum_gradient(descriptor))
