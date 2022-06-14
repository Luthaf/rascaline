import ase
import numpy as np

from rascaline import SoapPowerSpectrum

from save_data import save_calculator_input, save_numpy_array


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

calculator = SoapPowerSpectrum(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor.keys_to_samples("species_center")
descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

save_calculator_input("soap-power-spectrum-values", frame, hyperparameters)
save_numpy_array("soap-power-spectrum-values", descriptor.block().values)

# Use less values for gradients to keep the file size low
hyperparameters["max_radial"] = 4
hyperparameters["max_angular"] = 4
hyperparameters["gradients"] = True

calculator = SoapPowerSpectrum(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor.keys_to_samples("species_center")
descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])


def sum_gradient(descriptor):
    """compute the gradient w.r.t. each atom of the sum of all rows"""
    gradient = descriptor.block().gradient("positions")

    result = np.zeros((len(frame), 3, len(gradient.properties)))
    for sample, gradient in zip(gradient.samples, gradient.data):
        result[sample["atom"]] += gradient

    return result


save_calculator_input("soap-power-spectrum-gradients", frame, hyperparameters)
save_numpy_array("soap-power-spectrum-gradients", sum_gradient(descriptor))
