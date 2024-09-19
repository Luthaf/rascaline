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
    "cutoff": {
        "radius": 5.5,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 8,
        "radial": {"max_radial": 7, "type": "Gto"},
        "spline_accuracy": None,
    },
}

calculator = SoapPowerSpectrum(**hyperparameters)
descriptor = calculator.compute(frame, use_native_system=True)
descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

save_calculator_input("soap-power-spectrum-values", frame, hyperparameters)
save_numpy_array("soap-power-spectrum-values", descriptor.block().values)

# Use less values for gradients to keep the file size low
hyperparameters["basis"]["radial"]["max_radial"] = 2
hyperparameters["basis"]["max_angular"] = 4

frame.cell = [6.0, 6.0, 6.0]
frame.pbc = [True, True, True]

calculator = SoapPowerSpectrum(**hyperparameters)
descriptor = calculator.compute(
    frame,
    use_native_system=True,
    gradients=["positions", "strain", "cell"],
)
descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])


def sum_gradient(descriptor):
    """compute the gradient w.r.t. each atom of the sum of all rows"""
    gradient = descriptor.block().gradient("positions")

    result = np.zeros((len(frame), 3, len(gradient.properties)))
    for sample, gradient in zip(gradient.samples, gradient.values):
        result[sample["atom"]] += gradient

    return result


save_calculator_input("soap-power-spectrum-gradients", frame, hyperparameters)
save_numpy_array("soap-power-spectrum-positions-gradient", sum_gradient(descriptor))

strain_gradient = descriptor.block().gradient("strain").values
save_numpy_array("soap-power-spectrum-strain-gradient", strain_gradient)

cell_gradient = descriptor.block().gradient("cell").values
save_numpy_array("soap-power-spectrum-cell-gradient", cell_gradient)
