import ase

from rascaline.calculator import SphericalExpansion

from save_data import save_data

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
    "radial_basis": "GTO",
    "gradients": False,
    "cutoff_function": {
        "ShiftedCosine": {
            "width": 0.5,
        }
    },
}

calculator = SphericalExpansion(**hyperparameters)
descriptor = calculator.compute(frame)

save_data("spherical-expansion", frame, hyperparameters, descriptor)
