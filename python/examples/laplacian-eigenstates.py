from rascaline import SphericalExpansion

hypers_spherical_expansion = {
        "cutoff": 4.5,
        "max_radial": 4,
        "max_angular": 3,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {"Tabulated": {"file": "non-existent.txt"}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}},
    }

calculator = SphericalExpansion(**hypers_spherical_expansion)