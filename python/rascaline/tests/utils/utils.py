import metatensor
import numpy as np
import pytest

import rascaline


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402,F811


try:
    import metatensor.operations

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


def spherical_expansion():
    """Returns a rascaline SphericalExpansion"""
    hypers = {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 3,  # Exclusive
        "max_angular": 3,  # Inclusive
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }
    calculator = rascaline.SphericalExpansion(**hypers)

    frames = ase.Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [2.56633400, 2.50000000, 2.50370100],
            [1.97361700, 1.73067300, 2.47063400],
            [1.97361700, 3.26932700, 2.47063400],
        ],
    )
    density = calculator.compute(frames)
    density = density.keys_to_properties(
        metatensor.Labels(["neighbor_type"], np.array([1, 8]).reshape(-1, 1))
    )
    return density
