import sys
import ase
from ase import io

from rascaline import SphericalExpansion

# read structures using ASE
frames = ase.io.read(sys.argv[1], ":")

# define hyper parameters for the calculation
HYPER_PARAMETERS = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "gradients": False,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.5},
    },
}

calculator = SphericalExpansion(**HYPER_PARAMETERS)

# run the actual calculation, use_native_system=True is usually much faster
descriptor = calculator.compute(frames, use_native_system=True)

# Transform the descriptor to dense representation,
# with one sample for each atom-centered environment
descriptor.densify(["neighbor_species"])

# you can now use descriptor.values as the
# input of a machine learning algorithm
