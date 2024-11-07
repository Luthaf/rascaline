import os

from .clebsch_gordan import (  # noqa: F401
    ClebschGordanProduct,
    DensityCorrelations,
    calculate_cg_coefficients,
    cartesian_to_spherical,
)
from .hypers import BadHyperParameters, convert_hypers, hypers_to_json  # noqa: F401
from .power_spectrum import PowerSpectrum  # noqa: F401


_HERE = os.path.dirname(__file__)

cmake_prefix_path = os.path.realpath(os.path.join(_HERE, "..", "lib", "cmake"))
"""
Path containing the CMake configuration files for the underlying C library
"""
