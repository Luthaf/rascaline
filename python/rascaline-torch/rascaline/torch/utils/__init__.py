import os

from .clebsch_gordan import DensityCorrelations
from .power_spectrum import PowerSpectrum


_HERE = os.path.dirname(__file__)

cmake_prefix_path = os.path.realpath(os.path.join(_HERE, "..", "lib", "cmake"))
"""
Path containing the CMake configuration files for the underlying C library
"""

__all__ = ["PowerSpectrum", "DensityCorrelations"]
