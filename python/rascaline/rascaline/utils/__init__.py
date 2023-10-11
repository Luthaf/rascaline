import os

import metatensor

from .clebsch_gordan import *  # noqa
from .power_spectrum import PowerSpectrum  # noqa
from .splines import RadialIntegralFromFunction, RadialIntegralSplinerBase  # noqa


# path that can be used with cmake to access the rascaline library and headers
_HERE = os.path.realpath(os.path.dirname(__file__))

_rascaline_cmake_prefix = os.path.realpath(os.path.join(_HERE, "..", "lib", "cmake"))
cmake_prefix_path = f"{_rascaline_cmake_prefix};{metatensor.utils.cmake_prefix_path}"
"""
Path containing the CMake configuration files for the underlying C library
"""
