import os

import metatensor

from .clebsch_gordan import *  # noqa
from .power_spectrum import PowerSpectrum  # noqa
from .splines import (  # noqa
    AtomicDensityBase,
    DeltaDensity,
    GaussianDensity,
    GtoBasis,
    LodeDensity,
    LodeSpliner,
    MonomialBasis,
    RadialBasisBase,
    RadialIntegralFromFunction,
    RadialIntegralSplinerBase,
    SoapSpliner,
    SphericalBesselBasis,
)


_HERE = os.path.dirname(__file__)

cmake_prefix_path = os.path.realpath(os.path.join(_HERE, "..", "lib", "cmake"))
"""
Path containing the CMake configuration files for the underlying C library
"""
