import importlib.metadata


__version__ = importlib.metadata.version("rascaline-torch")


from ._c_lib import _load_library


_load_library()

from . import utils  # noqa
from .calculator_base import CalculatorModule, register_autograd  # noqa

# don't forget to also update `rascaline/__init__.py` and
# `rascaline/torch/calculators.py` when modifying this file
from .calculators import (  # noqa
    AtomicComposition,
    LodeSphericalExpansion,
    NeighborList,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SortedDistances,
    SphericalExpansion,
    SphericalExpansionByPair,
)
from .system import System, systems_to_torch  # noqa


__all__ = [
    "AtomicComposition",
    "LodeSphericalExpansion",
    "NeighborList",
    "SoapPowerSpectrum",
    "SoapRadialSpectrum",
    "SortedDistances",
    "SphericalExpansion",
    "SphericalExpansionByPair",
]
