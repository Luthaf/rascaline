import importlib.metadata


__version__ = importlib.metadata.version("rascaline-torch")


from ._c_lib import _load_library


_load_library()

from . import utils  # noqa: E402, F401
from .calculator_base import CalculatorModule, register_autograd  # noqa: E402, F401

# don't forget to also update `rascaline/__init__.py` and
# `rascaline/torch/calculators.py` when modifying this file
from .calculators import (  # noqa: E402, F401
    AtomicComposition,
    LodeSphericalExpansion,
    NeighborList,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SortedDistances,
    SphericalExpansion,
    SphericalExpansionByPair,
)
from .system import systems_to_torch  # noqa: E402, F401


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
