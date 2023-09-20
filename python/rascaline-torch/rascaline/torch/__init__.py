import sys


if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("rascaline-torch")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("rascaline-torch").version


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
