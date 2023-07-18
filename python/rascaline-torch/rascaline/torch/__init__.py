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
from .calculator_base import CalculatorModule  # noqa
from .system import System, systems_to_torch  # noqa


# don't forget to also update `rascaline/__init__.py` and
# `rascaline/torch/calculators.py` when modifying this file
from .calculators import AtomicComposition  # noqa  isort: skip
from .calculators import SortedDistances  # noqa  isort: skip
from .calculators import NeighborList  # noqa  isort: skip
from .calculators import LodeSphericalExpansion  # noqa isort: skip
from .calculators import SphericalExpansion  # noqa  isort: skip
from .calculators import SphericalExpansionByPair  # noqa  isort: skip
from .calculators import SoapRadialSpectrum  # noqa  isort: skip
from .calculators import SoapPowerSpectrum  # noqa  isort: skip
