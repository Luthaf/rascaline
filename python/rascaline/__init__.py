# -*- coding: utf-8 -*-
import os

from pkg_resources import DistributionNotFound, get_distribution

from .calculators import CalculatorBase  # noqa
from .calculators import SoapPowerSpectrum  # noqa
from .calculators import SortedDistances  # noqa
from .calculators import SphericalExpansion  # noqa
from .calculators import LodeSphericalExpansion  # noqa

from .log import set_logging_callback  # noqa
from .profiling import Profiler  # noqa
from .status import RascalError  # noqa
from .systems import IntoSystem, SystemBase  # noqa


# Get the __version__ attribute from setuptools metadata (which took it from
# Cargo.toml) cf https://stackoverflow.com/a/17638236/4692076
try:
    dist = get_distribution("rascaline")
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, "rascaline")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = "dev"
else:
    __version__ = dist.version
