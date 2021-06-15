# -*- coding: utf-8 -*-
from .status import RascalError
from .log import set_logging_callback

from .descriptor import Descriptor

from .systems import SystemBase

from .calculators import CalculatorBase
from .calculators import SortedDistances
from .calculators import SphericalExpansion
from .calculators import SoapPowerSpectrum

# Get the __version__ attribute from setuptools metadata (which took it from
# Cargo.toml) cf https://stackoverflow.com/a/17638236/4692076
from pkg_resources import get_distribution, DistributionNotFound
import os

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
