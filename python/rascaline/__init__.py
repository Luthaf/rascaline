import os

from pkg_resources import DistributionNotFound, get_distribution

from .log import set_logging_callback  # noqa
from .profiling import Profiler  # noqa
from .status import RascalError  # noqa
from .systems import IntoSystem, SystemBase  # noqa


from .calculators import CalculatorBase  # noqa  isort: skip
from .calculators import AtomicComposition  # noqa  isort: skip
from .calculators import SortedDistances  # noqa  isort: skip
from .calculators import NeighborList  # noqa  isort: skip
from .calculators import LodeSphericalExpansion  # noqa isort: skip
from .calculators import SphericalExpansion  # noqa  isort: skip
from .calculators import SphericalExpansionByPair  # noqa  isort: skip
from .calculators import SoapRadialSpectrum  # noqa  isort: skip
from .calculators import SoapPowerSpectrum  # noqa  isort: skip

from .splines import generate_splines  # noqa  isort: skip


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
