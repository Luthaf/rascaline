import os

import torch

from ._c_lib import parse_version


_HERE = os.path.dirname(__file__)


_TORCH_VERSION = parse_version(torch.__version__)
install_prefix = os.path.join(
    _HERE, f"torch-{_TORCH_VERSION.major}.{_TORCH_VERSION.minor}"
)

cmake_prefix_path = os.path.join(install_prefix, "lib", "cmake")
"""
Path containing the CMake configuration files for the underlying C++ library
"""
