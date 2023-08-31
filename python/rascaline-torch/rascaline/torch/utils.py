import os

import metatensor.torch

import rascaline


_HERE = os.path.realpath(os.path.dirname(__file__))

_rascaline_torch_cmake_prefix = os.path.join(os.path.dirname(__file__), "lib", "cmake")

cmake_prefix_path = ";".join(
    [
        _rascaline_torch_cmake_prefix,
        rascaline.utils.cmake_prefix_path,
        metatensor.torch.utils.cmake_prefix_path,
    ]
)
"""
Path containing the CMake configuration files for the underlying C library
"""
