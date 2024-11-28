import os


_HERE = os.path.dirname(__file__)

cmake_prefix_path = os.path.realpath(os.path.join(_HERE, "lib", "cmake"))
"""
Path containing the CMake configuration files for the underlying C library
"""
