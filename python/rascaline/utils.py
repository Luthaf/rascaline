import ctypes
import os

import equistore

from ._c_api import RASCAL_BUFFER_SIZE_ERROR
from .status import RascalError


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except RascalError as e:
            if e.status == RASCAL_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


# path that can be used with cmake to access the rascaline library and headers
_HERE = os.path.realpath(os.path.dirname(__file__))
cmake_prefix_path = (
    f"{os.path.join(_HERE, 'lib', 'cmake')};{equistore.utils.cmake_prefix_path}"
)
"""
Path containing the CMake configuration files for the underlying C library
"""
