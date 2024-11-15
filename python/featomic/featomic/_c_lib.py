import os
import sys
from ctypes import cdll

import metatensor

from ._c_api import setup_functions
from .log import _set_logging_callback_impl, default_logging_callback


_HERE = os.path.realpath(os.path.dirname(__file__))


class FeatomicFinder(object):
    def __init__(self):
        self._cached_dll = None

    def __call__(self):
        if self._cached_dll is None:
            # Load metatensor shared library in the process first, to ensure
            # the featomic shared library can find it
            metatensor._c_lib._get_library()

            path = _lib_path()

            self._cached_dll = cdll.LoadLibrary(path)
            setup_functions(self._cached_dll)
            _set_logging_callback_impl(self._cached_dll, default_logging_callback)
        return self._cached_dll


def _lib_path():
    if sys.platform.startswith("darwin"):
        windows = False
        path = os.path.join(_HERE, "lib", "libfeatomic.dylib")
    elif sys.platform.startswith("linux"):
        windows = False
        path = os.path.join(_HERE, "lib", "libfeatomic.so")
    elif sys.platform.startswith("win"):
        windows = True
        path = os.path.join(_HERE, "bin", "featomic.dll")
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find featomic shared library at " + path)


def _check_dll(path):
    """Check if the DLL at ``path`` matches the architecture of Python"""
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404
    IMAGE_FILE_MACHINE_ARM64 = 43620

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    python_machine = platform.machine()
    if python_machine == "x86":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit x86, but featomic.dll is not")
    elif python_machine == "AMD64":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit x86_64, but featomic.dll is not")
    elif python_machine == "ARM64":
        if machine != IMAGE_FILE_MACHINE_ARM64:
            raise ImportError("Python is 64-bit ARM, but featomic.dll is not")
    else:
        raise ImportError(
            f"Featomic doesn't provide a version for {python_machine} CPU. "
            "If you are compiling from source on a new architecture, edit this file"
        )


_get_library = FeatomicFinder()
