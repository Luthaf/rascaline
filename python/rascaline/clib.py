# -* coding: utf-8 -*
import os
import sys
from ctypes import cdll

from ._rascaline import setup_functions


class RascalFinder(object):
    def __init__(self):
        self._cache = None

    def __call__(self):
        if self._cache is None:
            path = _lib_path()
            self._cache = cdll.LoadLibrary(path)
            setup_functions(self._cache)
        return self._cache


def _lib_path():
    if sys.platform.startswith("darwin"):
        windows = False
        name = "librascaline.dylib"
    elif sys.platform.startswith("linux"):
        windows = False
        name = "librascaline.so"
    elif sys.platform.startswith("win"):
        windows = True
        name = "librascaline.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")

    path = os.path.join(os.path.dirname(__file__), name)
    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find rascaline shared library at " + path)


def _check_dll(path):
    '''
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    '''
    import struct
    import platform

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

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

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but rascaline.dll is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but rascaline.dll is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


_get_library = RascalFinder()
