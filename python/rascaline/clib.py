# -* coding: utf-8 -*
import os
import sys
import warnings
import ctypes
from ctypes import cdll
from .log_utils import (DEFAULT_LOG_CALLBACK, DEFAULT_LOG_LEVEL)

from ._rascaline import setup_functions

class RascalFinder(object):
    def __init__(self):
        self._cache = None

    def __call__(self):
        if self._cache is None:
            path = _lib_path()
            self._cache = cdll.LoadLibrary(path)
            setup_functions(self._cache)
            _set_default_logging_callback()
        return self._cache

def _set_default_logging_callback():
    '''
    Default logging function. For now it is only a print,
    but it can be replaced by a more sophisticated logging utility
    '''
    set_logging_callback(DEFAULT_LOG_CALLBACK, DEFAULT_LOG_LEVEL)

def set_logging_callback(callback_function, log_level):
    '''
    Call `function` on every logging event. The callback should take a string
    message and return nothing.
    '''

    def callback_wrapper(log_level, message):
        try:
            callback_function(log_level, message.decode("utf8"))
        except Exception as e:
            message = "exception raised in logging callback: {}".format(e)
            warnings.warn(message, ResourceWarning)

    global _CURRENT_CALLBACK
    _CURRENT_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)(callback_wrapper)
    _get_library().rascal_set_logging_callback(_CURRENT_CALLBACK, log_level)

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

_CURRENT_CALLBACK = None
_get_library = RascalFinder()
