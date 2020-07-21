# -*- coding: utf-8 -*-
'''
Automatically-generated file, do not edit!!!
'''
# flake8: noqa

import enum
import platform

import ctypes
from ctypes import POINTER, CFUNCTYPE
from numpy.ctypeslib import ndpointer

arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64


class rascal_indexes(enum.Enum):
    RASCAL_INDEXES_FEATURES = 0
    RASCAL_INDEXES_ENVIRONMENTS = 1
    RASCAL_INDEXES_GRADIENTS = 2


class rascal_calculator_t(ctypes.Structure):
    pass


class rascal_descriptor_t(ctypes.Structure):
    pass


class rascal_pair_t(ctypes.Structure):
    _fields_ = [
        ("first", c_uintptr_t),
        ("second", c_uintptr_t),
        ("distance", ctypes.c_double),
    ]


class rascal_system_t(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_void_p),
        ("size", CFUNCTYPE(None, ctypes.c_void_p, POINTER(c_uintptr_t))),
        ("species", CFUNCTYPE(None, ctypes.c_void_p, POINTER(ndpointer(c_uintptr_t, flags='C_CONTIGUOUS')))),
        ("positions", CFUNCTYPE(None, ctypes.c_void_p, POINTER(ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')))),
        ("cell", CFUNCTYPE(None, ctypes.c_void_p, POINTER(ctypes.c_double))),
        ("compute_neighbors", CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_double)),
        ("pairs", CFUNCTYPE(None, ctypes.c_void_p, POINTER(ndpointer(rascal_pair_t, flags='C_CONTIGUOUS')), POINTER(c_uintptr_t))),
    ]


def setup_functions(lib):

    lib.rascal_calculator.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p
    ]
    lib.rascal_calculator.restype = POINTER(rascal_calculator_t)

    lib.rascal_calculator_compute.argtypes = [
        POINTER(rascal_calculator_t),
        POINTER(rascal_descriptor_t),
        POINTER(rascal_system_t),
        c_uintptr_t
    ]
    lib.rascal_calculator_compute.restype = None

    lib.rascal_calculator_free.argtypes = [
        POINTER(rascal_calculator_t)
    ]
    lib.rascal_calculator_free.restype = None

    lib.rascal_calculator_name.argtypes = [
        POINTER(rascal_calculator_t),
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.rascal_calculator_name.restype = None

    lib.rascal_descriptor.argtypes = [
        
    ]
    lib.rascal_descriptor.restype = POINTER(rascal_descriptor_t)

    lib.rascal_descriptor_free.argtypes = [
        POINTER(rascal_descriptor_t)
    ]
    lib.rascal_descriptor_free.restype = None

    lib.rascal_descriptor_gradients.argtypes = [
        POINTER(rascal_descriptor_t),
        POINTER(POINTER(ctypes.c_double)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_gradients.restype = None

    lib.rascal_descriptor_indexes.argtypes = [
        POINTER(rascal_descriptor_t),
        ctypes.c_int,
        POINTER(POINTER(c_uintptr_t)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_indexes.restype = None

    lib.rascal_descriptor_indexes_names.argtypes = [
        POINTER(rascal_descriptor_t),
        ctypes.c_int,
        POINTER(ctypes.c_char_p),
        c_uintptr_t
    ]
    lib.rascal_descriptor_indexes_names.restype = None

    lib.rascal_descriptor_values.argtypes = [
        POINTER(rascal_descriptor_t),
        POINTER(POINTER(ctypes.c_double)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_values.restype = None
