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

# === WARNING: manual type definitions
# === Make sure these are kept in sync with rascaline.h
arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

rascal_status_t = ctypes.c_int32

rascal_logging_callback_t = CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
# === end of manual type definitions


RASCAL_SUCCESS = 0
RASCAL_INVALID_PARAMETER_ERROR = 1
RASCAL_JSON_ERROR = 2
RASCAL_UTF8_ERROR = 3
RASCAL_CHEMFILES_ERROR = 4
RASCAL_SYSTEM_ERROR = 128
RASCAL_INTERNAL_ERROR = 255
RASCAL_LOG_LEVEL_ERROR = 1
RASCAL_LOG_LEVEL_WARN = 2
RASCAL_LOG_LEVEL_INFO = 3
RASCAL_LOG_LEVEL_DEBUG = 4
RASCAL_LOG_LEVEL_TRACE = 5


class rascal_indexes(enum.Enum):
    RASCAL_INDEXES_FEATURES = 0
    RASCAL_INDEXES_SAMPLES = 1
    RASCAL_INDEXES_GRADIENT_SAMPLES = 2


class rascal_calculator_t(ctypes.Structure):
    pass


class rascal_descriptor_t(ctypes.Structure):
    pass


class rascal_pair_t(ctypes.Structure):
    _fields_ = [
        ("first", c_uintptr_t),
        ("second", c_uintptr_t),
        ("distance", ctypes.c_double),
        ("vector", ctypes.c_double * 3),
    ]


class rascal_system_t(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_void_p),
        ("size", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, POINTER(c_uintptr_t))),
        ("species", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, POINTER(ndpointer(c_uintptr_t, flags='C_CONTIGUOUS')))),
        ("positions", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, POINTER(ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')))),
        ("cell", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, POINTER(ctypes.c_double))),
        ("compute_neighbors", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, ctypes.c_double)),
        ("pairs", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, POINTER(ndpointer(rascal_pair_t, flags='C_CONTIGUOUS')), POINTER(c_uintptr_t))),
        ("pairs_containing", CFUNCTYPE(rascal_status_t, ctypes.c_void_p, c_uintptr_t, POINTER(ndpointer(rascal_pair_t, flags='C_CONTIGUOUS')), POINTER(c_uintptr_t))),
    ]


class rascal_calculation_options_t(ctypes.Structure):
    _fields_ = [
        ("use_native_system", ctypes.c_bool),
        ("selected_samples", POINTER(ctypes.c_int32)),
        ("selected_samples_count", c_uintptr_t),
        ("selected_features", POINTER(ctypes.c_int32)),
        ("selected_features_count", c_uintptr_t),
    ]


def setup_functions(lib):
    from .status import _check_rascal_status_t

    lib.rascal_last_error.argtypes = [
        
    ]
    lib.rascal_last_error.restype = ctypes.c_char_p

    lib.rascal_set_logging_callback.argtypes = [
        rascal_logging_callback_t
    ]
    lib.rascal_set_logging_callback.restype = _check_rascal_status_t

    lib.rascal_basic_systems_read.argtypes = [
        ctypes.c_char_p,
        POINTER(POINTER(rascal_system_t)),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_basic_systems_read.restype = _check_rascal_status_t

    lib.rascal_basic_systems_free.argtypes = [
        POINTER(rascal_system_t),
        c_uintptr_t
    ]
    lib.rascal_basic_systems_free.restype = _check_rascal_status_t

    lib.rascal_descriptor.argtypes = [
        
    ]
    lib.rascal_descriptor.restype = POINTER(rascal_descriptor_t)

    lib.rascal_descriptor_free.argtypes = [
        POINTER(rascal_descriptor_t)
    ]
    lib.rascal_descriptor_free.restype = _check_rascal_status_t

    lib.rascal_descriptor_values.argtypes = [
        POINTER(rascal_descriptor_t),
        POINTER(POINTER(ctypes.c_double)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_values.restype = _check_rascal_status_t

    lib.rascal_descriptor_gradients.argtypes = [
        POINTER(rascal_descriptor_t),
        POINTER(POINTER(ctypes.c_double)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_gradients.restype = _check_rascal_status_t

    lib.rascal_descriptor_indexes.argtypes = [
        POINTER(rascal_descriptor_t),
        ctypes.c_int,
        POINTER(POINTER(ctypes.c_int32)),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t)
    ]
    lib.rascal_descriptor_indexes.restype = _check_rascal_status_t

    lib.rascal_descriptor_indexes_names.argtypes = [
        POINTER(rascal_descriptor_t),
        ctypes.c_int,
        POINTER(ctypes.c_char_p),
        c_uintptr_t
    ]
    lib.rascal_descriptor_indexes_names.restype = _check_rascal_status_t

    lib.rascal_descriptor_densify.argtypes = [
        POINTER(rascal_descriptor_t),
        POINTER(ctypes.c_char_p),
        c_uintptr_t
    ]
    lib.rascal_descriptor_densify.restype = _check_rascal_status_t

    lib.rascal_calculator.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p
    ]
    lib.rascal_calculator.restype = POINTER(rascal_calculator_t)

    lib.rascal_calculator_free.argtypes = [
        POINTER(rascal_calculator_t)
    ]
    lib.rascal_calculator_free.restype = _check_rascal_status_t

    lib.rascal_calculator_name.argtypes = [
        POINTER(rascal_calculator_t),
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.rascal_calculator_name.restype = _check_rascal_status_t

    lib.rascal_calculator_parameters.argtypes = [
        POINTER(rascal_calculator_t),
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.rascal_calculator_parameters.restype = _check_rascal_status_t

    lib.rascal_calculator_compute.argtypes = [
        POINTER(rascal_calculator_t),
        POINTER(rascal_descriptor_t),
        POINTER(rascal_system_t),
        c_uintptr_t,
        rascal_calculation_options_t
    ]
    lib.rascal_calculator_compute.restype = _check_rascal_status_t
