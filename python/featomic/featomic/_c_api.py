# fmt: off
# flake8: noqa
'''Automatically-generated file, do not edit!!!'''

import ctypes
import enum
import platform
from ctypes import CFUNCTYPE, POINTER

from metatensor._c_api import mts_labels_t, mts_tensormap_t
from numpy.ctypeslib import ndpointer


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

FEATOMIC_SUCCESS = 0
FEATOMIC_INVALID_PARAMETER_ERROR = 1
FEATOMIC_JSON_ERROR = 2
FEATOMIC_UTF8_ERROR = 3
FEATOMIC_SYSTEM_ERROR = 128
FEATOMIC_BUFFER_SIZE_ERROR = 254
FEATOMIC_INTERNAL_ERROR = 255
FEATOMIC_LOG_LEVEL_ERROR = 1
FEATOMIC_LOG_LEVEL_WARN = 2
FEATOMIC_LOG_LEVEL_INFO = 3
FEATOMIC_LOG_LEVEL_DEBUG = 4
FEATOMIC_LOG_LEVEL_TRACE = 5


featomic_status_t = ctypes.c_int32
featomic_logging_callback_t = CFUNCTYPE(None, ctypes.c_int32, ctypes.c_char_p)


class featomic_calculator_t(ctypes.Structure):
    pass


class featomic_pair_t(ctypes.Structure):
    _fields_ = [
        ("first", c_uintptr_t),
        ("second", c_uintptr_t),
        ("distance", ctypes.c_double),
        ("vector", ctypes.c_double * 3),
        ("cell_shift_indices", ctypes.c_int32 * 3),
    ]


class featomic_system_t(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_void_p),
        ("size", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, POINTER(c_uintptr_t))),
        ("types", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, POINTER(ndpointer(ctypes.c_int32, flags='C_CONTIGUOUS')))),
        ("positions", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, POINTER(ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')))),
        ("cell", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, POINTER(ctypes.c_double))),
        ("compute_neighbors", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, ctypes.c_double)),
        ("pairs", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, POINTER(ndpointer(featomic_pair_t, flags='C_CONTIGUOUS')), POINTER(c_uintptr_t))),
        ("pairs_containing", CFUNCTYPE(featomic_status_t, ctypes.c_void_p, c_uintptr_t, POINTER(ndpointer(featomic_pair_t, flags='C_CONTIGUOUS')), POINTER(c_uintptr_t))),
    ]


class featomic_labels_selection_t(ctypes.Structure):
    _fields_ = [
        ("subset", POINTER(mts_labels_t)),
        ("predefined", POINTER(mts_tensormap_t)),
    ]


class featomic_calculation_options_t(ctypes.Structure):
    _fields_ = [
        ("gradients", POINTER(ctypes.c_char_p)),
        ("gradients_count", c_uintptr_t),
        ("use_native_system", ctypes.c_bool),
        ("selected_samples", featomic_labels_selection_t),
        ("selected_properties", featomic_labels_selection_t),
        ("selected_keys", POINTER(mts_labels_t)),
    ]


def setup_functions(lib):
    from .status import _check_featomic_status_t

    lib.featomic_last_error.argtypes = [
        
    ]
    lib.featomic_last_error.restype = ctypes.c_char_p

    lib.featomic_set_logging_callback.argtypes = [
        featomic_logging_callback_t
    ]
    lib.featomic_set_logging_callback.restype = _check_featomic_status_t

    lib.featomic_calculator.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p
    ]
    lib.featomic_calculator.restype = POINTER(featomic_calculator_t)

    lib.featomic_calculator_free.argtypes = [
        POINTER(featomic_calculator_t)
    ]
    lib.featomic_calculator_free.restype = _check_featomic_status_t

    lib.featomic_calculator_name.argtypes = [
        POINTER(featomic_calculator_t),
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.featomic_calculator_name.restype = _check_featomic_status_t

    lib.featomic_calculator_parameters.argtypes = [
        POINTER(featomic_calculator_t),
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.featomic_calculator_parameters.restype = _check_featomic_status_t

    lib.featomic_calculator_cutoffs.argtypes = [
        POINTER(featomic_calculator_t),
        POINTER(POINTER(ctypes.c_double)),
        POINTER(c_uintptr_t)
    ]
    lib.featomic_calculator_cutoffs.restype = _check_featomic_status_t

    lib.featomic_calculator_compute.argtypes = [
        POINTER(featomic_calculator_t),
        POINTER(POINTER(mts_tensormap_t)),
        POINTER(featomic_system_t),
        c_uintptr_t,
        featomic_calculation_options_t
    ]
    lib.featomic_calculator_compute.restype = _check_featomic_status_t

    lib.featomic_profiling_clear.argtypes = [
        
    ]
    lib.featomic_profiling_clear.restype = _check_featomic_status_t

    lib.featomic_profiling_enable.argtypes = [
        ctypes.c_bool
    ]
    lib.featomic_profiling_enable.restype = _check_featomic_status_t

    lib.featomic_profiling_get.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        c_uintptr_t
    ]
    lib.featomic_profiling_get.restype = _check_featomic_status_t
