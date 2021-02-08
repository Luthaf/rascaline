# -*- coding: utf-8 -*-
import json
import ctypes
import numpy as np

from ._rascaline import rascal_system_t, rascal_status_t, rascal_calculation_options_t
from .clib import _get_library
from .status import _check_rascal_pointer, RascalError
from .descriptor import Descriptor
from .systems import wrap_system


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except RascalError as e:
            if (
                e.status == rascal_status_t.RASCAL_INVALID_PARAMETER_ERROR.value
                and "string buffer is not big enough" in e.args[0]
            ):
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def _check_selected_indexes(indexes, kind):
    if len(indexes.shape) != 2:
        raise ValueError(f"selected {kind} array must be a two-dimensional array")

    if not np.can_cast(indexes.dtype, np.float64, "safe"):
        raise ValueError(f"selected {kind} array must contain float64 values")


def _is_iterable(object):
    try:
        for e in object:
            pass
        return True
    except TypeError:
        return False


def _convert_systems(systems):
    try:
        return (rascal_system_t * 1)(wrap_system(systems)._as_rascal_system_t())
    except TypeError:
        # try iterating over the systems
        return (rascal_system_t * len(systems))(*list(wrap_system(s) for s in systems))


def _options_to_c(options):
    known_options = [
        "use_native_system",
        "selected_samples",
        "selected_features",
    ]
    for option in options.keys():
        if option not in known_options:
            raise Exception(f"unknown option '{option}' passed to compute")

    samples = options.get("selected_samples")
    if samples is not None:
        samples = np.array(samples)
        if samples.dtype.fields is not None:
            # convert structured array back to float64 array
            size = len(samples)
            samples = samples.view(dtype=np.float64).reshape((size, -1))
        else:
            _check_selected_indexes(samples, "samples")
            samples = np.array(samples, dtype=np.float64)

    features = options.get("selected_features")
    if features is not None:
        features = np.array(features)
        if features.dtype.fields is not None:
            # convert structured array back to float64 array
            size = len(features)
            features = features.view(dtype=np.float64).reshape((size, -1))
        else:
            _check_selected_indexes(features, "features")
            features = np.array(features, dtype=np.float64)

    ptr_double = ctypes.POINTER(ctypes.c_double)
    c_options = rascal_calculation_options_t()
    c_options.use_native_system = bool(options.get("use_native_system", False))

    if samples is None:
        c_options.selected_samples = None
        c_options.selected_samples_count = 0
    else:
        c_options.selected_samples = samples.ctypes.data_as(ptr_double)
        c_options.selected_samples_count = samples.size

    if features is None:
        c_options.selected_features = None
        c_options.selected_features_count = 0
    else:
        c_options.selected_features = features.ctypes.data_as(ptr_double)
        c_options.selected_features_count = features.size

    return c_options


class CalculatorBase:
    def __init__(self, __rascal__name, **kwargs):
        self._lib = _get_library()
        parameters = json.dumps(kwargs).encode("utf8")
        self._as_parameter_ = self._lib.rascal_calculator(
            __rascal__name.encode("utf8"), parameters
        )
        _check_rascal_pointer(self._as_parameter_)

    def __del__(self):
        self._lib.rascal_calculator_free(self)
        self._as_parameter_ = 0

    @property
    def name(self):
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_name(
                self, buffer, bufflen
            )
        )

    def parameters(self):
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_parameters(
                self, buffer, bufflen
            )
        )

    def compute(self, systems, descriptor=None, **kwargs):
        if descriptor is None:
            descriptor = Descriptor()

        c_systems = _convert_systems(systems)
        self._lib.rascal_calculator_compute(
            self, descriptor, c_systems, c_systems._length_, _options_to_c(kwargs)
        )
        return descriptor


class DummyCalculator(CalculatorBase):
    def __init__(self, cutoff, delta, name, gradients):
        parameters = {
            "cutoff": cutoff,
            "delta": delta,
            "name": name,
            "gradients": gradients,
        }
        super().__init__("dummy_calculator", **parameters)


class SortedDistances(CalculatorBase):
    def __init__(self, cutoff, max_neighbors):
        parameters = {"cutoff": cutoff, "max_neighbors": max_neighbors}
        super().__init__("sorted_distances", **parameters)


class SphericalExpansion(CalculatorBase):
    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        radial_basis,
        gradients,
        cutoff_function,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "radial_basis": radial_basis,
            "gradients": gradients,
            "cutoff_function": cutoff_function,
        }
        super().__init__("spherical_expansion", **parameters)
