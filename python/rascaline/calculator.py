# -*- coding: utf-8 -*-
import json
import ctypes
import numpy as np

from ._rascaline import rascal_system_t, rascal_status_t, c_uintptr_t
from .clib import _get_library
from .status import _check_rascal_pointer, RascalError
from .descriptor import Descriptor


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

    if np.can_cast(indexes.dtype, np.uintp, 'safe'):
        raise ValueError(f"selected {kind} array must contain integer values")


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

    def compute(self, systems, descriptor=None):
        if descriptor is None:
            descriptor = Descriptor()

        if not isinstance(systems, list):
            systems = [systems]

        c_systems = (rascal_system_t * len(systems))(
            *list(s._as_rascal_system_t() for s in systems)
        )
        self._lib.rascal_calculator_compute(self, descriptor, c_systems, len(systems))
        return descriptor

    def compute_partial(self, systems, descriptor=None, samples=None, features=None):
        if descriptor is None:
            descriptor = Descriptor()

        if not isinstance(systems, list):
            systems = [systems]

        c_systems = (rascal_system_t * len(systems))(
            *list(s._as_rascal_system_t() for s in systems)
        )

        if samples is not None:
            samples = np.array(samples)
            if samples.dtype.fields is not None:
                # convert structured array back to uintptr_t array
                size = len(samples)
                samples = samples.view(dtype=np.uintp).reshape((size, -1))
            else:
                _check_selected_indexes(samples, "samples")
                samples = np.array(samples, dtype=np.uintp)

        if features is not None:
            features = np.array(features)
            if features.dtype.fields is not None:
                # convert structured array back to uintptr_t array
                size = len(features)
                features = features.view(dtype=np.uintp).reshape((size, -1))
            else:
                _check_selected_indexes(features, "features")
                features = np.array(features, dtype=np.uintp)

        samples_count = 0 if samples is None else samples.size
        features_count = 0 if features is None else features.size

        ptr_type = ctypes.POINTER(c_uintptr_t)
        self._lib.rascal_calculator_compute_partial(
            self,
            descriptor,
            c_systems,
            len(systems),
            samples.ctypes.data_as(ptr_type) if samples is not None else None,
            samples_count,
            features.ctypes.data_as(ptr_type) if features is not None else None,
            features_count,
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
