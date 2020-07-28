# -*- coding: utf-8 -*-
import json
import ctypes

from ._rascaline import rascal_system_t, rascal_status_t
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
