# -*- coding: utf-8 -*-
import json
import ctypes

from ._rascaline import rascal_system_t
from .clib import _get_library
from .descriptor import Descriptor


class CalculatorBase:
    def __init__(self, __rascal__name, **kwargs):
        self._lib = _get_library()
        parameters = json.dumps(kwargs).encode("utf8")
        self._as_parameter_ = self._lib.rascal_calculator(
            __rascal__name.encode("utf8"), parameters
        )
        try:
            self._as_parameter_.contents
        except ValueError:
            # TODO: better error message
            raise Exception("Got a NULL pointer")

    def __del__(self):
        self._lib.rascal_calculator_free(self)
        self._as_parameter_ = 0

    @property
    def name(self):
        bufflen = 1024
        buffer = ctypes.create_string_buffer(bufflen)
        self._lib.rascal_calculator_name(self, buffer, bufflen)
        return buffer.value.decode("utf8")

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
