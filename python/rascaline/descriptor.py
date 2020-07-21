# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from ctypes import c_double, c_char_p, POINTER

from ._rascaline import c_uintptr_t, rascal_indexes
from .clib import _get_library


class Descriptor:
    def __init__(self):
        self._lib = _get_library()
        self._as_parameter_ = self._lib.rascal_descriptor()
        try:
            self._as_parameter_.contents
        except ValueError:
            # TODO: better error message
            raise Exception("Got a NULL pointer")

    def __del__(self):
        self._lib.rascal_descriptor_free(self)
        self._as_parameter_ = 0

    @property
    def values(self):
        environments = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_values(self, data, environments, features)
        return np_array_view(
            data, (environments.value, features.value), dtype=np.float64
        )

    @property
    def gradients(self):
        environments = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_gradients(self, data, environments, features)
        return np_array_view(
            data, (environments.value, features.value), dtype=np.float64
        )

    def _indexes(self, kind, kind_name):
        count = c_uintptr_t()
        size = c_uintptr_t()
        data = POINTER(c_uintptr_t)()
        self._lib.rascal_descriptor_indexes(self, kind.value, data, count, size)
        data = np_array_view(data, (count.value, size.value), dtype=np.uintp)

        if count.value == 0:
            return []

        StringArray = c_char_p * size.value
        names = StringArray()
        self._lib.rascal_descriptor_indexes_names(self, kind.value, names, size)

        TupleType = namedtuple(kind_name, map(lambda n: n.decode("utf8"), names))
        return [TupleType(*values) for values in data]

    @property
    def environments(self):
        return self._indexes(
            rascal_indexes.RASCAL_INDEXES_ENVIRONMENTS, "EnvironmentIndex"
        )

    @property
    def features(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_FEATURES, "FeatureIndex")

    @property
    def gradients_environments(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_GRADIENTS, "GradientIndex")


def np_array_view(ptr, shape, dtype):
    assert len(shape) == 2
    if shape[0] != 0 and shape[1] != 0:
        return np.ctypeslib.as_array(ptr, shape=shape)
    else:
        data = np.array([], dtype=dtype)
        return data.reshape(shape)
