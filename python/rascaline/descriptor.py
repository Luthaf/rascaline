# -*- coding: utf-8 -*-
import numpy as np
from ctypes import c_double, c_char_p, POINTER

from ._rascaline import c_uintptr_t, rascal_indexes
from .clib import _get_library
from .status import _check_rascal_pointer


class Descriptor:
    def __init__(self):
        self._lib = _get_library()
        self._as_parameter_ = self._lib.rascal_descriptor()
        _check_rascal_pointer(self._as_parameter_)

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

        if not data:
            return None

        return np_array_view(
            data, (environments.value, features.value), dtype=np.float64
        )

    def _indexes(self, kind):
        count = c_uintptr_t()
        size = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_indexes(self, kind.value, data, count, size)

        if count.value == 0:
            return []

        StringArray = c_char_p * size.value
        names = StringArray()
        self._lib.rascal_descriptor_indexes_names(self, kind.value, names, size)

        dtype = [(name, np.float64) for name in map(lambda n: n.decode("utf8"), names)]
        return np_array_view(data, (count.value, size.value), dtype=dtype)

    @property
    def environments(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_ENVIRONMENTS)

    @property
    def features(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_FEATURES)

    @property
    def gradients_environments(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_GRADIENTS)

    def densify(self, variable):
        self._lib.rascal_descriptor_densify(self, variable.encode("utf8"))


def np_array_view(ptr, shape, dtype):
    assert len(shape) == 2
    if shape[0] != 0 and shape[1] != 0:
        array = np.ctypeslib.as_array(ptr, shape=shape)
        array.flags.writeable = False
        if isinstance(dtype, list):
            # view the array as a numpy structured array containing multiple
            # entries
            assert len(dtype) == shape[1]
            return array.view(dtype=dtype).reshape((shape[0],))
        else:
            return array
    else:
        data = np.array([], dtype=dtype)
        return data.reshape(shape)
