# -*- coding: utf-8 -*-
import numpy as np
from ctypes import c_double, c_char_p, POINTER, ARRAY

from ._rascaline import c_uintptr_t, rascal_indexes
from .clib import _get_library
from .status import _check_rascal_pointer


class Indexes(np.ndarray):
    """
    Small wrapper around `numpy.ndarray` that adds a `names` attribute
    containing the names of the indexes.
    """

    def __new__(cls, ptr, shape, names):
        assert len(shape) == 2
        assert len(names) == shape[1]

        dtype = [(name, np.float64) for name in names]
        if ptr is not None:
            array = np.ctypeslib.as_array(ptr, shape=shape)
            array.flags.writeable = False
            # view the array as a numpy structured array containing multiple
            # entries
            array = array.view(dtype=dtype).reshape((shape[0],))
        else:
            array = np.array([], dtype=dtype)

        obj = array.view(cls)
        obj.names = tuple(names)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", tuple())


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

        StringArray = c_char_p * size.value
        names = StringArray()
        self._lib.rascal_descriptor_indexes_names(self, kind.value, names, size)
        names = list(map(lambda n: n.decode("utf8"), names))

        shape = (count.value, size.value)
        ptr = data if count.value != 0 else None
        return Indexes(ptr=ptr, shape=shape, names=names)

    @property
    def environments(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_ENVIRONMENTS)

    @property
    def features(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_FEATURES)

    @property
    def gradients_environments(self):
        return self._indexes(rascal_indexes.RASCAL_INDEXES_GRADIENTS)

    def densify(self, variables):
        if isinstance(variables, str):
            variables = [variables]

        c_variables = ARRAY(c_char_p, len(variables))()
        for i, v in enumerate(variables):
            c_variables[i] = v.encode("utf8")
        self._lib.rascal_descriptor_densify(self, c_variables, c_variables._length_)


def np_array_view(ptr, shape, dtype):
    assert len(shape) == 2
    if shape[0] != 0 and shape[1] != 0:
        array = np.ctypeslib.as_array(ptr, shape=shape)
        array.flags.writeable = False
        return array
    else:
        data = np.array([], dtype=dtype)
        return data.reshape(shape)
