# -*- coding: utf-8 -*-
import numpy as np
from ctypes import c_double, c_int32, c_char_p, POINTER, ARRAY

from ._rascaline import c_uintptr_t, rascal_indexes
from .clib import _get_library
from .status import _check_rascal_pointer


class Indexes(np.ndarray):
    """
    This is a small wrapper around ``numpy.ndarray`` that adds a ``names``
    attribute containing the names of the indexes.

    .. py:attribute:: name
        :type: Tuple[str]

        name of each column in this indexes array
    """

    def __new__(cls, ptr, shape, names, index_callback):
        assert len(shape) == 2
        assert len(names) == shape[1]

        dtype = [(name, np.int32) for name in names]
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
        obj._index_callback = index_callback
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", tuple())
        self._index_callback = getattr(obj, "_index_callback", lambda _: None)

    def index(self, value):
        """
        Get the index associated with the given value in this array, or ``None``
        if the value is not inside this array.
        """
        return self._index_callback(value)


class Descriptor:
    """
    Descriptors store the result of a single calculation on a set of systems.

    They contains the values produced by the calculation; as well as metdata to
    interpret these values. In particular, it contains two additional named
    arrays describing the ``samples`` (associated with rows in the values) and
    ``features`` (associated with columns in the values).

    Optionally, a descriptor can also contain gradients of the samples. In this
    case, some additional metadata is available to describe the rows of the
    gradients array in ``gradients_samples``. The columns of the gradients array
    are still described by the same ``features``.
    """

    def __init__(self):
        self._lib = _get_library()
        self._as_parameter_ = self._lib.rascal_descriptor()
        _check_rascal_pointer(self._as_parameter_)

    def __del__(self):
        self._lib.rascal_descriptor_free(self)
        self._as_parameter_ = 0

    @property
    def values(self):
        """
        The values stored in this descriptor by a calculator, as a **read only**
        2D numpy ndarray with ``dtype=np.float64``.
        """
        samples = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_values(self, data, samples, features)
        return np_array_view(data, (samples.value, features.value), dtype=np.float64)

    @property
    def gradients(self):
        """
        The gradients stored in this descriptor by a calculator, as a **read
        only** 2D numpy ndarray with ``dtype=np.float64``, or ``None`` if no
        value was stored.
        """
        samples = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_gradients(self, data, samples, features)

        if not data:
            return None

        return np_array_view(data, (samples.value, features.value), dtype=np.float64)

    def _indexes(self, kind):
        count = c_uintptr_t()
        size = c_uintptr_t()
        data = POINTER(c_int32)()
        self._lib.rascal_descriptor_indexes(self, kind.value, data, count, size)

        StringArray = c_char_p * size.value
        names = StringArray()
        self._lib.rascal_descriptor_indexes_names(self, kind.value, names, size)
        names = list(map(lambda n: n.decode("utf8"), names))

        shape = (count.value, size.value)
        ptr = data if count.value != 0 else None

        def index_callback(value):
            value = np.array(value)
            assert len(value.shape) == 1
            if value.dtype.fields is not None:
                # convert structured array back to int32 array
                value = value.view(dtype=np.int32).reshape((1, -1))
            else:
                value = np.array(value, dtype=np.int32)

            ptr_int32 = POINTER(c_int32)
            output = c_int32()
            self._lib.rascal_descriptor_indexes_position(
                self, kind.value, value.ctypes.data_as(ptr_int32), len(value), output
            )
            if output.value == -1:
                return None
            else:
                return output.value

        return Indexes(ptr=ptr, shape=shape, names=names, index_callback=index_callback)

    @property
    def samples(self):
        """
        Metdata describing the samples/rows in :py:attr:`Descriptor.values`.

        This is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes.RASCAL_INDEXES_SAMPLES)

    @property
    def features(self):
        """
        Metdata describing the features/columns in :py:attr:`Descriptor.values`
        and :py:attr:`Descriptor.gradients`.

        This is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes.RASCAL_INDEXES_FEATURES)

    @property
    def gradients_samples(self):
        """
        Metdata describing the rows in :py:attr:`Descriptor.gradients`.

        If there are no gradients stored in this descriptor,
        ``gradients_samples`` is ``None``.

        This is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes.RASCAL_INDEXES_GRADIENT_SAMPLES)

    def densify(self, variables):
        """
        Move the values with names in ``variables`` from samples to features.

        :param variables: names of the variables to move
        :type variables: str | list[str]
        """
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
