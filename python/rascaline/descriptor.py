# -*- coding: utf-8 -*-
from ctypes import ARRAY, POINTER, c_char_p, c_double, c_int32, pointer

import numpy as np

from ._rascaline import (
    c_uintptr_t,
    rascal_densified_position_t,
    rascal_indexes_kind,
    rascal_indexes_t,
)
from .clib import _get_library
from .status import _check_rascal_pointer


class Indexes(np.ndarray):
    """Wrapper for ``numpy.ndarray`` adding ``names`` attribute containing indices names.

    .. py:attribute:: name
        :type: Tuple[str]

        name of each column in this indexes array
    """

    def __new__(cls, names, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("array parameter must be a numpy ndarray")

        if len(array.shape) != 2 or array.dtype != np.int32:
            raise ValueError("array parameter must be a 2D array of 32-bit integers")

        names = tuple(str(n) for n in names)

        if len(names) != array.shape[1]:
            raise ValueError(
                "names parameter must have an entry for each column of the array"
            )

        if array.shape != (0, 0):
            dtype = [(name, np.int32) for name in names]
            array = array.view(dtype=dtype).reshape((array.shape[0],))

        obj = array.view(cls)
        obj.names = names
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", tuple())


class Descriptor:
    """Descriptors store the result of a single calculation on a set of systems.

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
        if hasattr(self, "_lib"):
            # if we failed to load the lib, don't double error by trying to call
            # ``self._lib.rascal_descriptor_free``
            self._lib.rascal_descriptor_free(self)
        self._as_parameter_ = 0

    @property
    def values(self):
        """The values stored in this descriptor by a calculator.

        Values are stored as a **read only** 2D numpy ndarray with ``dtype=np.float64``.
        """
        samples = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_values(self, data, samples, features)

        return _ptr_to_ndarray(
            ptr=data,
            shape=(samples.value, features.value),
            dtype=np.float64,
        )

    @property
    def gradients(self):
        """The gradients stored in this descriptor by a calculator.

        Gradients are stored as a **read only** 2D numpy ndarray
        with ``dtype=np.float64``, or ``None`` if no value was stored.
        """
        samples = c_uintptr_t()
        features = c_uintptr_t()
        data = POINTER(c_double)()
        self._lib.rascal_descriptor_gradients(self, data, samples, features)

        if not data:
            return None

        return _ptr_to_ndarray(
            ptr=data,
            shape=(samples.value, features.value),
            dtype=np.float64,
        )

    def _indexes(self, kind):
        indexes = rascal_indexes_t()
        self._lib.rascal_descriptor_indexes(self, kind.value, pointer(indexes))

        shape = (indexes.count, indexes.size)
        ptr = indexes.values if indexes.count != 0 else None
        names = [indexes.names[i].decode("utf8") for i in range(indexes.size)]

        array = _ptr_to_ndarray(ptr=ptr, shape=shape, dtype=np.int32)
        array.flags.writeable = False
        return Indexes(array=array, names=names)

    @property
    def samples(self):
        """Sample metadata.

        Metdata describing the samples/rows in :py:attr:`Descriptor.values`.

        The data is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes_kind.RASCAL_INDEXES_SAMPLES)

    @property
    def features(self):
        """Feature metdata.

        Metdata describing the features/columns :py:attr:`Descriptor.values`
        and :py:attr:`Descriptor.gradients`.

        The data is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes_kind.RASCAL_INDEXES_FEATURES)

    @property
    def gradients_samples(self):
        """Gradient sample metadata.

        Metdata describing the rows in :py:attr:`Descriptor.gradients`.

        If there are no gradients stored in this descriptor,
        ``gradients_samples`` is ``None``.

        The data is stored as a :py:class:`rascaline.descriptor.Indexes` wrapping a
        **read only** 2D numpy ndarray with ``dtype=np.float64``. Each column of
        the array is named, and the names are available in ``Indexes.names``.
        """
        return self._indexes(rascal_indexes_kind.RASCAL_INDEXES_GRADIENT_SAMPLES)

    def densify(self, variables, requested=None):
        """Make this descriptor dense along the given ``variables``.

        :param variables: names of the variables to move
        :type variables: str | list[str]

        :param requested: set values taken by the variables in the new features
        :type requested: Optional[numpy.ndarray]

        This function "moves" the variables from the samples to the features,
        filling the new features with zeros if the corresponding sample is
        missing.

        The ``requested`` parameter defines which set of values taken by the
        ``variables`` should be part of the new features. If it is ``None``,
        this is the set of values taken by the variables in the samples.
        Otherwise, it must be an array with one row for each new feature block,
        and one column for each variable.

        For example, take a descriptor containing two samples variables
        (``structure`` and ``species``) and two features (``n`` and ``l``).
        Starting with this descriptor:

        .. code-block:: text

                                  +---+---+---+
                                  | n | 0 | 1 |
                                  +---+---+---+
                                  | l | 0 | 1 |
            +-----------+---------+===+===+===+
            | structure | species |           |
            +===========+=========+   +---+---+
            |     0     |    1    |   | 1 | 2 |
            +-----------+---------+   +---+---+
            |     0     |    6    |   | 3 | 4 |
            +-----------+---------+   +---+---+
            |     1     |    6    |   | 5 | 6 |
            +-----------+---------+   +---+---+
            |     1     |    8    |   | 7 | 8 |
            +-----------+---------+---+---+---+

        Calling ``descriptor.densify(["species"])`` will move ``species`` out of
        the samples and into the features, producing:

        .. code-block:: text

                        +---------+-------+-------+-------+
                        | species |   1   |   6   |   8   |
                        +---------+---+---+---+---+---+---+
                        |    n    | 0 | 1 | 0 | 1 | 0 | 1 |
                        +---------+---+---+---+---+---+---+
                        |    l    | 0 | 1 | 0 | 1 | 0 | 1 |
            +-----------+=========+===+===+===+===+===+===+
            | structure |
            +===========+         +---+---+---+---+---+---+
            |     0     |         | 1 | 2 | 3 | 4 | 0 | 0 |
            +-----------+         +---+---+---+---+---+---+
            |     1     |         | 0 | 0 | 5 | 6 | 7 | 8 |
            +-----------+---------+---+---+---+---+---+---+

        Notice how there is only one row/sample for each structure now, and how
        each value for ``species`` have created a full block of features.
        Missing values (e.g. structure 0/species 8) have been filled with 0.
        """
        if isinstance(variables, str):
            variables = [variables]

        requested, requested_size = _densify_prepare_requested_features(
            variables, requested
        )

        c_variables = ARRAY(c_char_p, len(variables))()
        for i, v in enumerate(variables):
            c_variables[i] = v.encode("utf8")
        self._lib.rascal_descriptor_densify(
            self, c_variables, c_variables._length_, requested, requested_size
        )

    def densify_values(self, variables, requested=None):
        """Densifiy descriptor values.

        Make this descriptor dense along the given ``variables``, only modifying
        the values array, and not the gradients array.

        This function behaves similarly to :py:func:`Descriptor.densify`, please
        refer to its documentation for more information.

        If this descriptor contains gradients, this function returns a vector
        containing all the information required to densify the gradient array.

        This is an advanced function most users should not need to use, used to
        implement backward propagation without having to densify the full
        gradient array.
        """
        if isinstance(variables, str):
            variables = [variables]

        requested, requested_size = _densify_prepare_requested_features(
            variables, requested
        )

        c_variables = ARRAY(c_char_p, len(variables))()
        for i, v in enumerate(variables):
            c_variables[i] = v.encode("utf8")

        densified_positions = POINTER(rascal_densified_position_t)()
        densified_positions_size = c_uintptr_t()
        self._lib.rascal_descriptor_densify_values(
            self,
            c_variables,
            c_variables._length_,
            requested,
            requested_size,
            densified_positions,
            densified_positions_size,
        )

        result = np.ctypeslib.as_array(
            densified_positions,
            shape=(densified_positions_size.value,),
        ).copy()

        self._lib.free(densified_positions)

        return result


def _ptr_to_ndarray(ptr, shape, dtype):
    assert len(shape) == 2
    if shape[0] != 0 and shape[1] != 0 and ptr is not None:
        array = np.ctypeslib.as_array(ptr, shape=shape)
        assert array.dtype == dtype
        array.flags.writeable = True
        return array
    else:
        data = np.array([], dtype=dtype)
        return data.reshape(shape)


def _densify_prepare_requested_features(variables, requested):
    if requested is not None:
        requested = np.array(requested)
        if len(requested.shape) == 1:
            requested = requested.reshape(requested.shape[0], 1)

        if len(requested.shape) != 2 or requested.shape[1] != len(variables):
            raise ValueError(
                "invalid requested features array shape: expected "
                f"(N, {len(variables)}); got {requested.shape}"
            )

        if not np.can_cast(requested, np.int32, casting="same_kind"):
            raise ValueError("the requested features must contain int32 values")
        requested = np.array(requested, dtype=np.int32)

        ptr_int32 = POINTER(c_int32)

        requested_size = requested.shape[0]
        requested = requested.ctypes.data_as(ptr_int32)
    else:
        requested_size = 0

    return requested, requested_size
