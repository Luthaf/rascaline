# -*- coding: utf-8 -*-
import numpy as np
from ctypes import c_double, c_int32, c_char_p, POINTER, ARRAY

from ._rascaline import c_uintptr_t, rascal_indexes, rascal_densified_position_t
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

    def __new__(cls, ptr, shape, names):
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
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", tuple())


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
        return _ptr_to_ndarray(data, (samples.value, features.value))

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

        return _ptr_to_ndarray(data, (samples.value, features.value))

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
        return Indexes(ptr=ptr, shape=shape, names=names)

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

    def densify(self, variables, requested=None):
        """
        Make this descriptor dense along the given ``variables``.

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
        """
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


def _ptr_to_ndarray(ptr, shape):
    assert len(shape) == 2
    if shape[0] != 0 and shape[1] != 0:
        array = np.ctypeslib.as_array(ptr, shape=shape)
        assert array.dtype == np.float64
        array.flags.writeable = True
        return array
    else:
        data = np.array([], dtype=np.float64)
        return data.reshape(shape)


def _densify_prepare_requested_features(variables, requested):
    if requested is not None:
        requested = np.array(requested)
        if len(requested.shape) == 1:
            requested = requested.reshape(requested.shape[0], 1)

        if len(requested.shape) != 2 or requested.shape[1] != len(variables):
            raise ValueError(
                "invalid requested features array shape: expected "
                + f"(N, {len(variables)}); got {requested.shape}"
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
