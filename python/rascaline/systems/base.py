# -*- coding: utf-8 -*-
import ctypes
from ctypes import POINTER, c_double, c_void_p, pointer

import numpy as np

from .._c_api import c_uintptr_t, rascal_pair_t, rascal_system_t
from ..status import _save_exception


def catch_exceptions(function):
    """Decorate a function catching any exception."""

    def inner(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as e:
            _save_exception(e)
            return -1
        return 0

    return inner


class SystemBase:
    """Base class implementing the ``System`` trait in rascaline.

    Developers should implement this class to add new kinds of system that
    work with rascaline.

    Most users should use one of the already provided implementation, such as
    :py:class:`rascaline.systems.AseSystem` or
    :py:class:`rascaline.systems.ChemfilesSystem` instead of using this class
    directly.
    """

    def __init__(self):
        # keep reference to some data to prevent garbage collection while Rust
        # might be using the data
        self._keepalive = {}

    def _as_rascal_system_t(self):
        """Convert a child instance of :py:class:`SystemBase`.

        Instances are converted to a C compatible ``rascal_system_t``.
        """
        struct = rascal_system_t()
        self._keepalive["c_struct"] = struct

        # user_data is a pointer to the PyObject `self`
        struct.user_data = ctypes.cast(pointer(ctypes.py_object(self)), c_void_p)

        def get_self(ptr):
            """Extract ``self`` from a pointer to the PyObject."""
            self = ctypes.cast(ptr, POINTER(ctypes.py_object)).contents.value
            assert isinstance(self, SystemBase)
            return self

        @catch_exceptions
        def rascal_system_size(user_data, size):
            """
            Implementation of ``rascal_system_t::size`` using
            :py:func:`SystemBase.size`.
            """
            size[0] = get_self(user_data).size()

        # use struct.XXX.__class__ to get the right type for all functions
        struct.size = struct.size.__class__(rascal_system_size)

        @catch_exceptions
        def rascal_system_species(user_data, data):
            """
            Implementation of ``rascal_system_t::species`` using
            :py:func:`SystemBase.species`.
            """
            self = get_self(user_data)

            species = np.array(self.species(), dtype=np.int32)
            data[0] = species.ctypes.data
            self._keepalive["species"] = species

        struct.species = struct.species.__class__(rascal_system_species)

        @catch_exceptions
        def rascal_system_positions(user_data, data):
            """
            Implementation of ``rascal_system_t::positions`` using
            :py:func:`SystemBase.positions`.
            """
            self = get_self(user_data)
            positions = np.array(self.positions(), dtype=c_double)

            assert len(positions.shape) == 2
            assert positions.shape[1] == 3

            data[0] = positions.ctypes.data
            self._keepalive["positions"] = positions

        struct.positions = struct.positions.__class__(rascal_system_positions)

        @catch_exceptions
        def rascal_system_cell(user_data, data):
            """
            Implementation of ``rascal_system_t::cell`` using
            :py:func:`SystemBase.cell`.
            """
            self = get_self(user_data)
            cell = np.array(self.cell(), dtype=c_double)
            assert cell.shape == (3, 3)

            data[0] = cell[0][0]
            data[1] = cell[0][1]
            data[2] = cell[0][2]
            data[3] = cell[1][0]
            data[4] = cell[1][1]
            data[5] = cell[1][2]
            data[6] = cell[2][0]
            data[7] = cell[2][1]
            data[8] = cell[2][2]

        struct.cell = struct.cell.__class__(rascal_system_cell)

        @catch_exceptions
        def rascal_system_compute_neighbors(user_data, cutoff):
            """
            Implementation of ``rascal_system_t::compute_neighbors`` using
            :py:func:`SystemBase.compute_neighbors`.
            """
            self = get_self(user_data)
            self.compute_neighbors(cutoff)

        struct.compute_neighbors = struct.compute_neighbors.__class__(
            rascal_system_compute_neighbors
        )

        @catch_exceptions
        def rascal_system_pairs(user_data, data, count):
            """
            Implementation of ``rascal_system_t::pairs`` using
            :py:func:`SystemBase.pairs`.
            """
            self = get_self(user_data)

            pairs = np.array(self.pairs(), dtype=rascal_pair_t)

            count[0] = c_uintptr_t(len(pairs))
            data[0] = pairs.ctypes.data
            self._keepalive["pairs"] = pairs

        struct.pairs = struct.pairs.__class__(rascal_system_pairs)

        @catch_exceptions
        def rascal_system_pairs_containing(user_data, center, data, count):
            """
            Implementation of ``rascal_system_t::pairs_containing`` using
            :py:func:`SystemBase.pairs_containing`.
            """
            self = get_self(user_data)

            pairs = np.array(self.pairs_containing(center), dtype=rascal_pair_t)

            count[0] = c_uintptr_t(len(pairs))
            data[0] = pairs.ctypes.data
            self._keepalive["pairs_containing"] = pairs

        struct.pairs_containing = struct.pairs_containing.__class__(
            rascal_system_pairs_containing
        )

        return struct

    def size(self):
        """Get the number of atoms in this system as an integer."""
        raise NotImplementedError("System.size method is not implemented")

    def species(self):
        """Get the atomic species of all atoms in the system.

        Get a list of integers or a 1D numpy array of integers containing the atomic
        species for each atom in the system. Each different atomic species
        should be identified with a different value.
        These values are usually the atomic number, but don't have to be.
        """
        raise NotImplementedError("System.species method is not implemented")

    def positions(self):
        """Get the cartesian position of all atoms in this system.

        The returned positions must be convertible to a numpy array of
        shape ``(self.size(), 3)``.
        """
        raise NotImplementedError("System.positions method is not implemented")

    def cell(self):
        """Get the 3x3 matrix representing unit cell of the system.

        The cell should be written in row major order, i.e. `[[ax, ay, az],
        [bx, by, bz], [cx, cy, cz]]`, where a/b/c are the unit cell vectors.
        """
        raise NotImplementedError("System.cell method is not implemented")

    def compute_neighbors(self, cutoff):
        """Compute the neighbor list with the given ``cutoff``.

        Store it for later access using :py:func:`rascaline.SystemBase.pairs`
        or :py:func:`rascaline.SystemBase.pairs_containing`.
        """
        raise NotImplementedError("System.compute_neighbors method is not implemented")

    def pairs(self):
        """Atoms pairs in this system.

        The pairs are those which were
        computed by the last call :py:func:`SystemBase.compute_neighbors`

        Get all neighbor pairs in this system as a list of tuples ``(int, int,
        float, (float, float, float))`` containing the indexes of the first and
        second atom in the pair, the distance between the atoms, and the
        wrapped between them. Alternatively, this function can return a numpy
        array with ``dtype=rascal_pair_t``.

        The list of pair should only contain each pair once (and not twice as
        ``i-j`` and ``j-i``), should not contain self pairs (``i-i``); and
        should only contains pairs where the distance between atoms is actually
        bellow the cutoff passed in the last call to
        :py:func:`rascaline.SystemBase.compute_neighbors`.

        This function is only valid to call after a call to
        :py:func:`rascaline.SystemBase.compute_neighbors` to set the cutoff.
        """
        raise NotImplementedError("System.pairs method is not implemented")

    def pairs_containing(self, center):
        """Get all neighbor pairs in this system containing the atom with index ``center``.

        The same restrictions on the list of pairs as
        :py:func:`rascaline.SystemBase.pairs` applies, with the additional
        condition that the pair ``i-j`` should be included both in the list
        returned by ``pairs_containing(i)`` and ``pairs_containing(j)``.
        """
        raise NotImplementedError("System.pairs_containing method is not implemented")
