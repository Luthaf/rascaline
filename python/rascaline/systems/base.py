# -*- coding: utf-8 -*-
import numpy as np
import ctypes
from ctypes import POINTER, pointer, c_void_p, c_double

from .._rascaline import rascal_system_t, rascal_pair_t, c_uintptr_t


class SystemBase:
    """
    Base class implementing the `System` trait in rascaline. Developers should
    implement this class to add new kinds of system that work with rascaline.

    Most users should use one of the already provided implementation, such as
    ASESystem instead of using this class directly.
    """

    def __init__(self):
        # keep reference to some data to prevent garbage collection while Rust
        # might be using the data
        self._keepalive = {}

    def _as_rascal_system_t(self):
        """
        Convert a child instance of `SystemBase` to a C compatible
        `rascal_system_t`.
        """
        struct = rascal_system_t()
        self._keepalive["c_struct"] = struct

        # user_data is a pointer to the PyObject `self`
        struct.user_data = ctypes.cast(pointer(ctypes.py_object(self)), c_void_p)

        def get_self(ptr):
            """Extract `self` from a pointer to the PyObject"""
            self = ctypes.cast(ptr, POINTER(ctypes.py_object)).contents.value
            assert isinstance(self, SystemBase)
            return self

        def rascal_system_size(user_data, size):
            """
            Implementation of ``rascal_system_t::size`` using
            ``SystemBase.size``.
            """
            size[0] = get_self(user_data).size()

        # use struct.XXX.__class__ to get the right type for all functions
        struct.size = struct.size.__class__(rascal_system_size)

        def rascal_system_species(user_data, data):
            """
            Implementation of ``rascal_system_t::species`` using
            ``SystemBase.species``.
            """
            self = get_self(user_data)

            species = np.array(self.species(), dtype=c_uintptr_t)
            data[0] = species.ctypes.data
            self._keepalive["species"] = species

        struct.species = struct.species.__class__(rascal_system_species)

        def rascal_system_positions(user_data, data):
            """
            Implementation of ``rascal_system_t::positions`` using
            ``SystemBase.positions``.
            """
            self = get_self(user_data)
            positions = np.array(self.positions(), dtype=c_double)

            assert len(positions.shape) == 2
            assert positions.shape[1] == 3

            data[0] = positions.ctypes.data
            self._keepalive["positions"] = positions

        struct.positions = struct.positions.__class__(rascal_system_positions)

        def rascal_system_cell(user_data, data):
            """
            Implementation of ``rascal_system_t::cell`` using
            ``SystemBase.cell``.
            """
            self = get_self(user_data)
            cell = np.array(self.cell(), dtype=c_double)
            if cell.shape == (3, 3):
                cell = cell.reshape(9)

            assert len(cell) == 9

            data[0] = cell[0]
            data[1] = cell[1]
            data[2] = cell[2]
            data[3] = cell[3]
            data[4] = cell[4]
            data[5] = cell[5]
            data[6] = cell[6]
            data[7] = cell[7]
            data[8] = cell[8]

        struct.cell = struct.cell.__class__(rascal_system_cell)

        def rascal_system_compute_neighbors(user_data, cutoff):
            """
            Implementation of ``rascal_system_t::compute_neighbors`` using
            ``SystemBase.compute_neighbors``.
            """
            self = get_self(user_data)
            self.compute_neighbors(cutoff)

        struct.compute_neighbors = struct.compute_neighbors.__class__(
            rascal_system_compute_neighbors
        )

        def rascal_system_pairs(user_data, data, count):
            """
            Implementation of ``rascal_system_t::pairs`` using
            ``SystemBase.pairs``.
            """
            self = get_self(user_data)

            pairs = np.array(self.pairs(), dtype=rascal_pair_t)

            count[0] = c_uintptr_t(len(pairs))
            data[0] = pairs.ctypes.data
            self._keepalive["pairs"] = pairs

        struct.pairs = struct.pairs.__class__(rascal_system_pairs)

        def rascal_system_pairs_containing(user_data, center, data, count):
            """
            Implementation of ``rascal_system_t::pairs_containing`` using
            ``SystemBase.pairs_containing``.
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
        """
        Get the number of atoms in this system as an integer
        """
        raise NotImplementedError("System.size method is not implemented")

    def species(self):
        """
        Get a list of integers or a 1D numpy array of integers containing the
        atomic species for each atom in the system. Each different atomic
        species should be identified with a different value. These values are
        usually the atomic number, but don't have to be.
        """
        raise NotImplementedError("System.species method is not implemented")

    def positions(self):
        """
        Get the cartesian position of all atoms in this system. The returned
        positions must be convertible to a numpy array of shape ``(self.size(),
        3)``.
        """
        raise NotImplementedError("System.positions method is not implemented")

    def cell(self):
        """
        Get the 3x3 matrix representing unit cell of the system.
        """
        raise NotImplementedError("System.cell method is not implemented")

    def compute_neighbors(self, cutoff):
        """
        Compute the neighbor list with the given ``cutoff``, and store it for
        later access using :py:`SystemBase.pairs` or
        :py:`SystemBase.pairs_containing`.
        """
        raise NotImplementedError("System.compute_neighbors method is not implemented")

    def pairs(self):
        """
        Get all neighbor pairs in this system as a list of tuples ``(int, int,
        (float, float, float))`` containing the indexes of the first and second
        atom in the pair; and the wrapped distance vector between them.
        Alternatively, this function can return a numpy array with
        ``dtype=rascal_pair_t``.

        The list of pair should only contain each pair once (and not twice as
        `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
        contains pairs where the distance between atoms is actually bellow the
        cutoff passed in the last call to :py:`SystemBase.compute_neighbors`.
        This function is only valid to call after a call to
        :py:`SystemBase.compute_neighbors`.
        """
        raise NotImplementedError("System.pairs method is not implemented")

    def pairs_containing(self, center):
        """
        Get all neighbor pairs in this system containing the atom with index
        ``center``.

        The same restrictions on the list of pairs as :py:`SystemBase.pairs`
        applies, with the additional condition that the pair `i-j` should be
        included both in the list returned by
        :py:`SystemBase.pairs_containing(i)` and
        :py:`SystemBase.pairs_containing(j)`.
        """
        raise NotImplementedError("System.pairs_containing method is not implemented")
