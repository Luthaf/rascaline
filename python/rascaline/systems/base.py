# -*- coding: utf-8 -*-
import numpy as np
import ctypes
from ctypes import POINTER, pointer, c_void_p, c_double

from .._rascaline import rascal_system_t, rascal_pair_t, c_uintptr_t


class SystemBase:
    def __init__(self):
        self._keepalive = {}

    def _as_rascal_system_t(self):
        # keep the struct alive to prevent garbage collection while the Rust
        # code is using it
        self._c_struct = rascal_system_t()
        self._c_struct.user_data = ctypes.cast(
            pointer(ctypes.py_object(self)), c_void_p
        )
        self._c_struct.size = self._c_struct.size.__class__(_size_cb)
        self._c_struct.species = self._c_struct.species.__class__(_species_cb)
        self._c_struct.positions = self._c_struct.positions.__class__(_positions_cb)
        self._c_struct.cell = self._c_struct.cell.__class__(_cell_cb)
        self._c_struct.compute_neighbors = self._c_struct.compute_neighbors.__class__(
            _compute_neighbors_cb
        )
        self._c_struct.pairs = self._c_struct.pairs.__class__(_pairs_cb)
        self._c_struct.pairs_containing = self._c_struct.pairs_containing.__class__(
            _pairs_containing_cb
        )

        return self._c_struct

    def size(self):
        raise NotImplementedError("System.size method is not implemented")

    def species(self):
        raise NotImplementedError("System.species method is not implemented")

    def positions(self):
        raise NotImplementedError("System.positions method is not implemented")

    def cell(self):
        raise NotImplementedError("System.cell method is not implemented")

    def compute_neighbors(self, cutoff):
        raise NotImplementedError("System.compute_neighbors method is not implemented")

    def pairs(self):
        raise NotImplementedError("System.pairs method is not implemented")

    def pairs_containing(self, center):
        raise NotImplementedError("System.pairs_containing method is not implemented")


def _get_self(ptr):
    self = ctypes.cast(ptr, POINTER(ctypes.py_object)).contents.value
    assert isinstance(self, SystemBase)
    return self


def _size_cb(user_data, size):
    size[0] = _get_self(user_data).size()


def _species_cb(user_data, data):
    self = _get_self(user_data)

    species = np.array(self.species(), dtype=c_uintptr_t)
    data[0] = species.ctypes.data
    self._keepalive["species"] = species


def _positions_cb(user_data, data):
    self = _get_self(user_data)
    positions = np.array(self.positions(), dtype=c_double)

    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    data[0] = positions.ctypes.data
    self._keepalive["positions"] = positions


def _cell_cb(user_data, data):
    self = _get_self(user_data)
    cell = np.array(self.cell(), dtype=c_double)
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


def _compute_neighbors_cb(user_data, cutoff):
    self = _get_self(user_data)
    self.compute_neighbors(float(cutoff))


def _pairs_cb(user_data, data, count):
    self = _get_self(user_data)

    pairs = np.array(self.pairs(), dtype=rascal_pair_t)

    count[0] = c_uintptr_t(len(pairs))
    data[0] = pairs.ctypes.data
    self._keepalive["pairs"] = pairs


def _pairs_containing_cb(user_data, center, data, count):
    self = _get_self(user_data)

    pairs = np.array(self.pairs_containing(center), dtype=rascal_pair_t)

    count[0] = c_uintptr_t(len(pairs))
    data[0] = pairs.ctypes.data
    self._keepalive["pairs_containing"] = pairs
