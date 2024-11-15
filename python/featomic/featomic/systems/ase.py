import copy
import warnings

import numpy as np


try:
    import ase
    from ase import neighborlist

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False

from .base import SystemBase


class AseSystem(SystemBase):
    """Implements :py:class:`featomic.SystemBase` using `ase.Atoms`_.

    Gets the data and `ase.neighborlist.neighbor_list`_ to
    compute the neighbor list.

    .. _ase.Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html

    .. _ase.neighborlist.neighbor_list:
        https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
    """

    @staticmethod
    def can_wrap(o):
        return isinstance(o, ase.Atoms)

    def __init__(self, atoms):
        """:param atoms: `ase.Atoms`_ object to be wrapped in this ``AseSystem``"""
        super().__init__()
        if not isinstance(atoms, ase.Atoms):
            raise TypeError("this class expects ASE.Atoms objects")

        # normalize pbc to three values
        if isinstance(atoms.pbc, bool):
            atoms_pbc = [atoms.pbc, atoms.pbc, atoms.pbc]
        else:
            atoms_pbc = copy.deepcopy(atoms.pbc)

        self._cell = np.array(atoms.cell)
        # validate pcb consistency with the cell matrix
        if np.all(np.abs(self._cell) < 1e-9):
            if np.any(atoms_pbc):
                raise ValueError(
                    "periodic boundary conditions are enabled, "
                    "but the cell matrix is zero everywhere. "
                    "You should set pbc to `False`, or the cell to its value."
                )
        elif np.any(np.bitwise_not(atoms_pbc)):
            if np.all(np.bitwise_not(atoms_pbc)):
                warnings.warn(
                    "periodic boundary conditions are disabled, but the cell "
                    "matrix is not zero, we will set the cell to zero.",
                    stacklevel=1,
                )
                self._cell[:, :] = 0.0
            else:
                raise ValueError(
                    "different periodic boundary conditions on different axis "
                    "are not supported"
                )

        self._atoms = atoms
        self._pairs = []
        self._pairs_by_atom = []
        self._last_cutoff = None

    def size(self):
        return len(self._atoms)

    def types(self):
        return self._atoms.numbers

    def positions(self):
        return self._atoms.positions

    def cell(self):
        return self._cell

    def compute_neighbors(self, cutoff):
        if self._last_cutoff == cutoff:
            return

        self._pairs = []

        nl_result = neighborlist.neighbor_list("ijdDS", self._atoms, cutoff)
        for i, j, d, D, S in zip(*nl_result):
            # we want a half neighbor list, so drop all duplicated neighbors
            if j < i:
                continue
            elif i == j:
                if S[0] == 0 and S[1] == 0 and S[2] == 0:
                    # only create pairs with the same atom twice if the pair spans more
                    # than one unit cell
                    continue
                elif S[0] + S[1] + S[2] < 0 or (
                    (S[0] + S[1] + S[2] == 0) and (S[2] < 0 or (S[2] == 0 and S[1] < 0))
                ):
                    # When creating pairs between an atom and one of its periodic
                    # images, the code generate multiple redundant pairs (e.g. with
                    # shifts 0 1 1 and 0 -1 -1); and we want to only keep one of these.
                    # We keep the pair in the positive half plane of shifts.
                    continue

            self._pairs.append((i, j, d, D, S))

        self._pairs_by_atom = []
        for _ in range(self.size()):
            self._pairs_by_atom.append([])

        for i, j, d, D, S in self._pairs:
            self._pairs_by_atom[i].append((i, j, d, D, S))
            self._pairs_by_atom[j].append((i, j, d, D, S))

    def pairs(self):
        return self._pairs

    def pairs_containing(self, atom):
        return self._pairs_by_atom[atom]
