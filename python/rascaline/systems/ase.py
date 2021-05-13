import numpy as np
import warnings
import copy

try:
    import ase
    from ase import neighborlist

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False

from .base import SystemBase


class AseSystem(SystemBase):
    """
    This class implements :py:class:`rascaline.SystemBase` using the
    `ase.Atoms`_ to get the data and `ase.neighborlist.neighbor_list`_ to
    compute the neighbor list.

    .. _ase.Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html

    .. _ase.neighborlist.neighbor_list:
        https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
    """

    @staticmethod
    def can_wrap(o):
        return isinstance(o, ase.Atoms)

    def __init__(self, atoms):
        """
        :param atoms: `ase.Atoms`_ object to be wrapped in this ``AseSystem``
        """
        super().__init__()
        if not isinstance(atoms, ase.Atoms):
            raise Exception("this class expects ASE.Atoms objects")

        # normalize pbc to three values
        if isinstance(atoms.pbc, bool):
            atoms_pbc = [atoms.pbc, atoms.pbc, atoms.pbc]
        else:
            atoms_pbc = copy.deepcopy(atoms.pbc)

        self._cell = np.array(atoms.cell)
        # validate pcb consistency with the cell matrix
        if np.all(np.abs(self._cell) < 1e-9):
            if np.any(atoms_pbc):
                raise Exception(
                    "periodic boundary conditions are enabled, "
                    "but the cell matrix is zero everywhere. "
                    "You should set pbc to `False`, or the cell to its value."
                )
        elif np.any(np.bitwise_not(atoms_pbc)):
            if np.all(np.bitwise_not(atoms_pbc)):
                warnings.warn(
                    "periodic boundary conditions are disabled, but the cell "
                    "matrix is not zero, we will set the cell to zero."
                )
                self._cell[:, :] = 0.0
            else:
                raise Exception(
                    "different periodic boundary conditions on different axis "
                    "are not supported"
                )

        self._atoms = atoms
        self._pairs = []
        self._pairs_by_center = []
        self._last_cutoff = None

    def size(self):
        return len(self._atoms)

    def species(self):
        return self._atoms.numbers

    def positions(self):
        return self._atoms.positions

    def cell(self):
        return self._cell

    def compute_neighbors(self, cutoff):
        if self._last_cutoff == cutoff:
            return

        self._pairs = []

        nl_result = neighborlist.neighbor_list("ijdD", self._atoms, cutoff)
        for (i, j, d, D) in zip(*nl_result):
            if j < i:
                # we want a half neighbor list, so drop all duplicated
                # neighbors
                continue
            self._pairs.append((i, j, d, D))

        self._pairs_by_center = []
        for _ in range(self.size()):
            self._pairs_by_center.append([])

        for (i, j, d, D) in self._pairs:
            self._pairs_by_center[i].append((i, j, d, D))
            self._pairs_by_center[j].append((i, j, d, D))

    def pairs(self):
        return self._pairs

    def pairs_containing(self, center):
        return self._pairs_by_center[center]
