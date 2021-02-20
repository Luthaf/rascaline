import numpy as np

try:
    import ase
    from ase import neighborlist

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False

from .base import SystemBase

if HAVE_ASE:

    class AseSystem(SystemBase):
        def __init__(self, atoms):
            super().__init__()
            if not isinstance(atoms, ase.Atoms):
                raise Exception("this class expects ASE.Atoms objects")
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
            return np.concatenate(self._atoms.cell[:, :])

        def compute_neighbors(self, cutoff):
            if self._last_cutoff == cutoff:
                return

            self._pairs = []

            nl_result = neighborlist.neighbor_list("ijD", self._atoms, cutoff)
            for (i, j, D) in zip(*nl_result):
                if j < i:
                    # we want a half neighbor list, so drop all duplicated
                    # neighbors
                    continue
                self._pairs.append((i, j, D))

            self._pairs_by_center = []
            for _ in range(self.size()):
                self._pairs_by_center.append([])

            for (i, j, D) in self._pairs:
                self._pairs_by_center[i].append((i, j, D))
                self._pairs_by_center[j].append((i, j, D))

        def pairs(self):
            return self._pairs

        def pairs_containing(self, center):
            return self._pairs_by_center[center]


else:

    class AseSystem(SystemBase):
        pass
