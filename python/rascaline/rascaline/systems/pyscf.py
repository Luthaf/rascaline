import warnings

import numpy as np

from .._c_api import c_uintptr_t
from .base import SystemBase


try:
    import pyscf
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False


# copied from pyscf/data/elements.py
def _std_symbol_without_ghost(symb_or_chg):
    '''For a given atom symbol (lower case or upper case) or charge, return the
    standardized atom symbol
    '''
    if isinstance(symb_or_chg, (str, unicode)):
        symb_or_chg = str(symb_or_chg.upper())
        rawsymb = pyscf.data.elements._rm_digit(symb_or_chg)
        if rawsymb in _ELEMENTS_UPPER:
            return _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 1 and symb_or_chg[0] == 'X' and symb_or_chg[:2] != 'XE':
            rawsymb = rawsymb[1:]  # Remove the prefix X
            return _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 5 and rawsymb[:5] == 'GHOST':
            rawsymb = rawsymb[5:]  # Remove the prefix GHOST
            return _ELEMENTS_UPPER[rawsymb]
        else:
            raise RuntimeError('Unsupported atom symbol %s' % symb_or_chg)
    else:
        return pyscf.data.elements.ELEMENTS[symb_or_chg]

class PyscfSystem(SystemBase):
    """Implements :py:class:`rascaline.SystemBase` wrapping a `pyscf.gto.mole.Mole`_ or `pyscf.pbc.gto.cell.Cell`_.

    Since pyscf does not offer a neighbors list, this
    implementation of system can only be used with ``use_native_system=True`` in
    :py:func:`rascaline.calculators.CalculatorBase.compute`.

    Atomic species are assigned as the atomic number if the atom ``type`` is one
    of the periodic table elements; or their opposite if they are ghost atoms.
    (Pyscf does not seem to support anything else)

    .. _pyscf.gto.mole.Mole: https://pyscf.org/user/gto.html
    .. _pyscf.pbc.gto.cell.Cell: https://pyscf.org/user/pbc/gto.html
    """

    @staticmethod
    def can_wrap(o):
        return isinstance(o, (pyscf.gto.mole.Mole,pyscf.pbc.gto.cell.Cell))

    def __init__(self, frame):
        """
        :param frame : `chemfiles.Frame`_ object object to be wrapped
            in this ``ChemfilesSystem``
        """
        super().__init__()
        if not isinstance(frame, (pyscf.gto.mole.Mole,pyscf.pbc.gto.cell.Cell)):
            raise Exception("this class expects pyscf.gto.mole.Mole or pyscf.pbc.gto.cell.Cell objects")

        self._frame = frame
        self._species = self._frame.atom_charges().copy()  # dtype=int32
        for atm_i, species in enumerate(self._species):
            if species == 0:
                symb = self._frame.atom_symbol(atm_i)
                chg = pyscf.data.elements.index(symb)
                self._species[atm_i] = -chg
        self.is_periodic = isinstance(self._frame, pyscf.pbc.gto.cell.Cell)

    def size(self):
        return self._frame.natm

    def species(self):
        return self._species

    def positions(self):
        return self._frame.atom_coords()

    def cell(self):
        if self.is_periodic:
            return self._frame.a
        else:
            return np.zeros((3,3),float)

    def compute_neighbors(self, cutoff):
        raise Exception(
            "pyscf systems can only be used with 'use_native_system=True'"
        )

    def pairs(self):
        raise Exception(
            "pyscf systems can only be used with 'use_native_system=True'"
        )

    def pairs_containing(self, center):
        raise Exception(
            "pyscf systems can only be used with 'use_native_system=True'"
        )
