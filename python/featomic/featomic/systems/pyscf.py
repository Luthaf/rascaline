import numpy as np

from .base import SystemBase


try:
    import pyscf

    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False


# copied from pyscf/data/elements.py
def _std_symbol_without_ghost(symb_or_chg):
    """For a given atom symbol (lower case or upper case) or charge, return the
    standardized atom symbol
    """
    if isinstance(symb_or_chg, str):
        symb_or_chg = str(symb_or_chg.upper())
        rawsymb = pyscf.data.elements._rm_digit(symb_or_chg)
        if rawsymb in pyscf.data.elements._ELEMENTS_UPPER:
            return pyscf.data.elements._ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 1 and (symb_or_chg[0] == "X" and symb_or_chg[:2] != "XE"):
            rawsymb = rawsymb[1:]  # Remove the prefix X
            return pyscf.data.elements._ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 5 and rawsymb[:5] == "GHOST":
            rawsymb = rawsymb[5:]  # Remove the prefix GHOST
            return pyscf.data.elements._ELEMENTS_UPPER[rawsymb]
        else:
            raise RuntimeError("Unsupported atom symbol %s" % symb_or_chg)
    else:
        return pyscf.data.elements.ELEMENTS[symb_or_chg]


class PyscfSystem(SystemBase):
    """Implements :py:class:`featomic.SystemBase` wrapping a
    `pyscf.gto.mole.Mole`_ or `pyscf.pbc.gto.cell.Cell`_.

    Since pyscf does not offer a neighbors list, this implementation of system can only
    be used with ``use_native_system=True`` in
    :py:func:`featomic.calculators.CalculatorBase.compute`.

    Atomic type are assigned as the atomic number if the atom ``type`` is one of the
    periodic table elements; or their opposite if they are ghost atoms. (Pyscf does not
    seem to support anything else)

    Please note that while pyscf uses Bohrs as length units internally, we convert those
    back into Angströms for featomic. A pyscf object's "unit" attribute determines the
    units of the coordinates given *to pyscf*, which are by default angströms.

    .. _pyscf.gto.mole.Mole: https://pyscf.org/user/gto.html
    .. _pyscf.pbc.gto.cell.Cell: https://pyscf.org/user/pbc/gto.html
    """

    @staticmethod
    def can_wrap(o):
        # assumption: if we have a periodic system, then pyscf.pbc is defined
        if hasattr(pyscf, "pbc"):
            return isinstance(o, (pyscf.gto.mole.Mole, pyscf.pbc.gto.cell.Cell))
        else:
            return isinstance(o, pyscf.gto.mole.Mole)

    def __init__(self, system):
        """
        :param system : PySCF object object to be wrapped in this ``PyscfSystem``
        """
        super().__init__()
        if not self.can_wrap(system):
            raise Exception(
                "this class expects pyscf.gto.mole.Mole"
                + "or pyscf.pbc.gto.cell.Cell objects"
            )

        self._system = system
        self._types = self._system.atom_charges().copy()  # dtype=int32
        for atm_i, atomic_type in enumerate(self._types):
            if atomic_type == 0:
                symb = self._system.atom_symbol(atm_i)
                chg = pyscf.data.elements.index(symb)
                self._types[atm_i] = -chg
        if hasattr(pyscf, "pbc"):
            self.is_periodic = isinstance(self._system, pyscf.pbc.gto.cell.Cell)
        else:
            self.is_periodic = False

    def size(self):
        return self._system.natm

    def types(self):
        return self._types

    def positions(self):
        return pyscf.data.nist.BOHR * self._system.atom_coords()

    def cell(self):
        if self.is_periodic:
            cell = self._system.a
            if self._system.unit[0].lower() == "a":
                # assume angströms, we are good to go
                return cell
            else:
                # assume bohrs, correct this
                return pyscf.data.nist.BOHR * np.asarray(cell)
        else:
            return np.zeros((3, 3), float)

    def compute_neighbors(self, cutoff):
        raise Exception("pyscf systems can only be used with 'use_native_system=True'")

    def pairs(self):
        raise Exception("pyscf systems can only be used with 'use_native_system=True'")

    def pairs_containing(self, atom):
        raise Exception("pyscf systems can only be used with 'use_native_system=True'")
