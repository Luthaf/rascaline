import warnings

import numpy as np

from .._c_api import c_uintptr_t
from .base import SystemBase


try:
    import chemfiles

    if not chemfiles.__version__.startswith("0.10"):
        warnings.warn(
            message="found chemfiles, but the version is not supported: "
            "we need chemfiles v0.10.",
            category=ImportWarning,
            stacklevel=1,
        )
        HAVE_CHEMFILES = False
    else:
        HAVE_CHEMFILES = True
except ImportError:
    HAVE_CHEMFILES = False


# global cache of types number for atoms outside of the periodic table
ATOMIC_TYPES_CACHE = {}


def get_type_for_non_element(name):
    """Get type associated with atom that is not in the periodic table."""
    if name in ATOMIC_TYPES_CACHE:
        return ATOMIC_TYPES_CACHE[name]
    else:
        # start at 120 since that more atoms that are in the periodic table
        atomic_type = 120 + len(ATOMIC_TYPES_CACHE)
        ATOMIC_TYPES_CACHE[name] = atomic_type
        return atomic_type


class ChemfilesSystem(SystemBase):
    """Implements :py:class:`featomic.SystemBase` wrapping a `chemfiles.Frame`_.

    Since chemfiles does not offer a neighbors list, this implementation of system can
    only be used with ``use_native_system=True`` in
    :py:func:`featomic.calculators.CalculatorBase.compute`.

    Atomic type are assigned as the atomic number if the atom ``type`` is one of the
    periodic table elements; or as a value above 120 if the atom type is not in the
    periodic table.

    .. _chemfiles.Frame: http://chemfiles.org/chemfiles.py/latest/reference/frame.html
    """

    @staticmethod
    def can_wrap(o):
        return isinstance(o, chemfiles.Frame)

    def __init__(self, frame):
        """
        :param frame : `chemfiles.Frame`_ object object to be wrapped
            in this ``ChemfilesSystem``
        """
        super().__init__()
        if not isinstance(frame, chemfiles.Frame):
            raise Exception("this class expects chemfiles.Frame objects")

        self._frame = frame
        self._type = np.zeros((self.size()), dtype=c_uintptr_t)
        for i, atom in enumerate(self._frame.atoms):
            if atom.atomic_number != 0:
                self._type[i] = atom.atomic_number
            else:
                self._type[i] = get_type_for_non_element(atom.type)

    def size(self):
        return len(self._frame.atoms)

    def types(self):
        return self._type

    def positions(self):
        return self._frame.positions

    def cell(self):
        # we need to transpose the cell since chemfiles stores it in column
        # major format
        return self._frame.cell.matrix.T

    def compute_neighbors(self, cutoff):
        raise Exception(
            "chemfiles systems can only be used with 'use_native_system=True'"
        )

    def pairs(self):
        raise Exception(
            "chemfiles systems can only be used with 'use_native_system=True'"
        )

    def pairs_containing(self, atom):
        raise Exception(
            "chemfiles systems can only be used with 'use_native_system=True'"
        )
