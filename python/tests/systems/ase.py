# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline.systems import AseSystem, HAVE_ASE

try:
    import ase
except ImportError:
    pass


@unittest.skipIf(not HAVE_ASE, "ASE is not installed")
class TestAseSystem(unittest.TestCase):
    def setUp(self):
        self.positions = [
            (0, 0, 0),
            (0, 0, 1.4),
            (0, 0, -1.6),
        ]
        atoms = ase.Atoms(
            "CO2",
            positions=self.positions,
        )
        atoms.pbc = [False, False, False]
        self.system = AseSystem(atoms)

    def test_system_implementation(self):
        self.assertEqual(self.system.size(), 3)
        self.assertTrue(np.all(self.system.species() == [6, 8, 8]))
        self.assertTrue(np.all(self.system.positions() == self.positions))
        self.assertTrue(np.all(self.system.cell() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def test_pairs(self):
        self.system.compute_neighbors(1.5)
        pairs = self.system.pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][:2], (0, 1))
        self.assertEqual(pairs[0][2], 1.4)
        self.assertTrue(np.all(pairs[0][3] == [0, 0, 1.4]))

        self.system.compute_neighbors(2.5)
        pairs = self.system.pairs()
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][:2], (0, 1))
        self.assertEqual(pairs[0][2], 1.4)
        self.assertTrue(np.all(pairs[0][3] == [0, 0, 1.4]))

        self.assertEqual(pairs[1][:2], (0, 2))
        self.assertEqual(pairs[1][2], 1.6)
        self.assertTrue(np.all(pairs[1][3] == [0, 0, -1.6]))

        self.system.compute_neighbors(3.5)
        pairs = self.system.pairs()
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0][:2], (0, 1))
        self.assertEqual(pairs[0][2], 1.4)
        self.assertTrue(np.all(pairs[0][3] == [0, 0, 1.4]))

        self.assertEqual(pairs[1][:2], (0, 2))
        self.assertEqual(pairs[1][2], 1.6)
        self.assertTrue(np.all(pairs[1][3] == [0, 0, -1.6]))

        self.assertEqual(pairs[2][:2], (1, 2))
        self.assertEqual(pairs[2][2], 3.0)
        self.assertTrue(np.all(pairs[2][3] == [0, 0, -3.0]))

    def test_pairs_containing(self):
        self.system.compute_neighbors(1.5)
        pairs = self.system.pairs_containing(0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][:2], (0, 1))

        pairs = self.system.pairs_containing(1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][:2], (0, 1))

        pairs = self.system.pairs_containing(2)
        self.assertEqual(len(pairs), 0)

        self.system.compute_neighbors(3.5)
        pairs = self.system.pairs_containing(0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][:2], (0, 1))
        self.assertEqual(pairs[1][:2], (0, 2))

        pairs = self.system.pairs_containing(1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][:2], (0, 1))
        self.assertEqual(pairs[1][:2], (1, 2))

        pairs = self.system.pairs_containing(2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][:2], (0, 2))
        self.assertEqual(pairs[1][:2], (1, 2))


@unittest.skipIf(not HAVE_ASE, "ASE is not installed")
class TestAseSystemErrors(unittest.TestCase):
    def test_pbc_no_cell(self):
        message = (
            "periodic boundary conditions are enabled, but the cell "
            + "matrix is zero everywhere. You should set pbc to `False`, "
            + "or the cell to its value."
        )

        atoms = ase.Atoms("C", positions=[(0, 0, 0)])
        atoms.pbc = [True, True, True]

        with self.assertRaises(Exception) as cm:
            _ = AseSystem(atoms)

        self.assertEqual(cm.exception.args[0], message)

        atoms.cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        atoms.pbc = True

        with self.assertRaises(Exception) as cm:
            _ = AseSystem(atoms)

        self.assertEqual(cm.exception.args[0], message)

    def test_partial_pbc(self):
        atoms = ase.Atoms("C", positions=[(0, 0, 0)])

        atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        atoms.pbc = [True, True, False]

        with self.assertRaises(Exception) as cm:
            _ = AseSystem(atoms)

        message = (
            "different periodic boundary conditions on different axis "
            + "are not supported"
        )

        self.assertEqual(cm.exception.args[0], message)

    def test_no_pbc_cell(self):
        atoms = ase.Atoms("C", positions=[(0, 0, 0)])

        atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        atoms.pbc = False

        with self.assertWarns(Warning) as cm:
            _ = AseSystem(atoms)

        message = (
            "periodic boundary conditions are disabled, but the cell matrix is "
            + "not zero, we will set the cell to zero."
        )

        self.assertEqual(cm.warning.args[0], message)
