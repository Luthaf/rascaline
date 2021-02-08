# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline.systems import AseSystem

try:
    import ase

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


if HAVE_ASE:

    class TestAseSystem(unittest.TestCase):
        def test_wrap(self):
            positions = [
                (0, 0, 0),
                (0, 0, 1.4),
                (0, 0, -1.6),
            ]
            atoms = ase.Atoms(
                "CO2",
                positions=positions,
            )

            atoms.pbc = [False, False, False]

            system = AseSystem(atoms)

            self.assertEqual(system.size(), 3)
            self.assertTrue(np.all(system.species() == [6, 8, 8]))
            self.assertTrue(np.all(system.positions() == positions))
            self.assertTrue(np.all(system.cell() == [0, 0, 0, 0, 0, 0, 0, 0, 0]))

        def test_neighbors(self):
            positions = [
                (0, 0, 0),
                (0, 0, 1.4),
                (0, 0, -1.6),
            ]
            atoms = ase.Atoms(
                "CO2",
                positions=positions,
            )

            system = AseSystem(atoms)
            system.compute_neighbors(1.5)
            pairs = system.pairs()
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][:2], (0, 1))
            self.assertTrue(np.all(pairs[0][2] == [0, 0, 1.4]))

            system.compute_neighbors(2.5)
            pairs = system.pairs()
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0][:2], (0, 1))
            self.assertTrue(np.all(pairs[0][2] == [0, 0, 1.4]))

            self.assertEqual(pairs[1][:2], (0, 2))
            self.assertTrue(np.all(pairs[1][2] == [0, 0, -1.6]))

            system.compute_neighbors(3.5)
            pairs = system.pairs()
            self.assertEqual(len(pairs), 3)
            self.assertEqual(pairs[0][:2], (0, 1))
            self.assertTrue(np.all(pairs[0][2] == [0, 0, 1.4]))

            self.assertEqual(pairs[1][:2], (0, 2))
            self.assertTrue(np.all(pairs[1][2] == [0, 0, -1.6]))

            self.assertEqual(pairs[2][:2], (1, 2))
            self.assertTrue(np.all(pairs[2][2] == [0, 0, -3.0]))
