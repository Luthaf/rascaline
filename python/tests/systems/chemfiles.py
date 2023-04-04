import unittest

import numpy as np

from rascaline.calculators import DummyCalculator
from rascaline.status import RascalError
from rascaline.systems import HAVE_CHEMFILES, ChemfilesSystem


try:
    import chemfiles
except ImportError:
    pass


@unittest.skipIf(not HAVE_CHEMFILES, "chemfiles is not installed")
class TestChemfilesSystem(unittest.TestCase):
    def test_system_implementation(self):
        frame = chemfiles.Frame()
        frame.add_atom(chemfiles.Atom("C"), (0, 0, 0))
        frame.add_atom(chemfiles.Atom("CH3"), (0, 1, 0))
        frame.add_atom(chemfiles.Atom("X"), (0, 2, 0))
        frame.add_atom(chemfiles.Atom("Zn1", "Zn"), (0, 3, 0))

        system = ChemfilesSystem(frame)

        self.assertEqual(system.size(), 4)
        self.assertTrue(np.all(system.species() == [6, 120, 121, 30]))
        positions = [
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
        ]
        self.assertTrue(np.all(system.positions() == positions))

    def test_cell(self):
        frame = chemfiles.Frame()

        system = ChemfilesSystem(frame)
        self.assertTrue(np.all(system.cell() == 0))

        frame.cell = chemfiles.UnitCell((3, 3, 3))
        self.assertTrue(np.allclose(system.cell(), np.diag([3, 3, 3])))

        frame.cell = chemfiles.UnitCell((3, 3, 3), (60, 60, 60))
        expected = [
            [3.0, 0.0, 0.0],
            [1.5, 2.59807621, 0.0],
            [1.5, 0.8660254, 2.44948974],
        ]
        self.assertTrue(np.allclose(system.cell(), expected))

    def test_compute(self):
        frame = chemfiles.Frame()
        frame.add_atom(chemfiles.Atom("C"), (0, 0, 0))
        frame.add_atom(chemfiles.Atom("C"), (0, 1, 0))
        frame.add_atom(chemfiles.Atom("C"), (0, 2, 0))
        frame.add_atom(chemfiles.Atom("C"), (0, 3, 0))

        calculator = DummyCalculator(cutoff=3.4, delta=1, name="")

        with self.assertRaises(RascalError) as cm:
            calculator.compute(frame, use_native_system=False)

        self.assertEqual(
            cm.exception.__cause__.args[0],
            "chemfiles systems can only be used with 'use_native_system=True'",
        )

        # use_native_system=True should work fine
        descriptor = calculator.compute(frame, use_native_system=True)
        self.assertEqual(len(descriptor.keys), 1)
