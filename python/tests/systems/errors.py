# -*- coding: utf-8 -*-
import unittest

from rascaline import RascalError
from rascaline.calculators import DummyCalculator
from rascaline.systems import SystemBase


class UnimplementedSystem(SystemBase):
    pass


class TestSystemExceptions(unittest.TestCase):
    def test_unimplemented(self):
        system = UnimplementedSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        with self.assertRaises(RascalError) as cm:
            _ = calculator.compute(system, use_native_system=False)

        self.assertEqual(
            cm.exception.args[0],
            "error from external code (status -1): call to rascal_system_t.species failed",  # noqa
        )

        self.assertEqual(
            cm.exception.__cause__.args[0],
            "System.species method is not implemented",
        )
