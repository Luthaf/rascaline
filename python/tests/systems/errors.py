# -*- coding: utf-8 -*-
import unittest

from rascaline import RascalError
from rascaline.systems import SystemBase
from rascaline.calculator import DummyCalculator


class UnimplementedSystem(SystemBase):
    pass


class TestSystemExceptions(unittest.TestCase):
    def test_unimplemented(self):
        system = UnimplementedSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)

        with self.assertRaises(RascalError) as cm:
            _ = calculator.compute(system)

        self.assertEqual(
            cm.exception.args[0],
            "error from external code (status -1): call to rascal_system_t.size failed",
        )

        self.assertEqual(
            cm.exception.__cause__.args[0],
            "System.size method is not implemented",
        )
