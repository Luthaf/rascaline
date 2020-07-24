# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline import SortedDistances
from rascaline.calculator import DummyCalculator


class TestDummyCalculator(unittest.TestCase):
    def test_name(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo", gradients=True)
        self.assertEqual(
            calculator.name,
            "dummy test calculator with cutoff: 3.2 - delta: 12"
            " - name: foo - gradients: true",
        )


class TestSortedDistances(unittest.TestCase):
    def test_name(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(calculator.name, "sorted distances vector")

    def test_bad_parameters(self):
        pass
        # TODO: test for error message when passing "wrong" parameters: wrong
        # type, negative cutoff, etc.

    def test_compute(self):
        pass
        # TODO: test that we get expected values
