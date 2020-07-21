# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline import SortedDistances
from rascaline.calculator import DummyCalculator

from test_systems import TestSystem


class TestDummyCalculator(unittest.TestCase):
    def test_name(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo", gradients=True)
        self.assertEqual(
            calculator.name,
            "dummy test calculator with cutoff: 3.2 - delta: 12"
            " - name: foo - gradients: true",
        )

    def test_compute(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo", gradients=False)
        descriptor = calculator.compute(system)

        env = descriptor.environments
        self.assertEqual(len(env), 4)
        self.assertEqual(env[0], (0, 0))
        self.assertEqual(env[1], (0, 1))
        self.assertEqual(env[2], (0, 2))
        self.assertEqual(env[3], (0, 3))

        features = descriptor.features
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0], (1, 0))
        self.assertEqual(features[1], (0, 1))

        values = descriptor.values
        self.assertEqual(values.shape, (4, 2))
        self.assertTrue(np.all(values[0] == (12, 0)))
        self.assertTrue(np.all(values[1] == (13, 1)))
        self.assertTrue(np.all(values[2] == (14, 2)))
        self.assertTrue(np.all(values[3] == (15, 3)))


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
