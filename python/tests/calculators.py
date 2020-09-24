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

        # very long name, checking that we can pass large string back and forth
        name = "abc" * 2048
        calculator = DummyCalculator(cutoff=3.2, delta=12, name=name, gradients=True)
        self.assertEqual(
            calculator.name,
            "dummy test calculator with cutoff: 3.2 - delta: 12"
            f" - name: {name} - gradients: true",
        )

    def test_parameters(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo", gradients=True)
        self.assertEqual(
            calculator.parameters(),
            """{"cutoff": 3.2, "delta": 12, "name": "foo", "gradients": true}""",
        )

    def test_bad_parameters(self):
        message = (
            'json error: invalid type: string "12", expected isize at line 1 column 29'
        )

        with self.assertRaisesRegex(Exception, message):
            _ = DummyCalculator(cutoff=3.2, delta="12", name="foo", gradients=True)

    def test_compute(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system)

        values = descriptor.values
        self.assertEqual(values.shape, (4, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (3, 3)))
        self.assertTrue(np.all(values[2] == (4, 6)))
        self.assertTrue(np.all(values[3] == (5, 5)))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (18, 2))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

    def test_compute_partial_samples(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system)

        # From a selection scheme, using numpy array indexing
        samples = descriptor.environments[[0, 2]]
        descriptor = calculator.compute_partial(system, samples=samples)

        values = descriptor.values
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (4, 6)))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (9, 2))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == (0, 1)))


        # Manually constructing the selected samples
        samples = [(0, 0), (0, 3), (0, 1)]
        descriptor = calculator.compute_partial(system, samples=samples)

        values = descriptor.values
        self.assertEqual(values.shape, (3, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (5, 5)))
        self.assertTrue(np.all(values[2] == (3, 3)))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (12, 2))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

    def test_compute_partial_features(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system)

        # From a selection scheme, using numpy array indexing
        features = descriptor.features[[1]]
        descriptor = calculator.compute_partial(system, features=features)

        values = descriptor.values
        self.assertEqual(values.shape, (4, 1))
        self.assertTrue(np.all(values[0] == [1]))
        self.assertTrue(np.all(values[1] == [3]))
        self.assertTrue(np.all(values[2] == [6]))
        self.assertTrue(np.all(values[3] == [5]))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (18, 1))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == [1]))

        # Manually constructing the selected features
        features = [[1, 0]]
        descriptor = calculator.compute_partial(system, features=features)

        values = descriptor.values
        self.assertEqual(values.shape, (4, 1))
        self.assertTrue(np.all(values[0] == [2]))
        self.assertTrue(np.all(values[1] == [3]))
        self.assertTrue(np.all(values[2] == [4]))
        self.assertTrue(np.all(values[3] == [5]))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (18, 1))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == [0]))


class TestSortedDistances(unittest.TestCase):
    def test_name(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(calculator.name, "sorted distances vector")

    def test_parameters(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(
            calculator.parameters(), """{"cutoff": 3.5, "max_neighbors": 12}"""
        )
