# -*- coding: utf-8 -*-
import unittest

import numpy as np

from rascaline import Indexes, RascalError, SortedDistances
from rascaline.calculators import DummyCalculator

from test_systems import TestSystem


class TestDummyCalculator(unittest.TestCase):
    def test_name(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo", gradients=True)
        self.assertEqual(
            calculator.name,
            "dummy test calculator with cutoff: 3.2 - delta: 12"
            " - name: foo - gradients: true",
        )

        self.assertEqual(calculator.c_name, "dummy_calculator")

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
            calculator.parameters,
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
        descriptor = calculator.compute(system, use_native_system=False)

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

    def test_compute_multiple_systems(self):
        systems = [TestSystem(), TestSystem(), TestSystem()]
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(systems, use_native_system=False)

        self.assertEqual(descriptor.values.shape, (12, 2))

    def test_compute_partial_samples(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system, use_native_system=False)

        # From a selection scheme, using numpy array indexing
        samples = descriptor.samples[[0, 2]]
        descriptor = calculator.compute(
            system, use_native_system=False, selected_samples=samples
        )

        values = descriptor.values
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (4, 6)))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (9, 2))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

        # Manually constructing the selected samples
        samples = Indexes(
            array=np.array([(0, 0), (0, 3), (0, 1)], dtype=np.int32),
            names=["structure", "center"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_samples=samples
        )

        values = descriptor.values
        self.assertEqual(values.shape, (3, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (5, 5)))
        self.assertTrue(np.all(values[2] == (3, 3)))

        # Only a subset of the variables defined
        samples = Indexes(
            array=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
            names=["center"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_samples=samples
        )

        values = descriptor.values
        self.assertEqual(values.shape, (3, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (5, 5)))
        self.assertTrue(np.all(values[2] == (3, 3)))

        # empty selected samples
        samples = Indexes(
            array=np.array([], dtype=np.int32).reshape(0, 2),
            names=["structure", "center"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_samples=samples
        )

        values = descriptor.values
        self.assertEqual(values.shape, (0, 2))

    def test_compute_partial_samples_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)

        samples = Indexes(
            array=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
            names=["bad name"],
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_samples=samples
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: got an invalid column name ('bad name') "
            "in selected indexes",
        )

        samples = Indexes(
            array=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
            names=["bad_name"],
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_samples=samples
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: 'bad_name' in requested samples is not part "
            "of the samples of this calculator",
        )

    def test_compute_partial_features(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system, use_native_system=False)

        # From a selection scheme, using numpy array indexing
        features = descriptor.features[[1]]
        descriptor = calculator.compute(
            system, use_native_system=False, selected_features=features
        )

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
        features = Indexes(
            array=np.array([[1, 0]], dtype=np.int32),
            names=["index_delta", "x_y_z"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_features=features
        )

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

        # Only a subset of the variables defined
        features = Indexes(
            array=np.array([1], dtype=np.int32).reshape(1, 1),
            names=["index_delta"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_features=features
        )

        values = descriptor.values
        self.assertEqual(values.shape, (4, 1))

        # empty selected features
        features = Indexes(
            array=np.array([], dtype=np.int32).reshape(0, 2),
            names=["index_delta", "x_y_z"],
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_features=features
        )

        values = descriptor.values
        self.assertEqual(values.shape, (4, 0))

    def test_compute_partial_features_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)

        features = Indexes(
            array=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
            names=["bad name"],
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_features=features
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: got an invalid column name ('bad name') in "
            "selected indexes",
        )

        features = Indexes(
            array=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
            names=["bad_name"],
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_features=features
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: 'bad_name' in requested features is not part "
            "of the features of this calculator",
        )

    def test_features_count(self):
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        self.assertEqual(calculator.features_count(), 2)


class TestSortedDistances(unittest.TestCase):
    def test_name(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(calculator.name, "sorted distances vector")
        self.assertEqual(calculator.c_name, "sorted_distances")

    def test_parameters(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(
            calculator.parameters, """{"cutoff": 3.5, "max_neighbors": 12}"""
        )
