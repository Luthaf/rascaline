# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline import SortedDistances
from rascaline.calculator import DummyCalculator

from rascaline.clib import (_get_library,
        set_logging_callback,
        _set_default_logging_callback)
from rascaline.log_utils import (
        RUST_LOG_LEVEL_WARN,
        RUST_LOG_LEVEL_INFO)
import ctypes

from test_systems import TestSystem

class TestDummyCalculator(unittest.TestCase):
    def test_log_levels(self):
        # tests if log levels can be set without error
        def dummy_callback_function(level, message):
            pass
        set_logging_callback(dummy_callback_function)
        _set_default_logging_callback()

    def test_log_message_in_calculator(self):
        # tests if targeted message is is recorded when running compute
        # function of the calculator and checks if the log level is correct
        recorded_levels = []
        recorded_messages = []

        def record_callback_function(level, message):
            recorded_levels.append(level)
            recorded_messages.append(message)
        set_logging_callback(record_callback_function)

        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        # in the compute function of the dummy calculator a message is put into the info log
        descriptor = calculator.compute(system)

        targeted_info_message = "rascaline::calculators::dummy_calculator -- " \
                "this is an info message used for testing purposes, do not remove"
        log_level = recorded_levels[recorded_messages.index(targeted_info_message)]
        self.assertEqual(log_level, RUST_LOG_LEVEL_INFO)

        targeted_info_message = "rascaline::calculators::dummy_calculator -- " \
                "this is a warning message used for testing purposes, do not remove"
        log_level = recorded_levels[recorded_messages.index(targeted_info_message)]
        self.assertEqual(log_level, RUST_LOG_LEVEL_WARN)

        _set_default_logging_callback()

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

    def test_compute_multiple_systems(self):
        systems = [TestSystem(), TestSystem(), TestSystem()]
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(systems)

        self.assertEqual(descriptor.values.shape, (12, 2))

    def test_compute_partial_samples(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="", gradients=True)
        descriptor = calculator.compute(system)

        # From a selection scheme, using numpy array indexing
        samples = descriptor.samples[[0, 2]]
        descriptor = calculator.compute(system, selected_samples=samples)

        values = descriptor.values
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.all(values[0] == (2, 1)))
        self.assertTrue(np.all(values[1] == (4, 6)))

        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (9, 2))
        for i in range(gradients.shape[0]):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

        #  Manually constructing the selected samples
        samples = [(0, 0), (0, 3), (0, 1)]
        descriptor = calculator.compute(system, selected_samples=samples)

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
        descriptor = calculator.compute(system, selected_features=features)

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
        descriptor = calculator.compute(system, selected_features=features)

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
            calculator.parameters, """{"cutoff": 3.5, "max_neighbors": 12}"""
        )
