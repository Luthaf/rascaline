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


class TestSortedDistances(unittest.TestCase):
    def test_name(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(calculator.name, "sorted distances vector")

    def test_parameters(self):
        calculator = SortedDistances(cutoff=3.5, max_neighbors=12)
        self.assertEqual(
            calculator.parameters(), """{"cutoff": 3.5, "max_neighbors": 12}"""
        )

    def test_compute(self):
        pass
        # TODO: test that we get expected values
