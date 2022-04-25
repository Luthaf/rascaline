# -*- coding: utf-8 -*-
import unittest

from test_systems import TestSystem

from rascaline import set_logging_callback
from rascaline.calculators import DummyCalculator
from rascaline.log import (
    RASCAL_LOG_LEVEL_INFO,
    RASCAL_LOG_LEVEL_WARN,
    default_logging_callback,
    )


class TestLogging(unittest.TestCase):
    @staticmethod
    def tearDownClass():
        set_logging_callback(default_logging_callback)

    def test_log_message(self):
        recorded_events = []

        def record_log_events(level, message):
            recorded_events.append((level, message))

        set_logging_callback(record_log_events)

        calculator = DummyCalculator(
            cutoff=3.2,
            delta=0,
            name="log-test-info: test info message",
            gradients=False,
        )
        _ = calculator.compute(TestSystem())

        message = (
            "rascaline::calculators::dummy_calculator -- "
            "log-test-info: test info message"
        )
        event = (RASCAL_LOG_LEVEL_INFO, message)
        self.assertTrue(event in recorded_events)

        calculator = DummyCalculator(
            cutoff=3.2,
            delta=0,
            name="log-test-warn: this is a test warning message",
            gradients=False,
        )
        _ = calculator.compute(TestSystem())

        message = (
            "rascaline::calculators::dummy_calculator -- "
            "log-test-warn: this is a test warning message"
        )
        event = (RASCAL_LOG_LEVEL_WARN, message)
        self.assertTrue(event in recorded_events)

    def test_exception_in_callback(self):
        def raise_on_log_event(level, message):
            raise Exception("this is an exception")

        set_logging_callback(raise_on_log_event)

        calculator = DummyCalculator(
            cutoff=3.2,
            delta=0,
            name="log-test-warn: testing errors",
            gradients=False,
        )

        with self.assertWarns(Warning) as cm:
            _ = calculator.compute(TestSystem())

        self.assertEqual(
            cm.warnings[0].message.args[0],
            "exception raised in logging callback: this is an exception",
        )
