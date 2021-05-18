# -*- coding: utf-8 -*-
import unittest

from rascaline.calculator import DummyCalculator

from rascaline import set_logging_callback
from rascaline.log import default_logging_callback
from rascaline.log import RASCAL_LOG_LEVEL_INFO, RASCAL_LOG_LEVEL_WARN

from test_systems import TestSystem


class TestLogging(unittest.TestCase):
    @staticmethod
    def tearDownClass():
        set_logging_callback(default_logging_callback)

    def test_log_message(self):
        recorded_events = []

        def record_log_events(level, message):
            recorded_events.append((level, message))

        set_logging_callback(record_log_events)

        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=0, name="", gradients=False)
        # in the compute function of the dummy calculator a message is put into
        # the info log
        _ = calculator.compute(system)

        message = (
            "rascaline::calculators::dummy_calculator -- "
            "this is an info message used for testing purposes, do not remove"
        )
        event = (RASCAL_LOG_LEVEL_INFO, message)
        self.assertTrue(event in recorded_events)

        message = (
            "rascaline::calculators::dummy_calculator -- "
            "this is a warning message used for testing purposes, do not remove"
        )
        event = (RASCAL_LOG_LEVEL_WARN, message)
        self.assertTrue(event in recorded_events)

    def test_exception_in_callback(self):
        def raise_on_log_event(level, message):
            raise Exception("this is not good")

        set_logging_callback(raise_on_log_event)

        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=0, name="", gradients=False)

        with self.assertWarns(Warning) as cm:
            _ = calculator.compute(system)

        self.assertEqual(
            cm.warnings[0].message.args[0],
            "exception raised in logging callback: this is not good",
        )
