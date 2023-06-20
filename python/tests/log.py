import pytest

from rascaline import set_logging_callback
from rascaline.calculators import DummyCalculator
from rascaline.log import (
    RASCAL_LOG_LEVEL_INFO,
    RASCAL_LOG_LEVEL_WARN,
    default_logging_callback,
)

from test_systems import SystemForTests


def setup_module(module):
    """setup any state specific to the execution of the given module."""


def teardown_module(module):
    set_logging_callback(default_logging_callback)


def test_log_message():
    recorded_events = []

    def record_log_events(level, message):
        recorded_events.append((level, message))

    set_logging_callback(record_log_events)

    calculator = DummyCalculator(
        cutoff=3.2,
        delta=0,
        name="log-test-info: test info message",
    )
    calculator.compute(SystemForTests())

    message = (
        "rascaline::calculators::dummy_calculator -- "
        "log-test-info: test info message"
    )
    event = (RASCAL_LOG_LEVEL_INFO, message)
    assert event in recorded_events

    calculator = DummyCalculator(
        cutoff=3.2,
        delta=0,
        name="log-test-warn: this is a test warning message",
    )
    calculator.compute(SystemForTests())

    message = (
        "rascaline::calculators::dummy_calculator -- "
        "log-test-warn: this is a test warning message"
    )
    event = (RASCAL_LOG_LEVEL_WARN, message)
    assert event in recorded_events


def test_exception_in_callback():
    def raise_on_log_event(level, message):
        raise Exception("this is an exception")

    set_logging_callback(raise_on_log_event)

    calculator = DummyCalculator(
        cutoff=3.2,
        delta=0,
        name="log-test-warn: testing errors",
    )

    message = "exception raised in logging callback: this is an exception"
    with pytest.warns(Warning, match=message):
        calculator.compute(SystemForTests())
