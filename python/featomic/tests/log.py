import pytest

from featomic import set_logging_callback
from featomic.calculators import DummyCalculator
from featomic.log import (
    FEATOMIC_LOG_LEVEL_INFO,
    FEATOMIC_LOG_LEVEL_WARN,
    default_logging_callback,
)

from .test_systems import SystemForTests


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
        "featomic::calculators::dummy_calculator -- " "log-test-info: test info message"
    )
    event = (FEATOMIC_LOG_LEVEL_INFO, message)
    assert event in recorded_events

    calculator = DummyCalculator(
        cutoff=3.2,
        delta=0,
        name="log-test-warn: this is a test warning message",
    )
    calculator.compute(SystemForTests())

    message = (
        "featomic::calculators::dummy_calculator -- "
        "log-test-warn: this is a test warning message"
    )
    event = (FEATOMIC_LOG_LEVEL_WARN, message)
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
