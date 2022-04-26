import logging
import warnings

from ._rascaline import (
    RASCAL_LOG_LEVEL_DEBUG,
    RASCAL_LOG_LEVEL_ERROR,
    RASCAL_LOG_LEVEL_INFO,
    RASCAL_LOG_LEVEL_TRACE,
    RASCAL_LOG_LEVEL_WARN,
    rascal_logging_callback_t,
)


_CURRENT_CALLBACK = None


def default_logging_callback(level, message):
    """Default callback function, to redirect message to the ``logging`` module."""
    if level == RASCAL_LOG_LEVEL_ERROR:
        logging.error(message)
    elif level == RASCAL_LOG_LEVEL_WARN:
        logging.warning(message)
    elif level == RASCAL_LOG_LEVEL_INFO:
        logging.info(message)
    elif level == RASCAL_LOG_LEVEL_DEBUG:
        logging.debug(message)
    elif level == RASCAL_LOG_LEVEL_TRACE:
        logging.debug(message)
    else:
        raise ValueError(f"Log level {level} is not supported.")


def set_logging_callback(function):
    """Call ``function`` on every log event.

    The callback functions should take two arguments: an integer value
    representing the log level and a string containing the log message. The
    function return value is ignored.
    """
    from .clib import _get_library

    library = _get_library()
    _set_logging_callback_impl(library, function)


def _set_logging_callback_impl(library, function):
    """Logging callback implementation.

    Implementation of :py:func:`set_logging_callback` getting the instance of
    :py:class:`ctypes.CDLL` for ``librascaline`` as a parameter.

    This is used to setup the default logging callback when loading the library,
    without a recursive call to :py:func:`_get_library` in this function.
    """

    def wrapper(log_level, message):
        try:
            function(log_level, message.decode("utf8"))
        except Exception as e:
            warnings.warn(f"exception raised in logging callback: {e}", ResourceWarning)

    # store the current callback in a global python variable to prevent it from
    # being garbage-collected.
    global _CURRENT_CALLBACK
    _CURRENT_CALLBACK = rascal_logging_callback_t(wrapper)

    library.rascal_set_logging_callback(_CURRENT_CALLBACK)
