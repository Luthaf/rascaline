import logging
import warnings

from ._c_api import (
    FEATOMIC_LOG_LEVEL_DEBUG,
    FEATOMIC_LOG_LEVEL_ERROR,
    FEATOMIC_LOG_LEVEL_INFO,
    FEATOMIC_LOG_LEVEL_TRACE,
    FEATOMIC_LOG_LEVEL_WARN,
    featomic_logging_callback_t,
)


_CURRENT_CALLBACK = None


def default_logging_callback(level, message):
    """Redirect message to the ``logging`` module."""
    if level == FEATOMIC_LOG_LEVEL_ERROR:
        logging.error(message)
    elif level == FEATOMIC_LOG_LEVEL_WARN:
        logging.warning(message)
    elif level == FEATOMIC_LOG_LEVEL_INFO:
        logging.info(message)
    elif level == FEATOMIC_LOG_LEVEL_DEBUG:
        logging.debug(message)
    elif level == FEATOMIC_LOG_LEVEL_TRACE:
        logging.debug(message)
    else:
        raise ValueError(f"Log level {level} is not supported.")


def set_logging_callback(function):
    """Call ``function`` on every log event.

    The callback functions should take two arguments: an integer value
    representing the log level and a string containing the log message. The
    function return value is ignored.
    """
    from ._c_lib import _get_library

    library = _get_library()
    _set_logging_callback_impl(library, function)


def _set_logging_callback_impl(library, function):
    """Implementation of :py:func:`set_logging_callback`

    This function gets the :py:class:`ctypes.CDLL` instance for ``libfeatomic``
    as a parameter.

    This is used to be able to setup the default logging callback when loading
    the library, without a recursive call to :py:func:`_get_library` in this
    function.
    """

    def wrapper(log_level, message):
        try:
            function(log_level, message.decode("utf8"))
        except Exception as e:
            warnings.warn(
                message=f"exception raised in logging callback: {e}",
                category=ResourceWarning,
                stacklevel=1,
            )

    # store the current callback in a global python variable to prevent it from
    # being garbage-collected.
    global _CURRENT_CALLBACK
    _CURRENT_CALLBACK = featomic_logging_callback_t(wrapper)

    library.featomic_set_logging_callback(_CURRENT_CALLBACK)
