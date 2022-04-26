# -*- coding: utf-8 -*-
from ._rascaline import RASCAL_SUCCESS
from .clib import _get_library


class RascalError(Exception):
    """Throw exceptions for all errors in rascaline."""

    def __init__(self, message, status=None):
        super(Exception, self).__init__(message)

        self.message = message
        """``str``, error message for this exception"""

        self.status = status
        """``Optional[int]``, status code for this exception"""


LAST_EXCEPTION = None


def _save_exception(e):
    global LAST_EXCEPTION
    LAST_EXCEPTION = e


def _check_rascal_status_t(status):
    if status == RASCAL_SUCCESS:
        return
    elif status > RASCAL_SUCCESS:
        raise RascalError(last_error(), status)
    elif status < RASCAL_SUCCESS:
        global LAST_EXCEPTION
        e = LAST_EXCEPTION
        LAST_EXCEPTION = None
        raise RascalError(last_error(), status) from e


def _check_rascal_pointer(pointer):
    if not pointer:
        raise RascalError(last_error())


def last_error():
    """Get the last error message on this thread."""
    lib = _get_library()
    message = lib.rascal_last_error()
    return message.decode("utf8")
