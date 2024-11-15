from ._c_api import FEATOMIC_SUCCESS
from ._c_lib import _get_library


class FeatomicError(Exception):
    """Exceptions thrown for all errors in featomic."""

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


def _check_featomic_status_t(status):
    if status == FEATOMIC_SUCCESS:
        return
    elif status > FEATOMIC_SUCCESS:
        raise FeatomicError(last_error(), status)
    elif status < FEATOMIC_SUCCESS:
        global LAST_EXCEPTION
        e = LAST_EXCEPTION
        LAST_EXCEPTION = None
        raise FeatomicError(last_error(), status) from e


def _check_featomic_pointer(pointer):
    if not pointer:
        raise FeatomicError(last_error())


def last_error():
    """Get the last error message on this thread."""
    lib = _get_library()
    message = lib.featomic_last_error()
    return message.decode("utf8")
