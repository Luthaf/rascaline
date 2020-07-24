# -*- coding: utf-8 -*-
from ._rascaline import rascal_status_t
from .clib import _get_library


def _check_rascal_status_t(status):
    if status == rascal_status_t.RASCAL_SUCCESS.value:
        return
    else:
        raise Exception(last_error())


def _check_rascal_pointer(pointer):
    try:
        pointer.contents
    except ValueError:
        raise Exception(last_error())


def last_error():
    lib = _get_library()
    message = lib.rascal_last_error()
    return message.decode("utf8")
