# -*- coding: utf-8 -*-
import ctypes

from ._rascaline import RASCAL_BUFFER_SIZE_ERROR
from .status import RascalError


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except RascalError as e:
            if e.status == RASCAL_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")
