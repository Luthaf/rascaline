import logging

from ._rascaline import RASCAL_LOG_LEVEL_ERROR, RASCAL_LOG_LEVEL_WARN
from ._rascaline import RASCAL_LOG_LEVEL_INFO, RASCAL_LOG_LEVEL_DEBUG
from ._rascaline import RASCAL_LOG_LEVEL_TRACE

DEFAULT_LOG_LEVEL = RASCAL_LOG_LEVEL_INFO

logging.basicConfig(level=DEFAULT_LOG_LEVEL)


def DEFAULT_LOG_CALLBACK(log_level, message):
    if log_level == RASCAL_LOG_LEVEL_ERROR:
        logging.error(message)
    elif log_level == RASCAL_LOG_LEVEL_WARN:
        logging.warning(message)
    elif log_level == RASCAL_LOG_LEVEL_INFO:
        logging.info(message)
    elif log_level == RASCAL_LOG_LEVEL_DEBUG:
        logging.debug(message)
    elif log_level == RASCAL_LOG_LEVEL_TRACE:
        logging.debug(message)
    else:
        raise ValueError(f"Log level {log_level} is not supported.")
