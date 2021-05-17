import logging

# enums as integers https://docs.rs/log/0.4.0/log/enum.LevelFilter.html
RUST_LOG_LEVEL_OFF = 0
RUST_LOG_LEVEL_ERROR = 1
RUST_LOG_LEVEL_WARN = 2
RUST_LOG_LEVEL_INFO = 3
RUST_LOG_LEVEL_DEBUG = 4
RUST_LOG_LEVEL_TRACE = 5

DEFAULT_LOG_LEVEL = RUST_LOG_LEVEL_INFO

logging.basicConfig(level=DEFAULT_LOG_LEVEL)

def DEFAULT_LOG_CALLBACK(log_level, message):
    if log_level == RUST_LOG_LEVEL_ERROR:
        logging.error(message)
    elif log_level == RUST_LOG_LEVEL_WARN:
        logging.warning(message)
    elif log_level == RUST_LOG_LEVEL_INFO:
        logging.info(message)
    elif log_level == RUST_LOG_LEVEL_DEBUG:
        logging.debug(message)
    elif log_level == RUST_LOG_LEVEL_TRACE:
        print(message)
    else:
        raise ValueError(f'Log level {log_level} is not supported.')
