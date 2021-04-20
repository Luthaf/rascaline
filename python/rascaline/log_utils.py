# Here we can build the logging utility

# enums as integers https://docs.rs/log/0.4.0/log/enum.LevelFilter.html
RUST_LOG_LEVEL_OFF = 0
RUST_LOG_LEVEL_ERROR = 1
RUST_LOG_LEVEL_WARN = 2
RUST_LOG_LEVEL_INFO = 3
RUST_LOG_LEVEL_DEBUG = 4
RUST_LOG_LEVEL_TRACE = 5

DEFAULT_LOG_LEVEL = RUST_LOG_LEVEL_WARN


def DEFAULT_LOG_CALLBACK(log_level, message):
    if log_level == RUST_LOG_LEVEL_WARN:
        print('WARN:' + message)
    elif log_level == RUST_LOG_LEVEL_INFO:
        print('INFO:' + message)
    elif log_level == RUST_LOG_LEVEL_DEBUG:
        print('DEBUG:' + message)
    elif log_level == RUST_LOG_LEVEL_TRACE:
        print('TRACE:' + message)
    else:
        raise ValueError('Log level ' + str(log_level) + ' is not supported.')

# just to illustrate the next step how initialize the logger
# import logging
#def DEFAULT_LOG(log_level, message):
#    if log_level == RUST_LOG_LEVEL_WARN:
#        logging.error(message)
#    elif log_level == RUST_LOG_LEVEL_WARN:
#        logging.warning(message)
#    elif log_level == RUST_LOG_LEVEL_INFO:
#        logging.info(message)
#    elif log_level == RUST_LOG_LEVEL_DEBUG:
#        logging.debug(message)
#    elif log_level == RUST_LOG_LEVEL_TRACE:
#        print(message)
#    else:
#        raise ValueError('Log level ' + str(log_level) + ' is not supported.')
# logging.basicConfig(filename='rascaline.log', level=DEFAULT_LOG_LEVEL)

