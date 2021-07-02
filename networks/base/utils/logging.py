import logging

import torch.distributed as dist

logger_initialized = {}

VERBOSE_LOG_FORMAT = ('%(asctime)s | %(levelname)s | pid-%(process)d | '
                   '%(filename)s:<%(funcName)s>:%(lineno)d | %(message)s')
BRIEF_LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'

LEVEL_DICT = dict(
    NOTSET=logging.NOTSET,      #  0
    DEBUG=logging.DEBUG,        # 10
    INFO=logging.INFO,          # 20
    WARNING=logging.WARNING,    # 30
    ERROR=logging.ERROR,        # 40
    CRITICAL=logging.CRITICAL,  # 50
)

VERBOSE_LEVELS = [logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]


def get_logger(name, log_file=None, log_level='INFO'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str, optional
        The log filename. If specified, a FileHandler will be added to the logger.
    log_level : str
        The logger level.

    Returns
    -------
    logger : logging.Logger
        The expected logger.
    """
    assert isinstance(log_level, str) and log_level.upper() in LEVEL_DICT

    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    log_level = LEVEL_DICT[log_level.upper()]
    if log_level in VERBOSE_LEVELS:
        log_format = VERBOSE_LOG_FORMAT
    else:
        log_format = BRIEF_LOG_FORMAT

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    formatters = [logging.Formatter(log_format)]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
        formatters.append(logging.Formatter(log_format))

    for handler, formatter in zip(handlers, formatters):
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Parameters
    ----------
    msg : str
        The message to be logged.
    logger : {logging.Logger, str}, optional
        The logger to be used.
        Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
    level : int
        Logging level. Only available when `logger` is a Logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
