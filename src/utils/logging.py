import logging
import logging.handlers
import sys


def setup_logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.handlers.WatchedFileHandler("task.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
