"""
Logging utilities for clipx.
"""

import logging


# Create a NullHandler to prevent "No handler found" warnings
class NullHandler(logging.Handler):
    def emit(self, record):
        pass


# Get the logger for the clipx package
logger = logging.getLogger("clipx")
logger.addHandler(NullHandler())

# Default log level
logger.setLevel(logging.WARNING)


def set_log_level(level):
    """
    Set the log level for the clipx package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logger.setLevel(level)


def enable_console_logging():
    """
    Enable logging to console for the clipx package.
    This is intended for users who want to see clipx logs but don't have
    a logging configuration of their own.
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)