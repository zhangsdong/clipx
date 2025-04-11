"""
Logging utilities for clipx.
"""

import logging
from typing import Union, Optional


# Create a NullHandler to prevent "No handler found" warnings
class NullHandler(logging.Handler):
    def emit(self, record):
        pass


# Get the logger for the clipx package
logger = logging.getLogger("clipx")
logger.addHandler(NullHandler())

# Default log level
logger.setLevel(logging.WARNING)


def set_log_level(level: Union[int, str]):
    """
    Set the log level for the clipx package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logger.setLevel(level)


def enable_console_logging(level: Optional[int] = None):
    """
    Enable logging to console for the clipx package.
    This is intended for users who want to see clipx logs but don't have
    a logging configuration of their own.

    Args:
        level: Optional logging level (defaults to INFO if not specified)
    """
    # Remove existing handlers of the same type to avoid duplicates
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Add new handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set level if specified, otherwise use INFO
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.INFO)