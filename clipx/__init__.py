from clipx.core import Clipx
from clipx.logging import set_log_level, enable_console_logging
from clipx.api import (
    remove_background,
    generate_mask,
    process_image,
    batch_process,
    extract_foreground,
    get_available_devices
)

__version__ = '0.2.0'
__all__ = [
    'Clipx',
    'set_log_level',
    'enable_console_logging',
    'remove_background',
    'generate_mask',
    'process_image',
    'batch_process',
    'extract_foreground',
    'get_available_devices'
]