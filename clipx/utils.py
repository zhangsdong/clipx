"""
Utility functions for clipx.
"""

import os
import sys
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger("clipx.utils")

def get_project_root():
    """
    Get the root directory of the project.
    """
    return Path(__file__).parent.parent

def check_gpu_availability():
    """
    Check if CUDA is available.
    """
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            logger.info("CUDA is available")
        else:
            logger.info("CUDA is not available")
        return available
    except ImportError:
        logger.warning("PyTorch not found, defaulting to CPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False