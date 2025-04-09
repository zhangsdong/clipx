"""
Utility functions for clipx.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not found, defaulting to CPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False