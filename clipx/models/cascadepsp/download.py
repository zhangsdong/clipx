"""
Download utilities for CascadePSP model.
"""

import os
import requests
import hashlib
import logging
from typing import Optional

# Get logger
logger = logging.getLogger("clipx.models.cascadepsp.download")

# CascadePSP model URL and MD5 checksum
MODEL_URL = "https://github.com/zhangsdong/clipx/releases/download/v0.1.0/model"
MODEL_MD5 = "705806e3f26b039ee17c7f5b851c33ad"

def get_model_path(skip_checksum: bool = False) -> str:
    """
    Get the path to the CascadePSP model file, downloading it if necessary

    Args:
        skip_checksum: Whether to skip MD5 checksum verification

    Returns:
        str: Path to the model file
    """
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/cascadepsp")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model")

    # Check if model exists and is valid
    if os.path.exists(model_path) and (skip_checksum or _is_valid_model(model_path)):
        logger.debug("Using existing CascadePSP model")
        return model_path

    # Download if not exists or invalid
    return download_cascadepsp_model(skip_checksum)

def download_cascadepsp_model(skip_checksum: bool = False) -> str:
    """
    Download the CascadePSP model

    Args:
        skip_checksum: Whether to skip MD5 checksum verification

    Returns:
        str: Path to the model file
    """
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/cascadepsp")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model")

    # Backup existing file if it exists but fails checksum
    if os.path.exists(model_path) and not skip_checksum and not _is_valid_model(model_path):
        backup_path = f"{model_path}.bak"
        logger.warning(f"Existing model failed checksum verification. Backing up to {backup_path}")
        try:
            os.rename(model_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup existing model: {e}")

    # Download model
    logger.info("Downloading CascadePSP model...")
    try:
        with requests.get(MODEL_URL, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Only show progress at 25% intervals
                    if total_size > 0 and downloaded % (total_size // 4) < 8192:
                        logger.debug(f"Download progress: {downloaded / total_size * 100:.0f}%")

        # Verify downloaded model
        if not skip_checksum and not _is_valid_model(model_path):
            logger.error("Model verification failed after download")
            # Try to restore backup
            if os.path.exists(f"{model_path}.bak"):
                logger.info("Restoring backup model")
                os.rename(f"{model_path}.bak", model_path)
            else:
                os.remove(model_path)  # Clean up partial download
            raise Exception("Model verification failed after download")

        logger.info("CascadePSP model downloaded successfully")
        # Remove backup if download successful
        if os.path.exists(f"{model_path}.bak"):
            os.remove(f"{model_path}.bak")

        return model_path
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)  # Clean up partial download
        # Restore backup if exists
        if os.path.exists(f"{model_path}.bak"):
            logger.info("Restoring backup model")
            os.rename(f"{model_path}.bak", model_path)

        raise Exception(f"Failed to download CascadePSP model: {e}")

def _is_valid_model(model_path: str) -> bool:
    """
    Check if the model file has valid MD5 checksum

    Args:
        model_path: Path to the model file

    Returns:
        bool: True if checksum is valid, False otherwise
    """
    try:
        md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        is_valid = md5.hexdigest().lower() == MODEL_MD5.lower()
        if not is_valid:
            logger.warning("Model checksum verification failed")
        return is_valid
    except Exception as e:
        logger.warning(f"Error during checksum verification: {e}")
        return False