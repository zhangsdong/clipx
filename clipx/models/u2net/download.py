"""
Download utilities for U2Net model.
"""

import os
import requests
import hashlib
import logging
from typing import Optional

# Get logger
logger = logging.getLogger("clipx.models.u2net.download")

class ClipxError(Exception):
    """Base exception class for clipx errors"""
    pass


# U2Net model URL and MD5 checksum
U2NET_URL = "https://github.com/zhangsdong/clipx/releases/download/v0.1.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"


def get_model_path(skip_checksum: bool = False) -> str:
    """
    Get the path to the U2Net model file, downloading it if necessary.

    Args:
        skip_checksum: Whether to skip MD5 checksum verification

    Returns:
        str: Path to the U2Net model file
    """
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/u2net")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "u2net.onnx")

    # Check if model exists and is valid
    skip_checksum = skip_checksum or os.environ.get('MODEL_CHECKSUM_DISABLED') is not None
    if os.path.exists(model_path) and (skip_checksum or _is_valid_model(model_path)):
        logger.debug("Using existing U2Net model")
        return model_path

    # Download if not exists or invalid
    return download_u2net_model(skip_checksum)


def download_u2net_model(skip_checksum: bool = False) -> str:
    """
    Download the U2Net model.

    Args:
        skip_checksum: Whether to skip MD5 checksum verification

    Returns:
        str: Path to the U2Net model file
    """
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/u2net")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "u2net.onnx")

    # Backup existing file if it exists but fails checksum
    if os.path.exists(model_path) and not skip_checksum and not _is_valid_model(model_path):
        backup_path = f"{model_path}.bak"
        logger.warning(f"Existing model failed checksum verification. Backing up to {backup_path}")
        try:
            os.rename(model_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup existing model: {e}")

    # Download model
    logger.info("Downloading U2Net model...")
    try:
        with requests.get(U2NET_URL, stream=True) as response:
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
            if os.path.exists(f"{model_path}.bak"):
                logger.info("Restoring backup model")
                os.rename(f"{model_path}.bak", model_path)
            else:
                os.remove(model_path)  # Clean up partial download
            raise ClipxError("Model verification failed after download")

        logger.info("U2Net model downloaded successfully")
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

        raise ClipxError(f"Failed to download U2Net model: {e}")


def _is_valid_model(model_path: str) -> bool:
    """
    Check if the model file has valid MD5 checksum.

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
        is_valid = md5.hexdigest().lower() == U2NET_MD5.lower()
        if not is_valid:
            logger.warning("Model checksum verification failed")
        return is_valid
    except Exception as e:
        logger.warning(f"Error during checksum verification: {e}")
        return False