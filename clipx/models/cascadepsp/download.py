"""
Download utilities for CascadePSP model.
"""

import os
import requests
import hashlib

# CascadePSP model URL and MD5 checksum
MODEL_URL = "https://github.com/zhangsdong/clipx/releases/download/v0.1.0/model"
MODEL_MD5 = "705806e3f26b039ee17c7f5b851c33ad"


def download_cascadepsp_model():
    """Download the CascadePSP model if not present or invalid."""
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/cascadepsp")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model")

    # Check if valid model already exists
    if os.path.exists(model_path) and _is_valid_model(model_path):
        print("Using existing CascadePSP model")
        return model_path

    # Download model
    print("Downloading CascadePSP model...")
    try:
        with requests.get(MODEL_URL, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Show progress at 25% intervals
                    if total_size > 0 and downloaded % (total_size // 4) < 8192:
                        print(f"Download: {downloaded / total_size * 100:.0f}%")

        # Verify downloaded model
        if not _is_valid_model(model_path):
            os.remove(model_path)
            raise Exception("Model verification failed after download")

        print("CascadePSP model downloaded successfully")
        return model_path
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)  # Clean up partial download
        raise Exception(f"Failed to download CascadePSP model: {e}")


def _is_valid_model(model_path):
    """Check if the model file has valid MD5 checksum"""
    try:
        md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest().lower() == MODEL_MD5.lower()
    except Exception:
        return False


# For backwards compatibility with existing code
def download_and_or_check_model_file(destination):
    """Download model if needed and check its integrity"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if not os.path.exists(destination):
        print("Model file does not exist, downloading...")
        _download_to_destination(destination)

    # Check model integrity
    if not _is_valid_model(destination):
        print("Model checksum failed, re-downloading...")
        _download_to_destination(destination)

        if not _is_valid_model(destination):
            raise Exception("Model verification failed after re-download")


def _download_to_destination(destination):
    """Download the model to the specified destination"""
    try:
        with requests.get(MODEL_URL, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise Exception(f"Download failed: {e}")