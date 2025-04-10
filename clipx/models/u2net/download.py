"""
Download utilities for U2Net model.
"""

import os
import requests
import hashlib


class ClipxError(Exception):
    """Base exception class for clipx errors"""
    pass


# U2Net model URL and MD5 checksum
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"


def download_u2net_model():
    """
    Download the U2Net model if not present or invalid.

    Returns:
        str: Path to the U2Net model file
    """
    # Setup model path
    model_dir = os.path.expanduser("~/.clipx/u2net")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "u2net.onnx")

    # Check if valid model already exists
    skip_checksum = os.environ.get('MODEL_CHECKSUM_DISABLED') is not None
    if os.path.exists(model_path) and (skip_checksum or _is_valid_model(model_path)):
        print("Using existing U2Net model")
        return model_path

    # Download model
    print("Downloading U2Net model...")
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
                        print(f"Download: {downloaded / total_size * 100:.0f}%")

        # Verify downloaded model
        if not skip_checksum and not _is_valid_model(model_path):
            os.remove(model_path)
            raise ClipxError("Model verification failed after download")

        print("U2Net model downloaded successfully")
        return model_path
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)  # Clean up partial download
        raise ClipxError(f"Failed to download U2Net model: {e}")


def _is_valid_model(model_path):
    """Check if the model file has valid MD5 checksum"""
    try:
        md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        is_valid = md5.hexdigest().lower() == U2NET_MD5.lower()
        if not is_valid:
            print("Model checksum verification failed")
        return is_valid
    except Exception:
        return False