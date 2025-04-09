# clipx/models/u2net/download.py
"""
Download utilities for U2Net model.
"""

import os
import requests
from pathlib import Path
import hashlib

class ClipxError(Exception):
    """Base exception class for clipx errors"""
    pass

# U2Net model URL and MD5 checksum
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"

def get_u2net_home():
    """
    Get the home directory for U2Net model.

    Returns:
        Path: U2Net model directory
    """
    # Default to ~/.clipx/u2net
    home_dir = os.path.expanduser("~/.clipx/u2net")
    os.makedirs(home_dir, exist_ok=True)
    return home_dir

def download_file(url, file_path):
    """
    Download a file from URL with progress reporting.

    Args:
        url: URL to download
        file_path: Path to save the file
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            print(f"Downloading {url} to {file_path}")
            print(f"File size: {total_size / (1024*1024):.2f} MB")

            downloaded_size = 0
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Log progress every 10%
                    progress = downloaded_size / total_size
                    if total_size > 0 and downloaded_size % (total_size // 10) == 0:
                        print(f"Download progress: {progress*100:.1f}%")

        print(f"Download completed: {file_path}")
    except Exception as e:
        raise ClipxError(f"Failed to download file from {url}: {e}")

def calculate_md5(file_path):
    """
    Calculate MD5 checksum of a file.

    Args:
        file_path: Path to the file

    Returns:
        str: MD5 hexadecimal digest
    """
    try:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        raise ClipxError(f"Failed to calculate MD5 for {file_path}: {e}")

def download_u2net_model():
    """
    Download the U2Net model if not present.

    Returns:
        str: Path to the U2Net model file
    """
    model_dir = get_u2net_home()
    model_path = os.path.join(model_dir, "u2net.onnx")

    # If model already exists, verify checksum
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")

        # Verify checksum
        if os.environ.get('MODEL_CHECKSUM_DISABLED') is None:
            print("Verifying model checksum")
            try:
                file_md5 = calculate_md5(model_path)

                if file_md5.lower() == U2NET_MD5.lower():
                    print("Checksum verification passed")
                else:
                    print(f"Checksum mismatch: {file_md5} != {U2NET_MD5}")
                    print("Re-downloading model file")
                    download_file(U2NET_URL, model_path)

                    # Verify again
                    file_md5 = calculate_md5(model_path)
                    if file_md5.lower() != U2NET_MD5.lower():
                        raise ClipxError(f"Downloaded model checksum mismatch: {file_md5} != {U2NET_MD5}")
            except ClipxError:
                raise
            except Exception as e:
                raise ClipxError(f"Error verifying model checksum: {e}")
    else:
        # Download model
        print(f"Model not found, downloading to {model_path}")
        try:
            download_file(U2NET_URL, model_path)

            # Verify checksum
            if os.environ.get('MODEL_CHECKSUM_DISABLED') is None:
                print("Verifying downloaded model checksum")
                file_md5 = calculate_md5(model_path)

                if file_md5.lower() != U2NET_MD5.lower():
                    raise ClipxError(f"Downloaded model checksum mismatch: {file_md5} != {U2NET_MD5}")
        except ClipxError:
            raise
        except Exception as e:
            raise ClipxError(f"Error downloading model: {e}")

    return model_path
