import os
import requests
import hashlib


def download_model(destination):
    """Download the model file"""
    print('Downloading model file into', destination)
    URL = 'https://github.com/zhangsdong/clipx/releases/download/v0.1.0/u2net.onnx'

    # Ensure directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, 'wb') as file:
        response = requests.get(URL)
        file.write(response.content)

    print('Download completed.')


def check_model(destination):
    """Check if the model file has the correct MD5 hash"""
    correct_md5 = '60024c5c889badc19c04ad937298a77b'

    with open(destination, 'rb') as f:
        md5_returned = hashlib.md5(f.read()).hexdigest()

    return correct_md5 == md5_returned


def download_and_or_check_model_file(destination):
    """Download and/or check the model file"""
    if os.path.exists(destination):
        # Model exists, no output needed
        pass
    else:
        print('Model does not exist.')
        download_model(destination)

    md5_passed = check_model(destination)

    if md5_passed:
        return

    print('MD5 of the model file does not match')
    print('Downloading the model again...')
    download_model(destination)

    md5_passed = check_model(destination)
    assert md5_passed, 'MD5 still does not pass'