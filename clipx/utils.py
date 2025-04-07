import os
import numpy as np
from PIL import Image


def is_image_file(filename):
    """
    Check if file is an image based on extension
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def load_image(image_path):
    """
    Load image from path
    """
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")


def load_mask(mask_path):
    """
    Load mask image from path
    """
    if mask_path is None:
        return None

    try:
        mask = Image.open(mask_path)
        # Convert to grayscale if not already
        if mask.mode != 'L':
            mask = mask.convert('L')
        return mask
    except Exception as e:
        raise ValueError(f"Error loading mask {mask_path}: {e}")


def save_image(image, output_path):
    """
    Save image to path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(image)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        pil_image = image

    # Save the image
    try:
        pil_image.save(output_path)
    except Exception as e:
        raise ValueError(f"Error saving image to {output_path}: {e}")


def apply_mask_to_image(image, mask):
    """
    Apply mask to image to create transparent background

    Args:
        image: PIL Image or numpy array (RGB)
        mask: PIL Image or numpy array (grayscale)

    Returns:
        PIL Image with alpha channel
    """
    # Convert to PIL Images if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    # Convert image to RGBA if it's not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Ensure mask is in correct mode
    if mask.mode != 'L':
        mask = mask.convert('L')

    # Resize mask to match image if needed
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.LANCZOS)

    # Apply mask as alpha channel
    image.putalpha(mask)

    return image