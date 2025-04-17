# Core functionality and utilities
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Union, Optional
from functools import lru_cache

# Global variables for model caching
_u2net_model = None
_refiner_models = {}


# Model management functions
def get_u2net_model():
    """Returns singleton instance of U2NetPredictor"""
    global _u2net_model
    if _u2net_model is None:
        # Load model only when needed
        from .models.u2net.main import U2NetPredictor
        _u2net_model = U2NetPredictor()
    return _u2net_model


def get_refiner_model(device: str):
    """Returns singleton instance of Refiner for specific device"""
    global _refiner_models
    if device not in _refiner_models:
        # Load model only when needed
        from .models.cascadepsp.main import Refiner
        _refiner_models[device] = Refiner(device=device)
    return _refiner_models[device]


def get_default_device() -> str:
    """Returns 'cuda:0' if available, else 'cpu'"""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def preprocess_image(image: Union[Image.Image, str]) -> tuple:
    """Preprocesses the input image and returns normalized arrays"""
    # Handle image input
    if isinstance(image, str):
        image = Image.open(image)

    # Convert to numpy arrays for processing
    img_rgb = np.array(image.convert("RGB"))

    return image, img_rgb


def create_transparent_output(img_original, refined_mask):
    """Creates transparent output image from original image and refined mask"""
    # Convert mask to binary format
    refined_mask_binary = (refined_mask > 127).astype(np.uint8)

    # Handle different image formats
    img_original_np = np.array(img_original)
    if len(img_original_np.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_original_np, cv2.COLOR_GRAY2BGR)
    elif img_original_np.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_original_np, cv2.COLOR_RGBA2BGR)
    elif img_original_np.shape[2] == 3:  # RGB
        img_cv = cv2.cvtColor(img_original_np, cv2.COLOR_RGB2BGR)
    else:
        # Fallback for other formats
        img_cv = cv2.cvtColor(np.array(img_original.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Apply mask to extract foreground
    foreground = cv2.bitwise_and(img_cv, img_cv, mask=refined_mask_binary)

    # Add alpha channel
    result_with_alpha = np.dstack((foreground, refined_mask_binary * 255))

    # Convert back to PIL Image with RGBA format
    result = Image.fromarray(cv2.cvtColor(result_with_alpha, cv2.COLOR_BGRA2RGBA))

    return result


def remove(
        image: Union[Image.Image, str],
        only_mask: bool = False,
        refine_mask: bool = True,
        device: Optional[str] = None,
        fast: bool = False,
        L: int = 900,
) -> Image.Image:
    """
    Removes background from an image

    Args:
        image: PIL Image or path to image file
        only_mask: If True, returns only the mask
        refine_mask: If True, refines the mask using CascadePSP
        device: Device to use for CascadePSP ('cpu' or 'cuda:0')
        fast: If True, uses fast mode for CascadePSP
        L: Hyperparameter for CascadePSP

    Returns:
        PIL Image with transparent background or mask
    """
    # Preprocess image
    original_image, img_rgb = preprocess_image(image)

    # Step 1: Get initial mask using U2Net (singleton model)
    predictor = get_u2net_model()
    mask = predictor.predict_mask(original_image)
    mask_np = np.array(mask.convert("L"))

    # Step 2: Refine mask if requested
    if refine_mask:
        # Use default device if none specified
        if device is None:
            device = get_default_device()

        # Get singleton refiner model for specific device
        refiner = get_refiner_model(device)
        refined_mask = refiner.refine(img_rgb, mask_np, fast=fast, L=L)
    else:
        refined_mask = mask_np

    # Return only the mask if requested
    if only_mask:
        return Image.fromarray(refined_mask)

    # Step 3: Create transparent image
    return create_transparent_output(original_image, refined_mask)