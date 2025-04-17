import numpy as np
import cv2
import torch
from PIL import Image
from typing import Union, Optional


def get_default_device() -> str:
    """Returns 'cuda:0' if available, else 'cpu'"""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


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
    # 延迟导入U2Net
    from .models.u2net.main import U2NetPredictor

    # Handle image input
    if isinstance(image, str):
        image = Image.open(image)

    # Step 1: Get initial mask using U2Net
    predictor = U2NetPredictor()
    mask = predictor.predict_mask(image)

    # Convert to numpy arrays for processing
    # Ensure image is in RGB format for processing
    img_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    # Step 2: Refine mask if requested
    if refine_mask:
        # 仅在需要时才导入和初始化Refiner
        from .models.cascadepsp.main import Refiner

        # Use default device if none specified
        if device is None:
            device = get_default_device()
        refiner = Refiner(device=device)
        refined_mask = refiner.refine(img_np, mask_np, fast=fast, L=L)
    else:
        refined_mask = mask_np

    # Return only the mask if requested
    if only_mask:
        return Image.fromarray(refined_mask)

    # Step 3: Create transparent image
    refined_mask_binary = (refined_mask > 127).astype(np.uint8)

    # Use the original image format for the final result
    img_original = np.array(image)

    # Handle different image formats
    if len(img_original.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    elif img_original.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_original, cv2.COLOR_RGBA2BGR)
    elif img_original.shape[2] == 3:  # RGB
        img_cv = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    else:
        # Fallback for other formats
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    foreground = cv2.bitwise_and(img_cv, img_cv, mask=refined_mask_binary)
    result_with_alpha = np.dstack((foreground, refined_mask_binary * 255))
    result = Image.fromarray(cv2.cvtColor(result_with_alpha, cv2.COLOR_BGRA2RGBA))

    return result