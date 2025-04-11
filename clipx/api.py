"""
High-level API functions for clipx.
This module provides simple, intuitive interfaces for common tasks.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
from PIL import Image
import time

from clipx.core import Clipx

# Configure logger
logger = logging.getLogger("clipx.api")


def remove_background(
        image_path: str,
        output_path: Optional[str] = None,
        model: str = "combined",
        edge_hardness: int = 50,
        fast_mode: bool = False,
        device: str = "auto",
        skip_checksum: bool = False
) -> str:
    """
    Remove the background from an image and save the result.

    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, uses input_remove.png)
        model: Model to use ('u2net', 'cascadepsp', or 'combined')
        edge_hardness: Edge hardness level (0-100) where higher values create harder edges
        fast_mode: Whether to use fast mode for processing (less accurate but faster)
        device: Device to use ('auto', 'cpu', or 'cuda')
        skip_checksum: Whether to skip MD5 checksum verification for models

    Returns:
        str: Path to the saved output image

    Example:
        # Simple usage
        from clipx.api import remove_background
        result_path = remove_background("input.jpg")

        # With more options
        result_path = remove_background(
            "input.jpg",
            "output.png",
            model="u2net",
            edge_hardness=70
        )
    """
    # Generate default output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_filename = f"{input_path.stem}_remove.png"
        output_path = str(input_path.parent / output_filename)
        logger.debug(f"Using default output path: {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize Clipx and process image
    clipx = Clipx(device=device, skip_checksum=skip_checksum)
    result_path, processing_time = clipx.process(
        input_path=image_path,
        output_path=output_path,
        model=model,
        edge_hardness=edge_hardness,
        fast_mode=fast_mode
    )

    logger.info(f"Background removed in {processing_time:.2f} seconds")
    return result_path


def generate_mask(
        image_path: str,
        output_path: Optional[str] = None,
        model: str = "u2net",
        edge_hardness: int = 50,
        fast_mode: bool = False,
        device: str = "auto",
        skip_checksum: bool = False
) -> str:
    """
    Generate a mask for an image and save it.

    Args:
        image_path: Path to the input image
        output_path: Path to save the output mask (if None, uses input_mask.png)
        model: Model to use ('u2net', 'cascadepsp', or 'combined')
        edge_hardness: Edge hardness level (0-100) where higher values create harder edges
        fast_mode: Whether to use fast mode for processing (less accurate but faster)
        device: Device to use ('auto', 'cpu', or 'cuda')
        skip_checksum: Whether to skip MD5 checksum verification for models

    Returns:
        str: Path to the saved mask image

    Example:
        from clipx.api import generate_mask
        mask_path = generate_mask("image.jpg")
    """
    # Generate default output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_filename = f"{input_path.stem}_mask.png"
        output_path = str(input_path.parent / output_filename)
        logger.debug(f"Using default output path: {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize Clipx
    clipx = Clipx(device=device, skip_checksum=skip_checksum)

    # Load model based on selection
    if model == 'u2net' or model == 'combined':
        u2net = clipx.load_u2net()
    elif model == 'cascadepsp':
        cascadepsp = clipx.load_cascadepsp()

    # Load input image
    start_time = time.time()
    img = Image.open(image_path).convert("RGB")
    logger.debug(f"Image loaded: {img.size[0]}x{img.size[1]}")

    # Generate mask based on model selection
    if model == 'u2net':
        mask = u2net.predict_mask(img)
    elif model == 'cascadepsp':
        mask = cascadepsp.predict_mask(img, fast=fast_mode)
    elif model == 'combined':
        # Generate initial mask with U2Net
        initial_mask = u2net.predict_mask(img)
        # Load CascadePSP
        cascadepsp = clipx.load_cascadepsp()
        # Refine mask with CascadePSP
        try:
            mask = cascadepsp.refine_mask(img, initial_mask, fast=fast_mode)
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}, using U2Net mask instead")
            mask = initial_mask

    # Apply edge hardness adjustment if needed
    if edge_hardness != 50:
        mask = clipx._adjust_mask_hardness(mask, edge_hardness)

    # Save mask
    mask.save(output_path)

    processing_time = time.time() - start_time
    logger.info(f"Mask generated in {processing_time:.2f} seconds")
    return output_path


def process_image(
        image_path: str,
        output_path: Optional[str] = None,
        mode: str = "remove_bg",  # "remove_bg" or "mask"
        model: str = "combined",
        edge_hardness: int = 50,
        fast_mode: bool = False,
        device: str = "auto",
        skip_checksum: bool = False
) -> str:
    """
    Process an image with more options, either removing background or generating mask.

    Args:
        image_path: Path to the input image
        output_path: Path to save the output (if None, auto-generated)
        mode: Processing mode - "remove_bg" or "mask"
        model: Model to use ('u2net', 'cascadepsp', or 'combined')
        edge_hardness: Edge hardness level (0-100) where higher values create harder edges
        fast_mode: Whether to use fast mode for processing (less accurate but faster)
        device: Device to use ('auto', 'cpu', or 'cuda')
        skip_checksum: Whether to skip MD5 checksum verification for models

    Returns:
        str: Path to the saved output image

    Example:
        from clipx.api import process_image

        # Generate mask with specific settings
        mask_path = process_image(
            "photo.jpg",
            mode="mask",
            model="u2net",
            edge_hardness=80
        )

        # Remove background with specific settings
        result_path = process_image(
            "photo.jpg",
            mode="remove_bg",
            model="combined",
            fast_mode=True
        )
    """
    if mode == "remove_bg":
        return remove_background(
            image_path=image_path,
            output_path=output_path,
            model=model,
            edge_hardness=edge_hardness,
            fast_mode=fast_mode,
            device=device,
            skip_checksum=skip_checksum
        )
    elif mode == "mask":
        return generate_mask(
            image_path=image_path,
            output_path=output_path,
            model=model,
            edge_hardness=edge_hardness,
            fast_mode=fast_mode,
            device=device,
            skip_checksum=skip_checksum
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'remove_bg' or 'mask'.")


def batch_process(
        image_paths: List[str],
        output_dir: Optional[str] = None,
        mode: str = "remove_bg",  # "remove_bg" or "mask"
        model: str = "combined",
        edge_hardness: int = 50,
        fast_mode: bool = False,
        device: str = "auto",
        skip_checksum: bool = False
) -> List[str]:
    """
    Process multiple images in batch.

    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save outputs (if None, saved alongside inputs)
        mode: Processing mode - "remove_bg" or "mask"
        model: Model to use ('u2net', 'cascadepsp', or 'combined')
        edge_hardness: Edge hardness level (0-100) where higher values create harder edges
        fast_mode: Whether to use fast mode for processing (less accurate but faster)
        device: Device to use ('auto', 'cpu', or 'cuda')
        skip_checksum: Whether to skip MD5 checksum verification for models

    Returns:
        List[str]: Paths to the saved output images

    Example:
        from clipx.api import batch_process

        # Process all images in a directory
        import glob
        images = glob.glob("images/*.jpg")
        results = batch_process(images, output_dir="results")
    """
    # Initialize Clipx (once for all images)
    clipx = Clipx(device=device, skip_checksum=skip_checksum)

    # Preload models based on selection
    if model == 'u2net' or model == 'combined':
        clipx.load_u2net()
    if model == 'cascadepsp' or model == 'combined':
        clipx.load_cascadepsp()

    # Process each image
    result_paths = []
    total_start_time = time.time()

    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i + 1}/{len(image_paths)}: {image_path}")

        try:
            # Generate output path
            if output_dir is None:
                # If no output_dir specified, save alongside input
                output_path = None
            else:
                # Save in output_dir with same filename
                input_path = Path(image_path)
                suffix = "_remove.png" if mode == "remove_bg" else "_mask.png"
                output_filename = f"{input_path.stem}{suffix}"
                output_path = str(Path(output_dir) / output_filename)

                # Create output directory if needed
                os.makedirs(output_dir, exist_ok=True)

            # Process the image
            result_path = process_image(
                image_path=image_path,
                output_path=output_path,
                mode=mode,
                model=model,
                edge_hardness=edge_hardness,
                fast_mode=fast_mode,
                device=device,
                skip_checksum=skip_checksum
            )

            result_paths.append(result_path)

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            logger.debug(f"Detailed error: {str(e)}", exc_info=True)

    total_time = time.time() - total_start_time
    logger.info(
        f"Batch processing completed: {len(result_paths)}/{len(image_paths)} images in {total_time:.2f} seconds")

    return result_paths


def extract_foreground(
        image_path: str,
        background_color: Union[tuple, str] = (255, 255, 255, 255),
        output_path: Optional[str] = None,
        model: str = "combined",
        edge_hardness: int = 50,
        fast_mode: bool = False,
        device: str = "auto",
        skip_checksum: bool = False
) -> str:
    """
    Extract foreground from image and place it on a custom background color.

    Args:
        image_path: Path to the input image
        background_color: Background color as RGB/RGBA tuple or color name
        output_path: Path to save the output image (if None, uses input_extract.png)
        model: Model to use ('u2net', 'cascadepsp', or 'combined')
        edge_hardness: Edge hardness level (0-100) where higher values create harder edges
        fast_mode: Whether to use fast mode for processing (less accurate but faster)
        device: Device to use ('auto', 'cpu', or 'cuda')
        skip_checksum: Whether to skip MD5 checksum verification for models

    Returns:
        str: Path to the saved output image

    Example:
        from clipx.api import extract_foreground

        # With white background
        result = extract_foreground("product.jpg")

        # With custom background color
        result = extract_foreground("portrait.jpg", background_color=(0, 120, 255))

        # With named color
        result = extract_foreground("object.jpg", background_color="blue")
    """
    # Generate default output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_filename = f"{input_path.stem}_extract.png"
        output_path = str(input_path.parent / output_filename)
        logger.debug(f"Using default output path: {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize Clipx
    clipx = Clipx(device=device, skip_checksum=skip_checksum)

    # Load input image
    start_time = time.time()
    img = Image.open(image_path).convert("RGB")
    logger.debug(f"Image loaded: {img.size[0]}x{img.size[1]}")

    # Process based on selected model to get mask
    if model == 'u2net':
        u2net = clipx.load_u2net()
        mask = u2net.predict_mask(img)
    elif model == 'cascadepsp':
        cascadepsp = clipx.load_cascadepsp()
        mask = cascadepsp.predict_mask(img, fast=fast_mode)
    elif model == 'combined':
        u2net = clipx.load_u2net()
        initial_mask = u2net.predict_mask(img)
        cascadepsp = clipx.load_cascadepsp()
        try:
            mask = cascadepsp.refine_mask(img, initial_mask, fast=fast_mode)
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}, using U2Net mask instead")
            mask = initial_mask
    else:
        raise ValueError(f"Unknown model: {model}")

    # Apply edge hardness adjustment if needed
    if edge_hardness != 50:
        mask = clipx._adjust_mask_hardness(mask, edge_hardness)

    # Create background with specified color
    background = Image.new("RGBA", img.size, background_color)

    # Composite the image with mask onto the background
    img_rgba = img.convert("RGBA")
    result = Image.composite(img_rgba, background, mask)

    # Save the result
    result.save(output_path)

    processing_time = time.time() - start_time
    logger.info(f"Foreground extracted in {processing_time:.2f} seconds")
    return output_path


def get_available_devices() -> Dict[str, bool]:
    """
    Get available processing devices.

    Returns:
        Dict[str, bool]: Dictionary with device availability

    Example:
        from clipx.api import get_available_devices

        devices = get_available_devices()
        if devices['cuda']:
            print("CUDA is available for faster processing")
    """
    result = {
        "cpu": True,  # CPU is always available
        "cuda": False
    }

    # Check CUDA availability
    try:
        import torch
        result["cuda"] = torch.cuda.is_available()
        if result["cuda"]:
            logger.info("CUDA is available")
        else:
            logger.info("CUDA is not available")
    except ImportError:
        logger.warning("PyTorch not found, CUDA not available")
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")

    return result