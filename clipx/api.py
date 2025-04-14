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
