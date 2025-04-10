"""
CascadePSP model implementation for segmentation refinement.
Original project: https://github.com/hkchengrex/CascadePSP
"""

import os
import logging
import numpy as np
from PIL import Image

from clipx.models.base import BaseModel
from clipx.models.cascadepsp.main import Refiner

# Configure logging
logger = logging.getLogger(__name__)

class CascadePSPModel(BaseModel):
    """
    CascadePSP model for segmentation refinement
    """

    def __init__(self):
        """
        Initialize CascadePSP model
        """
        super().__init__()
        self.name = "cascadepsp"
        self.model = None
        self.device = 'cpu'

    def load(self, device='auto'):
        """
        Load CascadePSP model to specified device

        Args:
            device: Device to load model to ('auto', 'cpu' or 'cuda')
                   When 'auto', GPU will be used if available
        """
        # Determine the best device to use
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Automatically selected device: {device}")
            except ImportError:
                device = 'cpu'
                logger.warning("PyTorch not available, falling back to CPU")
        elif device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = 'cpu'
                else:
                    logger.info("Using CUDA for inference")
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                device = 'cpu'

        self.device = device
        # Only initialize the model when needed
        if self.model is None:
            logger.info(f"Loading CascadePSP model to {device}...")
            self.model = Refiner(device=device)
        return self

    def process(self, image, mask=None, fast=False):
        """
        Refine segmentation mask of the input image

        Args:
            image: Input image (PIL Image or numpy array)
            mask: Input mask image (PIL Image or numpy array)
            fast: Whether to use fast mode (default: False)

        Returns:
            Refined mask as PIL Image
        """
        if self.model is None:
            self.load(self.device)

        # Check if mask is provided
        if mask is None:
            raise ValueError("CascadePSP requires a mask image for refinement")

        # Convert PIL images to numpy arrays
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask

        # Make sure image is RGB
        if len(image_np.shape) == 2:  # Grayscale image
            image_np = np.stack([image_np] * 3, axis=2)
        elif image_np.shape[2] == 4:  # RGBA image
            image_np = image_np[:, :, :3]

        # Process the image and mask
        try:
            logger.info(f"Processing mask with shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            refined_mask_np = self.model.refine(image_np, mask_np, fast=fast)

            # Convert back to PIL Image
            refined_mask = Image.fromarray(refined_mask_np)

            return refined_mask
        except Exception as e:
            logger.error(f"Error in CascadePSP refinement: {e}", exc_info=True)
            # Fallback to original mask if refinement fails
            logger.info("Falling back to original mask")
            if isinstance(mask, Image.Image):
                return mask
            else:
                return Image.fromarray(mask_np)