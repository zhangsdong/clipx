"""
CascadePSP model implementation with segmentation refinement functionality.
"""

import numpy as np
from PIL import Image
import logging

from clipx.models.base import BaseModel
from clipx.models.cascadepsp.main import Refiner
from clipx.models.cascadepsp.download import download_cascadepsp_model

# Get logger
logger = logging.getLogger("clipx.models.cascadepsp.model")

class CascadePSPModel(BaseModel):
    """
    CascadePSP model for segmentation refinement.
    """

    def __init__(self):
        """
        Initialize the CascadePSP model.
        """
        super().__init__()
        self.name = "cascadepsp"
        self.refiner = None

    def load(self, device='auto'):
        """
        Load the CascadePSP model.

        Args:
            device: Device to load the model on ('auto', 'cpu' or 'cuda')
                   When 'auto', GPU will be used if available

        Returns:
            self: The model instance
        """
        # Determine the best device to use
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.debug(f"Automatically selected device: {device}")

        logger.debug(f"Loading CascadePSP model to {device}...")

        # Download model if not exists
        download_cascadepsp_model()

        # Create Refiner
        self.refiner = Refiner(device=device)

        return self

    def process(self, image, mask=None, fast=False):
        """
        Process the input image and mask.

        Args:
            image: PIL Image to process
            mask: Optional mask to refine (PIL Image)
            fast: Whether to use fast mode

        Returns:
            PIL Image: The refined mask
        """
        if self.refiner is None:
            raise Exception("Model not loaded. Call load() first.")

        # If no mask is provided, create a simple grayscale-based one
        if mask is None:
            mask = image.convert("L")

        # Convert to numpy
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Process shape info
        logger.debug(f"Processing mask with shape: {mask_np.shape}, dtype: {mask_np.dtype}")

        # Refine mask
        result_np = self.refiner.refine(image_np, mask_np, fast=fast)

        # Convert back to PIL
        result = Image.fromarray(result_np)
        return result