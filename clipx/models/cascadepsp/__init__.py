"""
This module is adapted from CascadePSP's segmentation_refinement library
Original project: https://github.com/hkchengrex/CascadePSP
"""

import os
import numpy as np
from PIL import Image

from clipx.models.base import BaseModel
from clipx.models.cascadepsp.main import Refiner


class CascadePSPModel(BaseModel):
    """
    CascadePSP model for segmentation refinement
    """

    def __init__(self):
        """
        Initialize CascadePSP model
        """
        self.model = None
        self.device = 'cpu'

    def load(self, device='cpu'):
        """
        Load CascadePSP model to specified device

        Args:
            device: Device to load model to ('cpu' or 'cuda')
        """
        self.device = device
        # Only initialize the model when needed
        if self.model is None:
            print(f"Loading CascadePSP model to {device}...")
            self.model = Refiner(device=device)
        return self

    def process(self, image, mask=None, fast=False, **kwargs):
        """
        Refine segmentation mask of the input image

        Args:
            image: Input image (PIL Image or numpy array)
            mask: Input mask image (PIL Image or numpy array)
            fast: Whether to use fast mode (default: False)

        Returns:
            Refined mask as numpy array
        """
        if self.model is None:
            self.load(self.device)

        # Convert PIL images to numpy arrays if needed
        image_np = self.prepare_image(image)
        mask_np = self.prepare_mask(mask)

        if mask_np is None:
            raise ValueError("CascadePSP requires a mask image for refinement")

        # Make sure image is RGB
        if len(image_np.shape) == 2:  # Grayscale image
            image_np = np.stack([image_np] * 3, axis=2)
        elif image_np.shape[2] == 4:  # RGBA image
            image_np = image_np[:, :, :3]

        # Process the image and mask
        refined_mask = self.model.refine(image_np, mask_np, fast=fast)

        return refined_mask