"""
CascadePSP model implementation with segmentation refinement functionality.
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import logging
from typing import Tuple, List, Optional, Union, Dict, Any

# Get logger
logger = logging.getLogger("clipx.models.cascadepsp.model")

class CascadePSPPredictor:
    """
    CascadePSP model for segmentation refinement.
    """
    # Singleton implementation
    _instance = None

    def __new__(cls, model_path: str, providers: Optional[List[str]] = None):
        """
        Implement singleton pattern to ensure model is loaded only once

        Args:
            model_path: Path to the model file
            providers: Not used, only for interface consistency

        Returns:
            CascadePSPPredictor: Singleton instance
        """
        if cls._instance is None:
            logger.info("Creating new CascadePSPPredictor instance (Singleton)")
            cls._instance = super(CascadePSPPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize the CascadePSP model

        Args:
            model_path: Path to the model file
            providers: Not used, only for interface consistency
        """
        # Skip initialization if already done (singleton pattern)
        if getattr(self, "_initialized", False):
            logger.debug("CascadePSPPredictor already initialized (Singleton)")
            return

        logger.info(f"Initializing CascadePSP model from {model_path}")

        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Determine device (GPU or CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for CascadePSP")

        try:
            # Load model
            from clipx.models.cascadepsp.main import Refiner
            self.refiner = Refiner(
                device=device,
                model_folder=os.path.dirname(model_path),
                model_name=os.path.basename(model_path),
                download_and_check_model=False  # We already handle downloading and checking
            )

            # Set up transformations
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                ),
            ])

            self._initialized = True
            logger.info("CascadePSP model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CascadePSP model: {e}")
            raise

    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess the input image

        Args:
            image: PIL image or numpy array

        Returns:
            np.ndarray: Preprocessed numpy array
        """
        # If PIL image, convert to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Ensure the image is in RGB format
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
            logger.debug("Converted grayscale image to RGB")
        elif image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
            logger.debug("Converted RGBA image to RGB")

        return image_np

    def _preprocess_mask(self, mask: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess the input mask

        Args:
            mask: PIL mask or numpy array

        Returns:
            np.ndarray: Preprocessed binary mask
        """
        # If PIL image, convert to numpy array
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask

        # Ensure mask is 2D
        if len(mask_np.shape) == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np[:, :, 0]
        elif len(mask_np.shape) > 2:
            mask_np = mask_np[:, :, 0]

        # Binarize the mask
        binary_mask = (mask_np > 127).astype(np.uint8) * 255
        return binary_mask

    def predict_mask(self, img: Image.Image, initial_mask: Optional[Image.Image] = None,
                    fast: bool = False) -> Image.Image:
        """
        Generate or refine a segmentation mask

        Args:
            img: Input image
            initial_mask: Optional initial mask, created automatically if not provided
            fast: Whether to use fast mode

        Returns:
            PIL.Image: Processed mask
        """
        if not hasattr(self, "refiner") or self.refiner is None:
            raise RuntimeError("Model not initialized properly")

        # If no initial mask is provided, create a simple grayscale one
        if initial_mask is None:
            logger.debug("No initial mask provided, creating simple grayscale mask")
            initial_mask = img.convert("L")

        try:
            # Preprocess image and mask
            image_np = self._preprocess_image(img)
            mask_np = self._preprocess_mask(initial_mask)

            # Process with refiner
            logger.debug(f"Processing with fast mode: {fast}")
            result_np = self.refiner.refine(image_np, mask_np, fast=fast)

            # Convert back to PIL image
            result = Image.fromarray(result_np)
            logger.debug("Mask prediction/refinement completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in mask prediction: {e}")
            # Return original mask or create a new one in case of error
            logger.warning("Returning fallback mask due to processing error")
            if initial_mask is not None:
                return initial_mask
            else:
                return img.convert("L")

    def refine_mask(self, img: Image.Image, mask: Image.Image, fast: bool = False) -> Image.Image:
        """
        Refine an existing mask

        Args:
            img: Input image
            mask: Mask to be refined
            fast: Whether to use fast mode

        Returns:
            PIL.Image: Refined mask
        """
        return self.predict_mask(img, initial_mask=mask, fast=fast)

    def remove_background(self, img: Image.Image, mask: Optional[Image.Image] = None,
                          bgcolor: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          fast: bool = False) -> Image.Image:
        """
        Remove image background

        Args:
            img: Input image
            mask: Optional mask, created automatically if not provided
            bgcolor: Background color, transparent by default
            fast: Whether to use fast mode

        Returns:
            PIL.Image: Background-removed image
        """
        logger.debug("Removing background with CascadePSP")
        refined_mask = self.predict_mask(img, initial_mask=mask, fast=fast)
        cutout = Image.composite(img.convert("RGBA"),
                                Image.new("RGBA", img.size, bgcolor),
                                refined_mask)
        logger.debug("Background removal completed")
        return cutout