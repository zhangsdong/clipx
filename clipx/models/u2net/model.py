"""
U2Net model implementation with background removal functionality.
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import Tuple, List, Optional, Union, Dict, Any
import logging

# Get logger
logger = logging.getLogger("clipx.models.u2net.model")

class U2NetPredictor:
    """
    U2Net model for background removal.
    """
    # Singleton implementation
    _instance = None

    def __new__(cls, model_path: str, providers: Optional[List[str]] = None):
        """
        Implement singleton pattern to ensure model is loaded only once.

        Args:
            model_path: Path to the ONNX model file
            providers: List of ONNX execution providers

        Returns:
            U2NetPredictor: Singleton instance
        """
        if cls._instance is None:
            logger.info("Creating new U2NetPredictor instance (Singleton)")
            cls._instance = super(U2NetPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize the U2Net model.

        Args:
            model_path: Path to the ONNX model file
            providers: List of ONNX execution providers
        """
        # Skip initialization if already done (singleton pattern)
        if getattr(self, "_initialized", False):
            logger.debug("U2NetPredictor already initialized (Singleton)")
            return

        logger.info(f"Initializing U2Net model from {model_path}")

        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Set default providers if not specified
        if providers is None:
            providers = ['CPUExecutionProvider']

        # Initialize ONNX session
        self.session = self._load_model(model_path, providers)
        self._initialized = True

    def _load_model(self, model_path: str, providers: List[str]) -> ort.InferenceSession:
        """
        Load the ONNX model.

        Args:
            model_path: Path to the ONNX model file
            providers: List of ONNX execution providers

        Returns:
            ort.InferenceSession: ONNX runtime session
        """
        try:
            sess_options = ort.SessionOptions()
            logger.info(f"Loading model with providers: {providers}")
            return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def _normalize_image(self, img: Image.Image,
                          mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                          std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                          size: Tuple[int, int] = (320, 320)) -> Dict[str, np.ndarray]:
        """
        Normalize the input image for U2Net.

        Args:
            img: PIL Image
            mean: RGB mean values for normalization
            std: RGB standard deviation values for normalization
            size: Size to resize the image to

        Returns:
            dict: Normalized input for ONNX session
        """
        try:
            # Convert to RGB and resize
            im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize
            im_array = np.array(im) / 255.0

            # Apply mean and std normalization
            normalized = np.zeros(im_array.shape, dtype=np.float32)
            normalized[:, :, 0] = (im_array[:, :, 0] - mean[0]) / std[0]
            normalized[:, :, 1] = (im_array[:, :, 1] - mean[1]) / std[1]
            normalized[:, :, 2] = (im_array[:, :, 2] - mean[2]) / std[2]

            # Transpose to channel-first format and add batch dimension
            normalized = normalized.transpose((2, 0, 1))
            normalized = np.expand_dims(normalized, 0).astype(np.float32)

            return {self.session.get_inputs()[0].name: normalized}
        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise

    def predict_mask(self, img: Image.Image) -> Image.Image:
        """
        Generate a background mask for the input image.

        Args:
            img: PIL Image

        Returns:
            PIL.Image: Grayscale mask where white represents foreground
        """
        if not hasattr(self, "session") or self.session is None:
            raise RuntimeError("Model not initialized. Call __init__ first.")

        logger.debug("Generating mask with U2Net")

        try:
            # Get normalized input
            input_data = self._normalize_image(img)

            # Run inference
            ort_outs = self.session.run(None, input_data)

            # Get prediction
            pred = ort_outs[0][:, 0, :, :]

            # Normalize prediction to range [0, 1]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred = np.squeeze(pred)

            # Convert to PIL Image and resize to match input
            mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)

            logger.debug("U2Net mask generation completed")
            return mask
        except Exception as e:
            logger.error(f"Error processing image with U2Net: {e}")
            raise

    def remove_background(self, img: Image.Image,
                          bgcolor: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Image.Image:
        """
        Remove image background and return an image with transparent background.

        Args:
            img: PIL Image object
            bgcolor: Background color, default is transparent (0,0,0,0)

        Returns:
            PIL.Image: Transparent image with background removed
        """
        logger.debug("Removing background from image")
        mask = self.predict_mask(img)
        cutout = Image.composite(img.convert("RGBA"),
                                Image.new("RGBA", img.size, bgcolor),
                                mask)
        logger.debug("Background removal completed")
        return cutout