from abc import ABC, abstractmethod
import os
import numpy as np
from PIL import Image


class BaseModel(ABC):
    """
    Base class for all models in clipx
    """

    @abstractmethod
    def load(self, device='cpu'):
        """
        Load model to specified device
        """
        pass

    @abstractmethod
    def process(self, image, mask=None, **kwargs):
        """
        Process the image with the model
        Args:
            image: Input image (PIL Image or numpy array)
            mask: Optional mask image (PIL Image or numpy array)
            **kwargs: Additional model-specific parameters
        Returns:
            Processed image or mask
        """
        pass

    def prepare_image(self, image):
        """
        Convert image to numpy array if it's a PIL Image
        """
        if isinstance(image, Image.Image):
            return np.array(image)
        return image

    def prepare_mask(self, mask):
        """
        Convert mask to numpy array if it's a PIL Image
        """
        if mask is None:
            return None
        if isinstance(mask, Image.Image):
            return np.array(mask)
        return mask