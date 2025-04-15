import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import Tuple, List, Optional

from .download import download_and_or_check_model_file


class U2NetPredictor:
    def __init__(self, model_path: str = None, providers: List[str] = None,
                 download_and_check_model: bool = True):
        """Initialize model session"""
        # Default model path if not specified
        if model_path is None:
            model_folder = os.path.expanduser("~/.clipx/u2net")
            if not os.path.exists(model_folder):
                os.makedirs(model_folder, exist_ok=True)
            model_path = os.path.join(model_folder, 'u2net.onnx')

        # Download model if needed
        if download_and_check_model:
            download_and_or_check_model_file(model_path)

        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Set default providers if not specified
        if providers is None:
            providers = ['CPUExecutionProvider']

        # Load model
        self.session = self._load_model(model_path, providers)

    def _load_model(self, model_path: str, providers: List[str]):
        """Load ONNX model"""
        sess_opts = ort.SessionOptions()
        return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    def _normalize_image(self, img: Image.Image, mean: Tuple[float, float, float], std: Tuple[float, float, float],
                         size: Tuple[int, int]) -> dict:
        """Normalize and resize input image"""
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        im_array = np.array(im) / 255.0
        im_array = (im_array - mean) / std
        im_array = im_array.transpose((2, 0, 1)).astype(np.float32)
        return {self.session.get_inputs()[0].name: np.expand_dims(im_array, 0)}

    def predict_mask(self, img: Image.Image) -> Image.Image:
        """Generate background mask"""
        input_tensor = self._normalize_image(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320))
        pred = self.session.run(None, input_tensor)[0][0, 0]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        return mask.resize(img.size, Image.Resampling.LANCZOS)

    def remove_background(self, img: Image.Image,
                          bgcolor: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0)) -> Image.Image:
        """Remove background and return image with transparent background"""
        mask = self.predict_mask(img)
        cutout = Image.composite(img.convert("RGBA"), Image.new("RGBA", img.size, bgcolor), mask)
        return cutout