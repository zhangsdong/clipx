"""
U2Net model implementation with background removal functionality.
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import Tuple, List, Optional
import logging

# 获取logger
logger = logging.getLogger("clipx.models.u2net.model")

class U2NetPredictor:
    # 单例实例
    _instance = None

    def __new__(cls, model_path: str, providers: List[str] = None):
        """实现单例模式，确保模型只被加载一次"""
        if cls._instance is None:
            logger.info("Creating new U2NetPredictor instance (Singleton)")
            cls._instance = super(U2NetPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, providers: List[str] = None):
        """初始化模型会话"""
        # 如果已经初始化，则直接返回
        if getattr(self, "_initialized", False):
            logger.info("U2NetPredictor already initialized (Singleton)")
            return
        logger.info(f"Initializing U2Net model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.session = self._load_model(model_path, providers)
        self._initialized = True

    def _load_model(self, model_path: str, providers: List[str] = None):
        """加载 ONNX 模型"""
        if providers is None:
            providers = ['CPUExecutionProvider']
        sess_opts = ort.SessionOptions()
        logger.info(f"Loading model with providers: {providers}")
        return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    def _normalize_image(self, img: Image.Image, mean: Tuple[float, float, float], std: Tuple[float, float, float],
                         size: Tuple[int, int]) -> dict:
        """将输入图像进行标准化和调整大小"""
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        im_array = np.array(im) / 255.0
        im_array = (im_array - mean) / std
        im_array = im_array.transpose((2, 0, 1)).astype(np.float32)
        return {self.session.get_inputs()[0].name: np.expand_dims(im_array, 0)}

    def predict_mask(self, img: Image.Image) -> Image.Image:
        """生成背景掩码"""
        input_tensor = self._normalize_image(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320))
        pred = self.session.run(None, input_tensor)[0][0, 0]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        return mask.resize(img.size, Image.Resampling.LANCZOS)

    def remove_background(self, img: Image.Image,
                          bgcolor: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0)) -> Image.Image:
        """去除背景，返回带透明背景的图像"""
        logger.info("Starting background removal process")
        mask = self.predict_mask(img)
        cutout = Image.composite(img.convert("RGBA"), Image.new("RGBA", img.size, bgcolor), mask)
        logger.info("Background removal completed")
        return cutout