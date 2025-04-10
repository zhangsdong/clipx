"""
This module is adapted from CascadePSP's segmentation_refinement library
Original project: https://github.com/hkchengrex/CascadePSP
"""

import os
import logging
import numpy as np
import torch
from torchvision import transforms

from clipx.models.cascadepsp.models.psp.pspnet import RefinementModule
from clipx.models.cascadepsp.eval_helper import process_high_res_im, process_im_single_pass
from clipx.models.cascadepsp.download import download_and_or_check_model_file

# Get logger
logger = logging.getLogger("clipx.models.cascadepsp.main")

class Refiner:
    def __init__(self, device='cpu', model_folder=None, model_name='model', download_and_check_model=True):
        """
        Initialize the segmentation refinement model.
        device can be 'cpu' or 'cuda'
        model_folder specifies the folder in which the model will be downloaded and stored. Defaulted in ~/.clipx/cascadepsp.
        model_name specifies the model file name. Defaulted to 'model'.
        download_and_check_model specifies whether to verify MD5 checksum.
        """
        self.model = RefinementModule()
        self.device = device
        if model_folder is None:
            model_folder = os.path.expanduser("~/.clipx/cascadepsp")

        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, model_name)
        if download_and_check_model:
            download_and_or_check_model_file(model_path, skip_checksum=not download_and_check_model)

        logger.debug(f"Loading model from {model_path} to device {device}")
        model_dict = torch.load(model_path, map_location={'cuda:0': device})
        new_dict = {}
        for k, v in model_dict.items():
            name = k[7:]  # Remove module. from dataparallel
            new_dict[name] = v
        self.model.load_state_dict(new_dict)
        self.model.eval().to(device)
        logger.info("Model loaded successfully")

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

    def refine(self, image, mask, fast=False, L=900):
        """
        Refines an input segmentation mask of the image.

        image should be of size [H, W, 3]. Range 0~255.
        Mask should be of size [H, W] or [H, W, 1]. Range 0~255. We will make the mask binary by thresholding at 127.
        Fast mode - Use the global step only. Default: False. The speedup is more significant for high resolution images.
        L - Hyperparameter. Setting a lower value reduces memory usage. In fast mode, a lower L will make it runs faster as well.
        """
        with torch.no_grad():
            # 确保图像是三维 [H, W, 3]
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
                logger.debug("Converted grayscale image to RGB")
            elif image.shape[2] == 4:
                image = image[:, :, :3]
                logger.debug("Converted RGBA image to RGB")

            # 确保掩码是二维 [H, W]
            if len(mask.shape) == 3 and mask.shape[2] == 1:
                mask = mask[:, :, 0]
                logger.debug("Extracted mask from 3D array with single channel")
            elif len(mask.shape) > 2:
                # 如果掩码有多个通道，取第一个通道
                mask = mask[:, :, 0]
                logger.debug("Using first channel of multi-channel mask")

            # 转换为张量
            try:
                logger.debug(f"Processing with fast mode: {fast}, L: {L}")
                image_tensor = self.im_transform(image).unsqueeze(0).to(self.device)
                # 先转为二值掩码，再转为张量
                binary_mask = (mask > 127).astype(np.uint8) * 255
                mask_tensor = self.seg_transform(binary_mask).unsqueeze(0).to(self.device)

                if len(mask_tensor.shape) < 4:
                    mask_tensor = mask_tensor.unsqueeze(0)

                # 根据模式选择处理方法
                if fast:
                    logger.debug("Using fast mode (single pass)")
                    output = process_im_single_pass(self.model, image_tensor, mask_tensor, L)
                else:
                    logger.debug("Using high-res mode (multi-pass)")
                    output = process_high_res_im(self.model, image_tensor, mask_tensor, L)

                # 转回numpy数组
                result = (output[0, 0].cpu().numpy() * 255).astype('uint8')
                logger.debug("Refinement completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in refine method: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 出错时返回原始掩码
                logger.warning("Returning original mask due to processing error")
                return binary_mask if 'binary_mask' in locals() else ((mask > 127).astype(np.uint8) * 255)