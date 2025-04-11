"""
Core processing module for clipx.
"""

import os
import time
from PIL import Image
import logging
from typing import Tuple, Optional, Any, Union

# Configure logger
logger = logging.getLogger("clipx.core")


class Clipx:
    """
    Main class for image processing with U2Net and CascadePSP.
    """

    def __init__(self, device: str = 'auto', skip_checksum: bool = False):
        """
        Initialize the processing pipeline.

        Args:
            device: Device to use for processing ('auto', 'cpu' or 'cuda')
                   When 'auto', GPU will be used if available
            skip_checksum: Whether to skip MD5 checksum verification for models
        """
        self.device = device
        self.skip_checksum = skip_checksum
        self.u2net = None
        self.cascadepsp = None
        logger.debug(f"Initializing Clipx with device preference: {device}, skip_checksum: {skip_checksum}")

    def load_u2net(self):
        """
        Load the U2Net model.
        """
        if self.u2net is None:
            from clipx.models.u2net import U2NetPredictor
            try:
                # Get model path from model module
                from clipx.models.u2net.download import get_model_path
                model_path = get_model_path(skip_checksum=self.skip_checksum)

                # Determine providers based on device setting
                providers = self._get_providers_for_device()

                logger.debug("Loading U2Net model")
                self.u2net = U2NetPredictor(model_path, providers=providers)
            except Exception as e:
                logger.error(f"Failed to load U2Net model: {e}")
                raise
        return self.u2net

    def load_cascadepsp(self):
        """
        Load the CascadePSP model.
        """
        if self.cascadepsp is None:
            from clipx.models.cascadepsp import CascadePSPPredictor
            try:
                # Get model path from model module
                from clipx.models.cascadepsp.download import get_model_path
                model_path = get_model_path(skip_checksum=self.skip_checksum)

                # Determine providers based on device setting
                providers = self._get_providers_for_device()

                logger.debug("Loading CascadePSP model")
                self.cascadepsp = CascadePSPPredictor(model_path, providers=providers)
            except Exception as e:
                logger.error(f"Failed to load CascadePSP model: {e}")
                raise
        return self.cascadepsp

    def _get_providers_for_device(self):
        """
        Get the appropriate ONNX providers based on device setting.
        """
        import onnxruntime as ort

        if self.device == 'auto':
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                logger.info("Automatically selecting CUDA for inference")
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                logger.info("CUDA not available, using CPU for inference")
                return ['CPUExecutionProvider']
        elif self.device == 'cuda':
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return ['CPUExecutionProvider']
        else:  # cpu
            return ['CPUExecutionProvider']

    def process(self, input_path: str, output_path: str, model: str = 'combined',
               edge_hardness: int = 50, fast_mode: bool = False) -> Tuple[str, float]:
        """
        Process an image using the selected model(s).

        Args:
            input_path: Path to input image
            output_path: Path to save the output image
            model: Model to use ('u2net', 'cascadepsp', or 'combined')
            edge_hardness: Edge hardness level (0-100) where higher values create harder edges
            fast_mode: Whether to use fast mode for CascadePSP

        Returns:
            Tuple (output_path, processing_time): Path to the output image and processing time in seconds
        """
        logger.debug(f"Processing image: {input_path} with model: {model}")

        # Start timing
        start_time = time.time()

        # Load input image
        try:
            img = Image.open(input_path).convert("RGB")
            logger.debug(f"Image loaded: {img.size[0]}x{img.size[1]}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Failed to load image: {e}")

        # Record image loading time
        load_time = time.time() - start_time
        logger.debug(f"Image loading time: {load_time:.2f} seconds")

        # Process based on selected model
        try:
            if model == 'u2net':
                result = self._process_u2net(img, output_path, edge_hardness)
            elif model == 'cascadepsp':
                result = self._process_cascadepsp(img, output_path, edge_hardness, fast_mode)
            elif model == 'combined':
                result = self._process_combined(img, output_path, edge_hardness, fast_mode)
            else:
                raise ValueError(f"Unknown model: {model}")

            # Calculate total processing time
            total_time = time.time() - start_time
            logger.debug(f"Total processing time: {total_time:.2f} seconds")

            # Return result path and processing time
            return result, total_time

        except Exception as e:
            logger.error(f"Processing error: {e}")
            # Re-raise the exception for upper layer handling
            raise

    def _adjust_mask_hardness(self, mask: Image.Image, hardness: int) -> Image.Image:
        """
        Adjust the hardness of a mask image.

        Args:
            mask: The input mask image
            hardness: Hardness level (0-100) where higher values create harder edges

        Returns:
            PIL.Image: Adjusted mask
        """
        if hardness == 50:  # Default, no adjustment needed
            return mask

        # Convert hardness to gamma value
        gamma = 1.0 + (hardness - 50) / 50.0

        # Apply gamma correction to adjust edge hardness
        return Image.eval(mask, lambda x: int(255 * ((x / 255.0) ** gamma)))

    def _save_result(self, img: Image.Image, mask: Image.Image, output_path: str) -> str:
        """
        Save processing result based on output path extension.

        Args:
            img: The input image
            mask: The mask image
            output_path: Path to save the output

        Returns:
            str: The output path
        """
        # Check if output is an image or mask format
        if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Remove background using mask
            logger.debug("Removing background with mask")
            result = Image.composite(img.convert("RGBA"),
                                     Image.new("RGBA", img.size, (0, 0, 0, 0)),
                                     mask)
            result.save(output_path)
        else:
            # Save mask directly
            mask.save(output_path)

        return output_path

    def _process_u2net(self, img: Image.Image, output_path: str, edge_hardness: int) -> str:
        """
        Process with U2Net only.
        """
        # Start timing
        start_time = time.time()

        # Load model
        u2net = self.load_u2net()
        model_load_time = time.time() - start_time
        logger.debug(f"U2Net model load time: {model_load_time:.2f} seconds")

        # Generate mask
        logger.debug("Generating mask with U2Net")
        mask_start_time = time.time()
        mask = u2net.predict_mask(img)

        # Apply edge hardness adjustment if needed
        if edge_hardness != 50:
            mask = self._adjust_mask_hardness(mask, edge_hardness)

        mask_time = time.time() - mask_start_time
        logger.debug(f"U2Net mask generation time: {mask_time:.2f} seconds")

        # Save the result
        save_start_time = time.time()
        self._save_result(img, mask, output_path)
        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"U2Net processing completed in {total_time:.2f} seconds")

        return output_path

    def _process_cascadepsp(self, img: Image.Image, output_path: str, edge_hardness: int, fast_mode: bool) -> str:
        """
        Process with CascadePSP only.
        """
        # Start timing
        start_time = time.time()

        # Load model
        cascadepsp = self.load_cascadepsp()
        model_load_time = time.time() - start_time
        logger.debug(f"CascadePSP model load time: {model_load_time:.2f} seconds")

        # Generate mask
        logger.debug(f"Generating mask with CascadePSP (fast mode: {fast_mode})")
        mask_start_time = time.time()

        try:
            mask = cascadepsp.predict_mask(img, fast=fast_mode)

            # Apply edge hardness adjustment if needed
            if edge_hardness != 50:
                mask = self._adjust_mask_hardness(mask, edge_hardness)

            mask_time = time.time() - mask_start_time
            logger.debug(f"CascadePSP mask generation time: {mask_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}")
            # Generate a simple fallback mask using grayscale
            logger.warning("Falling back to simple grayscale mask")
            mask = img.convert("L")
            if edge_hardness != 50:
                mask = self._adjust_mask_hardness(mask, edge_hardness)

        # Save the result
        save_start_time = time.time()
        self._save_result(img, mask, output_path)
        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"CascadePSP processing completed in {total_time:.2f} seconds")

        return output_path

    def _process_combined(self, img: Image.Image, output_path: str, edge_hardness: int, fast_mode: bool) -> str:
        """
        Process with combined U2Net and CascadePSP.
        """
        # Start timing
        start_time = time.time()

        # Load models
        u2net = self.load_u2net()
        cascadepsp = self.load_cascadepsp()
        model_load_time = time.time() - start_time
        logger.debug(f"Combined models load time: {model_load_time:.2f} seconds")

        # Generate mask with U2Net
        logger.debug("Generating initial mask with U2Net")
        u2net_start_time = time.time()
        initial_mask = u2net.predict_mask(img)
        u2net_time = time.time() - u2net_start_time
        logger.debug(f"U2Net mask generation time: {u2net_time:.2f} seconds")

        # Refine mask with CascadePSP
        mask = initial_mask
        try:
            logger.debug(f"Refining mask with CascadePSP (fast mode: {fast_mode})")
            cascadepsp_start_time = time.time()
            mask = cascadepsp.refine_mask(img, initial_mask, fast=fast_mode)
            cascadepsp_time = time.time() - cascadepsp_start_time
            logger.debug(f"CascadePSP refinement time: {cascadepsp_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}, using U2Net mask instead")

        # Apply edge hardness adjustment if needed
        if edge_hardness != 50:
            mask = self._adjust_mask_hardness(mask, edge_hardness)

        # Save result
        save_start_time = time.time()
        self._save_result(img, mask, output_path)
        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"Combined processing completed in {total_time:.2f} seconds")

        return output_path