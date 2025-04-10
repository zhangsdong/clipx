"""
Core processing module for clipx.
"""

import os
import time
from PIL import Image
import logging

# Configure logger
logger = logging.getLogger("clipx.core")


class Clipx:
    """
    Main class for image processing with U2Net and CascadePSP.
    """

    def __init__(self, device='auto'):
        """
        Initialize the processing pipeline.

        Args:
            device: Device to use for processing ('auto', 'cpu' or 'cuda')
                   When 'auto', GPU will be used if available
        """
        self.device = device
        self.u2net = None
        self.cascadepsp = None
        logger.debug(f"Initializing Clipx with device preference: {device}")

    def load_u2net(self):
        """
        Load the U2Net model.
        """
        if self.u2net is None:
            from clipx.models.u2net import U2Net
            logger.debug("Loading U2Net model")
            self.u2net = U2Net().load(device=self.device)
        return self.u2net

    def load_cascadepsp(self):
        """
        Load the CascadePSP model.
        """
        if self.cascadepsp is None:
            from clipx.models.cascadepsp import CascadePSPModel
            logger.debug("Loading CascadePSP model")
            self.cascadepsp = CascadePSPModel().load(device=self.device)
        return self.cascadepsp

    def process(self, input_path, output_path, model='combined', threshold=130, fast_mode=False):
        """
        Process an image using the selected model(s).

        Args:
            input_path: Path to input image
            output_path: Path to save the output image
            model: Model to use ('u2net', 'cascadepsp', or 'combined')
            threshold: Threshold for binary mask generation (0-255)
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
                result = self._process_u2net(img, output_path, threshold)
            elif model == 'cascadepsp':
                result = self._process_cascadepsp(img, output_path, threshold, fast_mode)
            elif model == 'combined':
                result = self._process_combined(img, output_path, threshold, fast_mode)
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

    def _process_u2net(self, img, output_path, threshold):
        """
        Process with U2Net only.
        """
        # Start timing
        start_time = time.time()

        # Load model
        u2net = self.load_u2net()
        model_load_time = time.time() - start_time
        logger.debug(f"U2Net model load time: {model_load_time:.2f} seconds")

        # Generate mask and remove background
        logger.debug("Generating binary mask with U2Net")
        mask_start_time = time.time()
        binary_mask = u2net.get_binary_mask(img, threshold)
        mask_time = time.time() - mask_start_time
        logger.debug(f"U2Net mask generation time: {mask_time:.2f} seconds")

        # Save result
        save_start_time = time.time()
        if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Remove background
            logger.debug("Removing background with binary mask")
            result = Image.composite(img.convert("RGBA"),
                                     Image.new("RGBA", img.size, (0, 0, 0, 0)),
                                     binary_mask)
            result.save(output_path)
        else:
            # Save mask directly
            binary_mask.save(output_path)

        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"U2Net processing completed in {total_time:.2f} seconds")

        return output_path

    def _process_cascadepsp(self, img, output_path, threshold, fast_mode):
        """
        Process with CascadePSP only.

        Note: CascadePSP requires a binary mask as input, so we need to generate
        a mask first using a simple thresholding method.
        """
        # Start timing
        start_time = time.time()

        # Load model
        cascadepsp = self.load_cascadepsp()
        model_load_time = time.time() - start_time
        logger.debug(f"CascadePSP model load time: {model_load_time:.2f} seconds")

        # Generate a simple mask (grayscale conversion and thresholding)
        logger.debug("Generating simple mask for CascadePSP input")
        mask_start_time = time.time()
        gray = img.convert("L")
        simple_mask = gray.point(lambda p: 255 if p > threshold else 0)
        mask_time = time.time() - mask_start_time
        logger.debug(f"Simple mask generation time: {mask_time:.2f} seconds")

        # Refine mask with CascadePSP
        logger.debug(f"Refining mask with CascadePSP (fast mode: {fast_mode})")
        refine_start_time = time.time()
        try:
            refined_mask = cascadepsp.process(
                image=img,
                mask=simple_mask,
                fast=fast_mode
            )
            refine_time = time.time() - refine_start_time
            logger.debug(f"CascadePSP refinement time: {refine_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}")
            refined_mask = simple_mask
            logger.debug("Using simple mask instead")

        # Save result
        save_start_time = time.time()
        if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Remove background
            logger.debug("Removing background with refined mask")
            result = Image.composite(img.convert("RGBA"),
                                     Image.new("RGBA", img.size, (0, 0, 0, 0)),
                                     refined_mask)
            result.save(output_path)
        else:
            # Save mask directly
            refined_mask.save(output_path)

        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"CascadePSP processing completed in {total_time:.2f} seconds")

        return output_path

    def _process_combined(self, img, output_path, threshold, fast_mode):
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
        logger.debug("Generating binary mask with U2Net")
        u2net_start_time = time.time()
        binary_mask = u2net.get_binary_mask(img, threshold)
        u2net_time = time.time() - u2net_start_time
        logger.debug(f"U2Net mask generation time: {u2net_time:.2f} seconds")

        try:
            # Refine mask with CascadePSP
            logger.debug(f"Refining mask with CascadePSP (fast mode: {fast_mode})")
            cascadepsp_start_time = time.time()
            refined_mask = cascadepsp.process(
                image=img,
                mask=binary_mask,
                fast=fast_mode
            )
            cascadepsp_time = time.time() - cascadepsp_start_time
            logger.debug(f"CascadePSP refinement time: {cascadepsp_time:.2f} seconds")

            # Ensure refined_mask is a PIL Image object
            if not isinstance(refined_mask, Image.Image):
                logger.warning("CascadePSP result is not a PIL Image object, attempting conversion")
                refined_mask = Image.fromarray(refined_mask)
        except Exception as e:
            logger.error(f"CascadePSP processing failed: {e}, using U2Net mask instead")
            refined_mask = binary_mask

        # Save result
        save_start_time = time.time()
        if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Remove background
            logger.debug("Removing background with mask")
            result = Image.composite(img.convert("RGBA"),
                                     Image.new("RGBA", img.size, (0, 0, 0, 0)),
                                     refined_mask)
            result.save(output_path)
        else:
            # Save mask directly
            refined_mask.save(output_path)

        save_time = time.time() - save_start_time
        logger.debug(f"Save result time: {save_time:.2f} seconds")

        # Total processing time
        total_time = time.time() - start_time
        logger.debug(f"Combined processing completed in {total_time:.2f} seconds")

        return output_path