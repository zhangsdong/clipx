import os
import torch
from pathlib import Path
import sys

from clipx.models import get_model
from clipx.commands import get_output_path
from clipx.utils import (
    load_image,
    load_mask,
    save_image,
    apply_mask_to_image,
    is_image_file
)


def process_single_image(input_path, output_path=None, model_name='auto', use_mask=None,
                         only_mask=False, fast=False):
    """
    Process a single image with the specified model

    Args:
        input_path: Path to input image
        output_path: Path to output image or None to use default
        model_name: Name of model to use
        use_mask: Path to existing mask or None
        only_mask: Whether to output only the mask
        fast: Whether to use fast mode for CascadePSP
    """
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(input_path)

    # Choose device (use CUDA if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load input image
    input_image = load_image(input_path)

    # Load mask if provided
    mask = load_mask(use_mask)

    # Get the appropriate model
    model = get_model(model_name).load(device)

    # If model is CascadePSP and no mask is provided, error
    if model_name == 'cascadepsp' and mask is None:
        print("Error: CascadePSP requires a mask. Please provide one with --use-mask.")
        sys.exit(1)

    # Process with model
    if model_name == 'cascadepsp':
        refined_mask = model.process(input_image, mask, fast=fast)

        # Save result
        if only_mask:
            save_image(refined_mask, output_path)
            print(f"Mask saved to {output_path}")
        else:
            # Apply refined mask to input image
            result_image = apply_mask_to_image(input_image, refined_mask)
            save_image(result_image, output_path)
            print(f"Image with refined mask saved to {output_path}")

    # U2Net and auto model handling will be implemented later
    else:
        print(f"Model {model_name} not fully implemented yet")
        sys.exit(1)


def process_directory(input_dir, output_dir=None, model_name='auto', use_mask=None,
                      only_mask=False, fast=False):
    """
    Process all images in a directory

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output images or None to use input_dir
        model_name: Name of model to use
        use_mask: Path to existing mask directory or None
        only_mask: Whether to output only masks
        fast: Whether to use fast mode for CascadePSP
    """
    input_dir = Path(input_dir)

    # Use input directory as output if not specified
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the directory
    image_files = [f for f in input_dir.iterdir() if f.is_file() and is_image_file(str(f))]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image
    for img_path in image_files:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        # If use_mask is a directory, look for corresponding mask
        mask_path = None
        if use_mask and os.path.isdir(use_mask):
            mask_file = Path(use_mask) / rel_path
            if mask_file.exists():
                mask_path = str(mask_file)
        elif use_mask:
            mask_path = use_mask

        process_single_image(
            str(img_path),
            str(out_path),
            model_name,
            mask_path,
            only_mask,
            fast
        )


def process_command(options):
    """
    Process command line options

    Args:
        options: Dictionary of command line options
    """
    input_path = options.get('input')
    output_path = options.get('output')
    model_name = options.get('model', 'auto')
    use_mask = options.get('use_mask')
    only_mask = options.get('only_mask', False)
    fast = options.get('fast', False)

    # Process directory or single file
    if os.path.isdir(input_path):
        process_directory(
            input_path,
            output_path,
            model_name,
            use_mask,
            only_mask,
            fast
        )
    else:
        process_single_image(
            input_path,
            output_path,
            model_name,
            use_mask,
            only_mask,
            fast
        )