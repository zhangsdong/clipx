import click
import sys
import os

from . import __version__


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-v", "--version", prog_name="clipx")
@click.option("-i", "--input", required=True, help="Path to input image")
@click.option("-o", "--output", required=False,
              help="Path to output image (defaults to input filename + '_remove.png' in the same directory)")
@click.option("--mask", is_flag=True, help="Output only mask")
@click.option("--no-refine", is_flag=True, help="Skip mask refinement")
@click.option("--device", default=None, help="Device to use (e.g., 'cpu', 'cuda:0'), defaults to 'cuda:0' if available")
@click.option("--fast", is_flag=True, help="Use fast mode for refinement")
def main(input, output, mask, no_refine, device, fast):
    """
    CLI tool for removing background from images
    """
    if not os.path.isfile(input):
        sys.exit(f"Error: Input file '{input}' does not exist")

    if output is None:
        # Use current working directory instead of input directory
        input_filename = os.path.basename(input)
        base_name = os.path.splitext(input_filename)[0]
        output = os.path.join(os.getcwd(), f"{base_name}_remove.png")

    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            sys.exit(f"Error: Cannot create output directory: {e}")

    from .core import remove

    try:
        result = remove(
            input,
            only_mask=mask,
            refine_mask=not no_refine,
            device=device,
            fast=fast,
            L=900,
        )

        result.save(output)
    except Exception as e:
        sys.exit(f"Error processing image: {e}")