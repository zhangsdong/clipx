"""
Command line interface for clipx.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from clipx.core import Clipx
from clipx import __version__
from clipx.logging import enable_console_logging

# CLI-specific logger
logger = logging.getLogger("clipx.cli")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Image background removal and mask generation tool')

    # Version information
    parser.add_argument('-v', '--version', action='store_true',
                      help='Show version information and exit')

    # Input and output
    parser.add_argument('-i', '--input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional, defaults to input_file_remove.png in the same directory)')

    # Model selection
    parser.add_argument('-m', '--model', choices=['u2net', 'cascadepsp', 'combined'],
                        default='combined', help='Model to use (default: combined)')

    # Advanced options
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for processing (default: auto - use GPU if available)')
    parser.add_argument('--edge-hardness', type=int, default=50,
                       help='Edge hardness (0-100, default: 50). Higher values create harder edges.')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode for CascadePSP (less accurate but faster)')
    parser.add_argument('--skip-checksum', action='store_true',
                       help='Skip MD5 checksum verification for models')

    # Logging options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress non-error output')

    return parser.parse_args()

def main():
    """
    Main entry point for CLI.
    """
    args = parse_args()

    # Setup CLI logging
    enable_console_logging()
    root_logger = logging.getLogger("clipx")

    if args.debug:
        root_logger.setLevel(logging.DEBUG)
    elif args.quiet:
        root_logger.setLevel(logging.ERROR)
    else:
        root_logger.setLevel(logging.INFO)

    # Handle version display
    if args.version:
        print(f"clipx version {__version__}")
        return 0

    # Validate required arguments when not displaying version
    if args.input is None:
        logger.error("Input path is required. Example usage: clipx -i input.jpg")
        logger.debug("For more options, use clipx --help")
        return 1

    # Check if input file exists
    if not os.path.isfile(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return 1

    # Generate default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_filename = f"{input_path.stem}_remove.png"
        args.output = str(input_path.parent / output_filename)
        logger.debug(f"Using default output path: {args.output}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Check if output directory is writable
    try:
        if output_dir:
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
    except Exception as e:
        logger.error(f"Output directory is not writable: {output_dir}")
        logger.debug(f"Write permission error: {e}")
        return 1

    try:
        # Initialize and run processing
        clipx = Clipx(device=args.device, skip_checksum=args.skip_checksum)
        result_path, processing_time = clipx.process(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            edge_hardness=args.edge_hardness,
            fast_mode=args.fast
        )

        # Output processing result and time separately
        logger.info(f"Output saved to: {result_path}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.debug(f"Detailed error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())