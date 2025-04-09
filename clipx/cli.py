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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser.add_argument('-o', '--output', help='Output image path')

    # Model selection
    parser.add_argument('-m', '--model', choices=['u2net', 'cascadepsp', 'combined'],
                        default='combined', help='Model to use (default: combined)')

    # Advanced options
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for processing (default: auto - use GPU if available)')
    parser.add_argument('--threshold', type=int, default=130,
                       help='Threshold for binary mask generation (0-255, default: 130)')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode for CascadePSP (less accurate but faster)')

    return parser.parse_args()

def main():
    """
    Main entry point for CLI.
    """
    args = parse_args()

    # Handle version display
    if args.version:
        print(f"clipx version {__version__}")
        return 0

    # Validate required arguments when not displaying version
    if args.input is None or args.output is None:
        logger.error("Input and output paths are required unless using --version")
        print("Error: Input and output paths are required. Use -i INPUT and -o OUTPUT.")
        print("For help, use --help")
        return 1

    # Check if input file exists
    if not os.path.isfile(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize and run processing
        clipx = Clipx(device=args.device)
        result_path = clipx.process(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            threshold=args.threshold,
            fast_mode=args.fast
        )

        logger.info(f"Processing completed successfully: {result_path}")
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())