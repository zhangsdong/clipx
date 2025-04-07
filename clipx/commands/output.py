import click
import os
from pathlib import Path


def add_output_options(command):
    """
    Add output-related options to command
    """
    command = click.option(
        '-o', '--output',
        type=click.Path(file_okay=True, dir_okay=True, writable=True),
        help='Output image or folder. Default: [input_name]_remove.[ext]'
    )(command)

    return command


def get_output_path(input_path, output_path=None, suffix="_remove"):
    """
    Generate output path based on input path

    If output_path is not provided, will generate a path with suffix added to input filename
    If output_path is a directory, will use input filename with suffix in that directory

    Args:
        input_path: Path to input file
        output_path: Optional path to output file or directory
        suffix: Suffix to add to filename if generating output path

    Returns:
        Path to output file
    """
    input_path = Path(input_path)

    if output_path is None:
        # Generate output path with suffix
        stem = input_path.stem
        ext = input_path.suffix
        return str(input_path.with_name(f"{stem}{suffix}{ext}"))

    output_path = Path(output_path)

    if output_path.exists() and output_path.is_dir():
        # Output path is a directory, use input filename
        stem = input_path.stem
        ext = input_path.suffix
        return str(output_path / f"{stem}{suffix}{ext}")

    # Output path is a file, use as is
    return str(output_path)