import click


def add_input_options(command):
    """
    Add input-related options to command
    """
    command = click.option(
        '-i', '--input',
        type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
        required=True,
        help='Input image or folder.'
    )(command)

    return command