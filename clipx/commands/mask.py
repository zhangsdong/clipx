import click


def add_mask_options(command):
    """
    Add mask-related options to command
    """
    command = click.option(
        '-k', '--only-mask',
        is_flag=True,
        help='Output only mask image.'
    )(command)

    command = click.option(
        '-u', '--use-mask',
        type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
        help='Use an existing mask image.'
    )(command)

    return command