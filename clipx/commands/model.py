import click
from clipx.models import get_model


def add_model_options(command):
    """
    Add model selection options to command
    """
    command = click.option(
        '-m', '--model',
        type=click.Choice(['u2net', 'cascadepsp', 'auto'], case_sensitive=False),
        default='auto',
        help='Model to use: u2net, cascadepsp, auto (default).'
    )(command)

    command = click.option(
        '--fast',
        is_flag=True,
        help='Use fast mode for CascadePSP (less accurate but faster).'
    )(command)

    return command