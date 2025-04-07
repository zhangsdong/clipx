import click

from clipx.commands.input import add_input_options
from clipx.commands.output import add_output_options
from clipx.commands.model import add_model_options
from clipx.commands.mask import add_mask_options


def create_cli_parser():
    """
    Create CLI parser with all options
    """

    @click.command()
    @add_input_options
    @add_output_options
    @add_model_options
    @add_mask_options
    @click.option('-v', '--version', is_flag=True, help='Show version information.')
    @click.help_option('-h', '--help')
    def clipx_command(**kwargs):
        """
        clipx - Image background removal tool.
        """
        # Just collect options, actual processing happens in core.py
        return kwargs

    return clipx_command