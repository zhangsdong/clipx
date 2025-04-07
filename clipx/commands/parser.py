import click

from clipx.commands.input import add_input_options
from clipx.commands.output import add_output_options
from clipx.commands.model import add_model_options
from clipx.commands.mask import add_mask_options


def create_cli_parser():
    """
    Create CLI parser with all options
    """

    def version_callback(ctx, param, value):
        if value:
            from clipx import __version__
            click.echo(f"clipx version {__version__}")
            ctx.exit()
        return value

    @click.command()
    @click.option('-v', '--version', is_flag=True,
                  help='Show version information.',
                  callback=version_callback,
                  is_eager=True,
                  expose_value=False)
    @add_input_options
    @add_output_options
    @add_model_options
    @add_mask_options
    @click.help_option('-h', '--help')
    def clipx_command(**kwargs):
        """
        clipx - Image background removal tool.
        """
        # 收集选项，实际处理在 core.py 中完成
        return kwargs

    return clipx_command