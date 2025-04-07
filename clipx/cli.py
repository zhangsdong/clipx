from clipx.commands import create_cli_parser
from clipx.core import process_command


def main():
    """
    Main entry point for the CLI
    """
    # Create the CLI parser
    clipx_cli = create_cli_parser()

    # Parse arguments and process the command
    options = clipx_cli(standalone_mode=False)

    # Process the command
    process_command(options)


if __name__ == '__main__':
    main()