from clipx.commands import create_cli_parser
from clipx.core import process_command


def main():
    """
    Main entry point for the CLI
    """
    # Create the CLI parser
    clipx_cli = create_cli_parser()

    try:
        # Parse arguments using default standalone_mode=True
        options = clipx_cli()

        # Process the command - only reached if -h/--help or -v/--version weren't used
        process_command(options)
    except SystemExit:
        # Click raises SystemExit when handling help/version with standalone_mode=True
        pass


if __name__ == '__main__':
    main()