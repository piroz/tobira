"""tobira.cli - Command-line interface for tobira."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Configured ArgumentParser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="tobira",
        description="tobira - gateway toolkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Register subcommands
    from tobira.cli.serve import register as register_serve

    register_serve(subparsers)

    from tobira.cli.doctor import register as register_doctor

    register_doctor(subparsers)

    from tobira.cli.setup import register as register_init

    register_init(subparsers)

    from tobira.cli.monitor_cmd import register as register_monitor

    register_monitor(subparsers)

    from tobira.cli.evaluate import register as register_evaluate

    register_evaluate(subparsers)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the tobira CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)  # type: ignore[no-any-return]
