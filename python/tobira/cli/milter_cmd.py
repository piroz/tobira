"""tobira milter - Start the milter daemon for Postfix integration."""

from __future__ import annotations

import argparse
import logging
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``milter`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "milter",
        help="Start the milter daemon for Postfix integration",
        description=(
            "Start the tobira milter daemon that integrates Postfix "
            "directly with the tobira API server for spam classification."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to milter configuration file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``milter`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from tobira.milter.config import load_config
    except ImportError:
        print(
            "Error: pymilter is required for the milter command. "
            "Install it with: pip install tobira[milter]"
        )
        return 1

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    from tobira.milter.filter import run_milter

    try:
        run_milter(
            socket_spec=config.socket,
            api_url=config.api_url,
            timeout=config.timeout,
            fail_action=config.fail_action,
            reject_threshold=config.reject_threshold,
            add_headers=config.add_headers,
        )
    except ImportError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down milter...")

    return 0
