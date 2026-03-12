"""tobira serve - Start the API server."""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``serve`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "serve",
        help="Start the API server",
        description="Start the tobira API server with the given configuration.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``serve`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success).
    """
    from tobira.serving.server import main as server_main

    server_main(config_path=args.config, host=args.host, port=args.port)
    return 0
