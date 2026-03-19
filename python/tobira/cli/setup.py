"""tobira init - MTA auto-detection and configuration file generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tobira.cli.generators import generate_configs, get_install_instructions
from tobira.cli.mta_detectors import detect_mtas
from tobira.errors import CONFIG_INVALID_SYNTAX, format_cli_error

# Supported MTA names for manual selection.
_SUPPORTED_MTAS = ("rspamd", "spamassassin", "haraka")


def _prompt_select_mta(names: list[str]) -> str | None:
    """Prompt user to select an MTA from a list.

    Args:
        names: List of MTA names to choose from.

    Returns:
        Selected MTA name, or None if cancelled.
    """
    print("\nDetected MTA services:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = input(f"\nSelect MTA [1-{len(names)}] (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice.lower() == "q":
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
        except ValueError:
            pass

        print(f"Invalid choice. Enter 1-{len(names)} or q to quit.")


def _prompt_api_url() -> str:
    """Prompt user for the tobira API server URL.

    Returns:
        API server URL string.
    """
    default = "http://127.0.0.1:8000"
    try:
        url = input(f"\ntobira API server URL [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default

    return url if url else default


def run_init(
    output_dir: Path | None = None,
    api_url: str | None = None,
) -> int:
    """Run the init wizard.

    Detects installed MTA services, prompts the user to select one,
    and generates configuration files in the output directory.

    Args:
        output_dir: Output directory for generated files. Defaults to cwd.
        api_url: API server URL. If None, prompts the user.

    Returns:
        Exit code (0 for success, 1 for failure/cancellation).
    """
    if output_dir is None:
        output_dir = Path.cwd()

    # Step 1: Detect MTAs.
    print("Detecting MTA services...")
    detected = detect_mtas()

    if detected:
        names = [d.name for d in detected]
        for d in detected:
            print(f"  Found: {d.name} (via {d.source})")

        if len(names) == 1:
            mta = names[0]
            print(f"\nUsing detected MTA: {mta}")
        else:
            selected = _prompt_select_mta(names)
            if selected is None:
                print("Cancelled.")
                return 1
            mta = selected
    else:
        print("No MTA services detected.")
        print("\nSupported MTAs:")
        for i, name in enumerate(_SUPPORTED_MTAS, 1):
            print(f"  {i}. {name}")

        selected = _prompt_select_mta(list(_SUPPORTED_MTAS))
        if selected is None:
            print("Cancelled.")
            return 1
        mta = selected

    # Step 2: Get API URL.
    if api_url is None:
        api_url = _prompt_api_url()

    # Step 3: Generate config files.
    print(f"\nGenerating {mta} configuration files...")
    try:
        generated = generate_configs(mta=mta, output_dir=output_dir, api_url=api_url)
    except (ValueError, FileNotFoundError) as exc:
        print(
            format_cli_error(CONFIG_INVALID_SYNTAX, str(exc)),
            file=sys.stderr,
        )
        return 1

    for path in generated:
        print(f"  Created: {path}")

    # Step 4: Show install instructions.
    instructions = get_install_instructions(mta=mta, output_dir=output_dir)
    if instructions:
        print("\nTo install, copy the files to the MTA config directory:")
        for line in instructions:
            print(line)

    print("\nDone.")
    return 0


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``init`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "init",
        help="Detect MTA and generate plugin configuration",
        description=(
            "Auto-detect MTA services and generate "
            "tobira plugin configuration files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="tobira API server URL (default: prompted interactively)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``init`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    output_dir = Path(args.output_dir) if args.output_dir else None
    return run_init(output_dir=output_dir, api_url=args.api_url)
