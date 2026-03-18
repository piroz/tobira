"""tobira ab-test - Manage A/B test experiments."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``ab-test`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "ab-test",
        help="Show A/B test results from a running server",
        description="Query a running tobira server for A/B test results.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the tobira API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``ab-test`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success).
    """
    try:
        from urllib.request import urlopen
    except ImportError:
        print("error: urllib is required", file=sys.stderr)
        return 1

    url = args.url.rstrip("/") + "/api/ab-test/results"
    try:
        with urlopen(url, timeout=args.timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"error: failed to fetch A/B test results: {e}", file=sys.stderr)
        return 1

    variants = data.get("variants", {})
    if not variants:
        print("No A/B test results available.")
        return 0

    print("A/B Test Results")
    print("=" * 60)
    for name, stats in variants.items():
        print(f"\nVariant: {name}")
        print(f"  Predictions:    {stats['predictions']}")
        print(f"  Avg Latency:    {stats['avg_latency_ms']:.2f} ms")
        print(f"  Avg Score:      {stats['avg_score']:.4f}")
        print(f"  Errors:         {stats['errors']}")
        label_counts = stats.get("label_counts", {})
        if label_counts:
            labels_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(label_counts.items())
            )
            print(f"  Label Counts:   {labels_str}")
    print()
    return 0
