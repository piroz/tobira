"""``tobira migrate-store`` — migrate JSONL data to an external store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    """Register the ``migrate-store`` subcommand."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "migrate-store",
        help="Migrate JSONL data to an external store backend",
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to the source JSONL file",
    )
    parser.add_argument(
        "collection",
        help="Target collection name (e.g. 'predictions', 'feedback')",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to tobira.toml config file with [store] section",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count records without writing to the target store",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the migrate-store command."""
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        return 1

    from tobira.config import load_toml
    from tobira.monitoring.store import create_store

    try:
        config = load_toml(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    store_config = config.get("store")
    if not store_config:
        print(
            "Error: [store] section not found in config file",
            file=sys.stderr,
        )
        return 1

    # Read source records
    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed line {lineno}: {exc}",
                    file=sys.stderr,
                )

    print(f"Source: {jsonl_path} ({len(records)} records)")
    print(f"Target: {store_config.get('type', 'jsonl')} / {args.collection}")

    if args.dry_run:
        print("Dry run — no records written.")
        return 0

    store = create_store(store_config)
    migrated = 0
    for record in records:
        store.append(args.collection, record)
        migrated += 1

    print(f"Migrated {migrated} records to '{args.collection}'.")
    return 0
