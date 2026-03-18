"""``tobira active-learning`` — Active Learning queue management.

Provides CLI access to the Active Learning labeling queue: listing
uncertain samples, assigning labels, and viewing queue statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``active-learning`` subcommand."""
    parser = subparsers.add_parser(
        "active-learning",
        help="Active Learning queue management",
        description="List uncertain samples, assign labels, and view queue statistics.",
    )
    al_sub = parser.add_subparsers(dest="al_command")

    # --- queue ---
    queue_parser = al_sub.add_parser(
        "queue", help="Show pending samples in the labeling queue",
    )
    queue_parser.add_argument(
        "--queue-path",
        default="/var/lib/tobira/active_learning_queue.jsonl",
        help="Path to the labeling queue file",
    )
    queue_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    queue_parser.set_defaults(func=_run_queue)

    # --- label ---
    label_parser = al_sub.add_parser("label", help="Assign a label to a queued sample")
    label_parser.add_argument("sample_id", help="ID of the sample to label")
    label_parser.add_argument("label", choices=["spam", "ham"], help="Label to assign")
    label_parser.add_argument(
        "--queue-path",
        default="/var/lib/tobira/active_learning_queue.jsonl",
        help="Path to the labeling queue file",
    )
    label_parser.set_defaults(func=_run_label)

    # --- stats ---
    stats_parser = al_sub.add_parser("stats", help="Show queue statistics")
    stats_parser.add_argument(
        "--queue-path",
        default="/var/lib/tobira/active_learning_queue.jsonl",
        help="Path to the labeling queue file",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    stats_parser.set_defaults(func=_run_stats)

    # --- select ---
    select_parser = al_sub.add_parser(
        "select",
        help="Select uncertain samples from prediction logs and add to queue",
    )
    select_parser.add_argument(
        "--log-path",
        default="/var/lib/tobira/predictions.jsonl",
        help="Path to the prediction log file",
    )
    select_parser.add_argument(
        "--queue-path",
        default="/var/lib/tobira/active_learning_queue.jsonl",
        help="Path to the labeling queue file",
    )
    select_parser.add_argument(
        "--strategy",
        choices=["entropy", "margin", "least_confidence"],
        default="entropy",
        help="Uncertainty sampling strategy (default: entropy)",
    )
    select_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum uncertainty threshold (default: 0.3)",
    )
    select_parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of samples to select (default: 10)",
    )
    select_parser.set_defaults(func=_run_select)

    parser.set_defaults(func=_run_help(parser))


def _run_help(parser: argparse.ArgumentParser):  # type: ignore[no-untyped-def]
    """Return a function that prints help when no subcommand is given."""

    def _func(args: argparse.Namespace) -> int:
        if not getattr(args, "al_command", None):
            parser.print_help()
            return 1
        return 0

    return _func


def _run_queue(args: argparse.Namespace) -> int:
    from tobira.evaluation.active_learning import UncertaintySampler

    sampler = UncertaintySampler(queue_path=args.queue_path)
    pending = sampler.get_pending()

    if args.output_json:
        items = [
            {
                "id": s.id,
                "text": s.text[:80] + ("..." if len(s.text) > 80 else ""),
                "score": s.score,
                "uncertainty": s.uncertainty,
                "strategy": s.strategy,
            }
            for s in pending
        ]
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0

    if not pending:
        print("No pending samples in the queue.")
        return 0

    print(f"Pending samples: {len(pending)}\n")
    for s in pending:
        text_preview = s.text[:60].replace("\n", " ")
        if len(s.text) > 60:
            text_preview += "..."
        print(f"  ID: {s.id}")
        print(f"  Text: {text_preview}")
        print(f"  Score: {s.score:.4f}  Uncertainty: {s.uncertainty:.4f}")
        print(f"  Strategy: {s.strategy}")
        print()
    return 0


def _run_label(args: argparse.Namespace) -> int:
    from tobira.evaluation.active_learning import UncertaintySampler

    sampler = UncertaintySampler(queue_path=args.queue_path)
    updated = sampler.label_sample(args.sample_id, args.label)

    if updated is None:
        print(f"Error: sample {args.sample_id} not found in queue.", file=sys.stderr)
        return 1

    print(f"Labeled sample {args.sample_id} as '{args.label}'.")
    return 0


def _run_stats(args: argparse.Namespace) -> int:
    from tobira.evaluation.active_learning import UncertaintySampler

    sampler = UncertaintySampler(queue_path=args.queue_path)
    stats = sampler.queue_stats()

    if args.output_json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    print("Queue statistics:")
    print(f"  Total:   {stats['total']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  Labeled: {stats['labeled']}")
    if stats["label_counts"]:
        print("  Labels:")
        for label, count in sorted(stats["label_counts"].items()):
            print(f"    {label}: {count}")
    return 0


def _run_select(args: argparse.Namespace) -> int:
    from tobira.evaluation.active_learning import UncertaintySampler
    from tobira.monitoring.store import read_records

    try:
        records = read_records(args.log_path)
    except FileNotFoundError:
        print(f"Error: prediction log not found: {args.log_path}", file=sys.stderr)
        return 1

    sampler = UncertaintySampler(
        strategy=args.strategy,
        uncertainty_threshold=args.threshold,
        queue_path=args.queue_path,
    )
    candidates = sampler.select_candidates(records, n=args.n)

    if not candidates:
        print("No uncertain samples found above the threshold.")
        return 0

    added = sampler.add_to_queue(candidates)
    print(f"Selected {len(candidates)} candidates, added {added} to queue.")
    return 0
