"""tobira monitor - Analyse prediction metrics log."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any


def _format_text(report: Any) -> str:
    """Format an AnalysisReport as human-readable text."""
    lines: list[str] = []
    lines.append("=== Monitor Report ===")
    lines.append(f"Records analysed: {report.total_records}")
    if report.period_start:
        lines.append(f"Period: {report.period_start} — {report.period_end}")
    lines.append("")

    if report.psi is not None:
        lines.append(f"Score drift (PSI): {report.psi.psi}  [{report.psi.level}]")
    if report.ks is not None:
        sig = "significant" if report.ks.significant else "not significant"
        lines.append(f"KS statistic: {report.ks.statistic}  ({sig})")
    lines.append("")

    if report.fp_rates:
        lines.append("FP rate by day:")
        for pt in report.fp_rates:
            lines.append(f"  {pt.period}  {pt.rate:.4f}  (n={pt.count})")
    if report.fn_rates:
        lines.append("FN rate by day:")
        for pt in report.fn_rates:
            lines.append(f"  {pt.period}  {pt.rate:.4f}  (n={pt.count})")
    lines.append("")

    if report.threshold_suggestion is not None:
        ts = report.threshold_suggestion
        lines.append(
            f"Threshold suggestion: {ts.current} → {ts.suggested}"
            f"  (expected F1={ts.expected_f1})"
        )

    if report.backend_suggestion is not None:
        bs = report.backend_suggestion
        lines.append(
            f"Backend suggestion: {bs.suggestion}"
            f"  (labeled={bs.labeled_count})"
        )
    lines.append("")

    if report.phase_advice is not None:
        pa = report.phase_advice
        lines.append("Phase transition advice:")
        lines.append(
            f"  Current phase: {pa.current_phase.value}"
            f"  ({'ready' if pa.ready else 'not ready'})"
        )
        for cond in pa.conditions:
            mark = "✅" if cond.met else "❌"
            lines.append(f"  {mark} {cond.description}: {cond.detail}")
        lines.append(f"  → {pa.recommendation}")
        lines.append("")

    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")

    return "\n".join(lines)


def _format_json(report: Any) -> str:
    """Format an AnalysisReport as JSON."""
    return json.dumps(
        asdict(report), indent=2, ensure_ascii=False, default=str,
    )


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``monitor`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "monitor",
        help="Analyse prediction metrics log",
        description="Analyse a JSONL prediction log for drift, FP/FN rates, "
        "and threshold suggestions.",
    )
    parser.add_argument(
        "log_file",
        help="Path to the JSONL prediction log file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        metavar="DAYS",
        help="Analyse only the last N days of data",
    )
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C", "D"],
        default=None,
        help="Current deployment phase for transition advice",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``monitor`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.monitoring.analyzer import DeploymentPhase, analyze
    from tobira.monitoring.store import read_records

    try:
        records = read_records(args.log_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    current_phase = DeploymentPhase(args.phase) if args.phase else None
    report = analyze(
        records, period_days=args.period, current_phase=current_phase,
    )

    if args.output_format == "json":
        print(_format_json(report))
    else:
        print(_format_text(report))

    return 0
