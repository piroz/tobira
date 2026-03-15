"""tobira monitor - Analyse prediction metrics log."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        "--watch",
        action="store_true",
        default=False,
        help="Run in daemon mode with periodic analysis",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Analysis interval in seconds for --watch mode (default: 300)",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        dest="daemon_log",
        help="Log file path for --watch mode output",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        metavar="PATH",
        help="PID file path to prevent multiple instances",
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
    if args.watch:
        return _run_watch(args)
    return _run_once(args)


def _run_once(args: argparse.Namespace) -> int:
    """Run a single analysis pass.

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


def _acquire_pid_file(pid_path: str) -> bool:
    """Write the current PID to the given file.

    Returns ``False`` if another process already holds the PID file and
    is still running.

    Args:
        pid_path: Path to the PID file.

    Returns:
        ``True`` if the lock was acquired.
    """
    import os

    p = Path(pid_path)
    if p.exists():
        try:
            existing_pid = int(p.read_text().strip())
            os.kill(existing_pid, 0)
            return False
        except (ValueError, OSError):
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))
    return True


def _release_pid_file(pid_path: str) -> None:
    """Remove the PID file.

    Args:
        pid_path: Path to the PID file.
    """
    p = Path(pid_path)
    if p.exists():
        p.unlink(missing_ok=True)


def _setup_daemon_logging(log_path: str | None) -> None:
    """Configure logging for daemon mode.

    When *log_path* is given the root logger writes to that file.
    Otherwise output goes to stderr.

    Args:
        log_path: Optional path for the log file.
    """
    handler: logging.Handler
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(p), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    root = logging.getLogger("tobira")
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Set the shutdown flag on SIGINT / SIGTERM."""
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.info("Received signal %s, shutting down…", signum)


def _run_watch(args: argparse.Namespace) -> int:
    """Run the monitor in daemon (watch) mode.

    Periodically reads the prediction log and runs analysis until
    interrupted by SIGINT or SIGTERM.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False

    from tobira.monitoring.analyzer import DeploymentPhase, analyze
    from tobira.monitoring.store import read_records

    _setup_daemon_logging(args.daemon_log)

    if args.pid_file is not None:
        if not _acquire_pid_file(args.pid_file):
            logger.error(
                "Another monitor instance is already running (PID file: %s)",
                args.pid_file,
            )
            return 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    interval: int = args.interval
    logger.info(
        "Starting watch mode (interval=%ds, log_file=%s)",
        interval,
        args.log_file,
    )

    try:
        while not _shutdown_requested:
            try:
                records = read_records(args.log_file)
            except FileNotFoundError:
                logger.warning("Log file not found: %s", args.log_file)
                _sleep_interruptible(interval)
                continue

            current_phase = DeploymentPhase(args.phase) if args.phase else None
            report = analyze(
                records, period_days=args.period, current_phase=current_phase,
            )

            if args.output_format == "json":
                output = _format_json(report)
            else:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                output = f"[{timestamp}]\n{_format_text(report)}"

            logger.info("Analysis complete: %d records", report.total_records)

            if report.warnings:
                for w in report.warnings:
                    logger.warning("DRIFT: %s", w)

            print(output, flush=True)

            _sleep_interruptible(interval)
    finally:
        if args.pid_file is not None:
            _release_pid_file(args.pid_file)
        logger.info("Monitor watch stopped.")

    return 0


def _sleep_interruptible(seconds: int) -> None:
    """Sleep for *seconds*, checking the shutdown flag every second.

    Args:
        seconds: Total number of seconds to sleep.
    """
    for _ in range(seconds):
        if _shutdown_requested:
            break
        time.sleep(1)
