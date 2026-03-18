"""tobira demo - Run a demo environment with Docker Compose."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def _find_docker_compose_dir() -> Path | None:
    """Find the docker/ directory relative to common installation paths.

    Searches upward from the package directory and checks common locations.

    Returns:
        Path to the docker directory, or None if not found.
    """
    # Try relative to the tobira package (source tree)
    pkg_dir = Path(__file__).resolve().parent.parent  # tobira/
    candidates = [
        pkg_dir.parent.parent / "docker",  # python/../docker
    ]
    for candidate in candidates:
        if (candidate / "docker-compose.yml").exists():
            return candidate
    return None


def _check_prerequisites() -> tuple[bool, str]:
    """Check that Docker and Docker Compose are available.

    Returns:
        Tuple of (ok, message).
    """
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker command not found. Please install Docker."

    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "Docker and Docker Compose are available."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return False, "docker compose not available. Please install Docker Compose V2."


def _wait_for_health(
    url: str,
    timeout: int = 60,
    interval: float = 2.0,
) -> bool:
    """Wait for the health endpoint to return HTTP 200.

    Args:
        url: Health check URL.
        timeout: Maximum wait time in seconds.
        interval: Polling interval in seconds.

    Returns:
        True if healthy within timeout, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (URLError, OSError, TimeoutError):
            pass
        time.sleep(interval)
    return False


def _compose_up(docker_dir: Path) -> int:
    """Start the demo environment with docker compose.

    Args:
        docker_dir: Path to the docker directory.

    Returns:
        Exit code from docker compose.
    """
    result = subprocess.run(
        ["docker", "compose", "up", "--build", "-d"],
        cwd=str(docker_dir),
    )
    return result.returncode


def _compose_down(docker_dir: Path) -> int:
    """Stop and clean up the demo environment.

    Args:
        docker_dir: Path to the docker directory.

    Returns:
        Exit code from docker compose.
    """
    result = subprocess.run(
        ["docker", "compose", "down", "--volumes", "--remove-orphans"],
        cwd=str(docker_dir),
    )
    return result.returncode


def _print_usage_examples() -> None:
    """Print example commands for interacting with the demo."""
    print(
        """
Demo environment is ready!

Try the following commands:

  # Health check
  curl http://localhost:8000/health

  # Classify a spam message
  curl -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"text": "Limited time offer! Click here to claim your prize now!"}'

  # Classify a ham message
  curl -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"text": "Hi, the meeting has been moved to 3pm tomorrow."}'

  # Send a test email via Haraka (port 2525)
  # swaks --to test@example.com --from sender@example.com \\
  #   --server localhost:2525 --body "Buy now! Special offer!"

To stop the demo:
  tobira demo --down
"""
    )


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``demo`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "demo",
        help="Run a demo environment with Docker Compose",
        description=(
            "Start or stop the tobira demo environment. "
            "Launches API server, Haraka, rspamd, and SpamAssassin "
            "via Docker Compose."
        ),
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="Stop and clean up the demo environment",
    )
    parser.add_argument(
        "--docker-dir",
        default=None,
        help="Path to the docker/ directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for health check (default: 60)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``demo`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Resolve docker directory
    docker_dir: Path | None
    if args.docker_dir:
        docker_dir = Path(args.docker_dir)
    else:
        docker_dir = _find_docker_compose_dir()

    if docker_dir is None or not (docker_dir / "docker-compose.yml").exists():
        print(
            "Could not find docker/docker-compose.yml. "
            "Use --docker-dir to specify the path."
        )
        return 1

    # Check prerequisites
    ok, msg = _check_prerequisites()
    if not ok:
        print(msg)
        return 1

    # Handle --down
    if args.down:
        print("Stopping demo environment...")
        rc = _compose_down(docker_dir)
        if rc == 0:
            print("Demo environment stopped.")
        return rc

    # Start the demo
    print("Starting demo environment...")
    rc = _compose_up(docker_dir)
    if rc != 0:
        print("Failed to start demo environment.")
        return rc

    # Wait for health
    health_url = "http://localhost:8000/health"
    print(f"Waiting for API server ({health_url})...", end="", flush=True)
    if _wait_for_health(health_url, timeout=args.timeout):
        print(" ready!")
        _print_usage_examples()
        return 0
    else:
        print(" timed out.")
        print(
            f"API server did not become healthy within {args.timeout}s. "
            "Check logs with: docker compose -f <docker-dir>/docker-compose.yml logs"
        )
        return 1
