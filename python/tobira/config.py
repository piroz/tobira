"""Shared configuration loading utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def load_toml(config_path: str) -> dict[str, Any]:
    """Load and parse a TOML configuration file.

    Args:
        config_path: Path to the TOML file.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[import-not-found]
    else:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib

    with open(path, "rb") as f:
        result: dict[str, Any] = tomllib.load(f)
        return result
