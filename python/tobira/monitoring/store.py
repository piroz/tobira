"""JSONL storage utilities for prediction metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file.

    Args:
        path: Path to the JSONL file. Parent directories are created if needed.
        record: Dictionary to serialise as a single JSON line.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    records: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
