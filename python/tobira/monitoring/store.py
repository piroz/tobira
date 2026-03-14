"""Storage utilities for prediction metrics.

Provides JSONL file I/O and an optional Redis-backed score store for
drift detection.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


class RedisScoreStore:
    """Redis-backed store for prediction scores used in drift detection.

    Scores are stored in a sorted set keyed by UNIX timestamp, allowing
    efficient range queries for baseline/current window comparison.

    A separate key holds the baseline score snapshot used as a reference
    distribution.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        key_prefix: Prefix for all Redis keys.
        window_seconds: Maximum age of scores to keep (default 86400 = 24h).
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "tobira:scores",
        window_seconds: int = 86400,
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisScoreStore. "
                "Install with: pip install tobira[redis]"
            ) from exc

        self._client: Any = redis_lib.from_url(redis_url, decode_responses=True)
        self._scores_key = f"{key_prefix}:current"
        self._baseline_key = f"{key_prefix}:baseline"
        self._window_seconds = window_seconds

    def push_score(self, score: float, timestamp: float | None = None) -> None:
        """Add a prediction score with its timestamp.

        Args:
            score: The prediction confidence score.
            timestamp: UNIX timestamp. Defaults to current time.
        """
        ts = timestamp if timestamp is not None else time.time()
        # Use timestamp as sorted-set score, value is "ts:prediction_score"
        member = f"{ts}:{score}"
        self._client.zadd(self._scores_key, {member: ts})
        self._trim_old_scores()

    def get_scores(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> list[float]:
        """Retrieve prediction scores within a time range.

        Args:
            start: Start UNIX timestamp (inclusive). Defaults to
                ``now - window_seconds``.
            end: End UNIX timestamp (inclusive). Defaults to now.

        Returns:
            List of prediction scores in the time range.
        """
        now = time.time()
        if start is None:
            start = now - self._window_seconds
        if end is None:
            end = now

        members: list[str] = self._client.zrangebyscore(
            self._scores_key, start, end,
        )
        scores: list[float] = []
        for m in members:
            parts = m.split(":", 1)
            if len(parts) == 2:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    continue
        return scores

    def save_baseline(self, scores: list[float]) -> None:
        """Save a baseline score distribution snapshot.

        Args:
            scores: List of scores to use as the reference distribution.
        """
        pipeline: Any = self._client.pipeline()
        pipeline.delete(self._baseline_key)
        if scores:
            members = {f"{i}:{s}": float(i) for i, s in enumerate(scores)}
            pipeline.zadd(self._baseline_key, members)
        pipeline.execute()

    def get_baseline(self) -> list[float]:
        """Retrieve the stored baseline score distribution.

        Returns:
            List of baseline scores, or empty list if no baseline is stored.
        """
        members: list[str] = self._client.zrangebyscore(
            self._baseline_key, "-inf", "+inf",
        )
        scores: list[float] = []
        for m in members:
            parts = m.split(":", 1)
            if len(parts) == 2:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    continue
        return scores

    def _trim_old_scores(self) -> None:
        """Remove scores older than the configured window."""
        cutoff = time.time() - self._window_seconds
        self._client.zremrangebyscore(self._scores_key, "-inf", cutoff)

    def clear(self) -> None:
        """Remove all stored scores and baseline data."""
        self._client.delete(self._scores_key, self._baseline_key)
