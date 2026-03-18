"""Storage utilities for prediction metrics.

Provides JSONL file I/O, an optional Redis-backed score store for
drift detection, and a pluggable store abstraction for external backends
(PostgreSQL, Redis).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Store protocol ──────────────────────────────────────────────


@runtime_checkable
class StoreProtocol(Protocol):
    """Abstract interface for record storage backends.

    Implementations must support appending records to named collections
    and reading all records from a collection.
    """

    def append(self, collection: str, record: dict[str, Any]) -> None:
        """Append a single record to *collection*."""
        ...

    def read_all(self, collection: str) -> list[dict[str, Any]]:
        """Read all records from *collection*.

        Returns an empty list when the collection does not exist.
        """
        ...


# ── JSONL store (default) ──────────────────────────────────────


class JsonlStore:
    """JSONL file-backed store.

    Each *collection* maps to a ``<base_dir>/<collection>.jsonl`` file.

    Args:
        base_dir: Root directory for JSONL files.
    """

    def __init__(self, base_dir: str | Path = "/var/lib/tobira") -> None:
        self._base_dir = Path(base_dir)

    def _path_for(self, collection: str) -> Path:
        return self._base_dir / f"{collection}.jsonl"

    def append(self, collection: str, record: dict[str, Any]) -> None:
        append_record(self._path_for(collection), record)

    def read_all(self, collection: str) -> list[dict[str, Any]]:
        path = self._path_for(collection)
        if not path.exists():
            return []
        return read_records(path)


# ── PostgreSQL store ───────────────────────────────────────────


class PostgresStore:
    """PostgreSQL-backed record store.

    Records are stored in a single ``records`` table as JSONB with a
    ``collection`` discriminator column.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``postgresql://user:pass@localhost/tobira``).
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
    """

    def __init__(
        self,
        dsn: str,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
    ) -> None:
        try:
            import psycopg  # noqa: F401
            import psycopg_pool  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "psycopg and psycopg_pool packages are required for "
                "PostgresStore. Install with: pip install tobira[postgres]"
            ) from exc

        from psycopg_pool import ConnectionPool

        self._pool: Any = ConnectionPool(
            conninfo=dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the records table if it does not exist."""
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id BIGSERIAL PRIMARY KEY,
                    collection TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_records_collection
                ON records (collection)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_records_created_at
                ON records (created_at)
                """
            )
            conn.commit()

    def append(self, collection: str, record: dict[str, Any]) -> None:
        import psycopg.types.json

        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO records (collection, data) VALUES (%s, %s)",
                (collection, psycopg.types.json.Jsonb(record)),
            )
            conn.commit()

    def read_all(self, collection: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT data FROM records WHERE collection = %s "
                "ORDER BY id",
                (collection,),
            ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()


# ── Redis store ────────────────────────────────────────────────


class RedisRecordStore:
    """Redis-backed record store using lists.

    Each *collection* maps to a Redis list at
    ``<key_prefix>:records:<collection>``.

    Args:
        redis_url: Redis connection URL.
        key_prefix: Prefix for all Redis keys.
        max_connections: Maximum number of connections in the pool.
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "tobira",
        max_connections: int = 10,
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisRecordStore. "
                "Install with: pip install tobira[redis]"
            ) from exc

        self._client: Any = redis_lib.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
        )
        self._key_prefix = key_prefix

    def _key_for(self, collection: str) -> str:
        return f"{self._key_prefix}:records:{collection}"

    def append(self, collection: str, record: dict[str, Any]) -> None:
        self._client.rpush(
            self._key_for(collection),
            json.dumps(record, ensure_ascii=False),
        )

    def read_all(self, collection: str) -> list[dict[str, Any]]:
        key = self._key_for(collection)
        raw: list[str] = self._client.lrange(key, 0, -1)
        return [json.loads(item) for item in raw]

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()


# ── Factory ────────────────────────────────────────────────────


def create_store(config: dict[str, Any]) -> StoreProtocol:
    """Create a store instance from a configuration dict.

    The ``type`` key selects the backend:

    - ``"jsonl"`` (default): :class:`JsonlStore`
    - ``"postgres"``: :class:`PostgresStore`
    - ``"redis"``: :class:`RedisRecordStore`

    Remaining keys are forwarded to the chosen constructor.

    Args:
        config: Store configuration dict.

    Returns:
        A :class:`StoreProtocol` implementation.

    Raises:
        ValueError: If the ``type`` is unknown.

    Examples::

        # JSONL (default)
        store = create_store({"type": "jsonl", "base_dir": "/data"})

        # PostgreSQL
        store = create_store({
            "type": "postgres",
            "dsn": "postgresql://localhost/tobira",
        })

        # Redis
        store = create_store({
            "type": "redis",
            "redis_url": "redis://localhost:6379/0",
        })
    """
    cfg = dict(config)
    store_type = cfg.pop("type", "jsonl")

    if store_type == "jsonl":
        return JsonlStore(**cfg)
    elif store_type == "postgres":
        return PostgresStore(**cfg)
    elif store_type == "redis":
        return RedisRecordStore(**cfg)
    else:
        raise ValueError(f"Unknown store type: {store_type!r}")


# ── Legacy JSONL helpers (backward compatible) ─────────────────


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


# ── Redis score store (drift detection) ────────────────────────


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
