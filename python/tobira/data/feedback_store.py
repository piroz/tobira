"""Pluggable storage for user feedback (learn_spam / learn_ham).

Feedback records are persisted via a :class:`~tobira.monitoring.store.StoreProtocol`
backend.  The default backend writes JSONL files, but PostgreSQL or Redis
can be selected through :func:`~tobira.monitoring.store.create_store`.

Legacy path-based helpers (:func:`store_feedback`, :func:`load_feedback`)
are retained for backward compatibility.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tobira.monitoring.store import append_record, read_records
from tobira.preprocessing.anonymizer import anonymize

if TYPE_CHECKING:
    from tobira.monitoring.store import StoreProtocol

DEFAULT_FEEDBACK_PATH = "/var/lib/tobira/feedback.jsonl"

_FEEDBACK_COLLECTION = "feedback"


@dataclass(frozen=True)
class FeedbackRecord:
    """A single feedback entry.

    Attributes:
        id: Unique identifier for the feedback.
        text: Anonymised email text.
        label: ``"spam"`` or ``"ham"``.
        source: Reporting MTA (e.g. ``"rspamd"``).
        timestamp: ISO-8601 timestamp of when the feedback was received.
    """

    id: str
    text: str
    label: str
    source: str
    timestamp: str


# ── Store-aware API ────────────────────────────────────────────


class FeedbackStore:
    """Feedback storage backed by a :class:`StoreProtocol` implementation.

    Args:
        store: A :class:`StoreProtocol` backend (e.g. :class:`JsonlStore`,
            :class:`PostgresStore`, :class:`RedisRecordStore`).
        collection: Name of the collection within the store.
    """

    def __init__(
        self,
        store: StoreProtocol,
        collection: str = _FEEDBACK_COLLECTION,
    ) -> None:
        self._store = store
        self._collection = collection

    def save(
        self,
        text: str,
        label: str,
        source: str,
    ) -> FeedbackRecord:
        """Anonymise *text* and persist the feedback record.

        Args:
            text: Raw email text reported by the user / MTA plugin.
            label: ``"spam"`` or ``"ham"``.
            source: Identifier for the reporting MTA.

        Returns:
            The persisted :class:`FeedbackRecord`.
        """
        anon = anonymize(text)
        record = FeedbackRecord(
            id=str(uuid.uuid4()),
            text=anon.text,
            label=label,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._store.append(self._collection, _record_to_dict(record))
        return record

    def load_all(self) -> list[FeedbackRecord]:
        """Load all feedback records from the store.

        Returns:
            A list of :class:`FeedbackRecord` instances.
        """
        raw = self._store.read_all(self._collection)
        return [_dict_to_record(r) for r in raw]


# ── Legacy path-based API (backward compatible) ────────────────


def store_feedback(
    text: str,
    label: str,
    source: str,
    *,
    path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> FeedbackRecord:
    """Anonymise *text* and persist the feedback record.

    Args:
        text: Raw email text reported by the user / MTA plugin.
        label: ``"spam"`` or ``"ham"``.
        source: Identifier for the reporting MTA.
        path: Path to the JSONL feedback file.

    Returns:
        The persisted :class:`FeedbackRecord`.
    """
    anon = anonymize(text)
    record = FeedbackRecord(
        id=str(uuid.uuid4()),
        text=anon.text,
        label=label,
        source=source,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    append_record(path, _record_to_dict(record))
    return record


def load_feedback(path: str | Path = DEFAULT_FEEDBACK_PATH) -> list[FeedbackRecord]:
    """Load all feedback records from a JSONL file.

    Args:
        path: Path to the JSONL feedback file.

    Returns:
        A list of :class:`FeedbackRecord` instances.
    """
    raw = read_records(path)
    return [_dict_to_record(r) for r in raw]


# ── Helpers ────────────────────────────────────────────────────


def _record_to_dict(record: FeedbackRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "text": record.text,
        "label": record.label,
        "source": record.source,
        "timestamp": record.timestamp,
    }


def _dict_to_record(d: dict[str, Any]) -> FeedbackRecord:
    return FeedbackRecord(
        id=d["id"],
        text=d["text"],
        label=d["label"],
        source=d["source"],
        timestamp=d["timestamp"],
    )
