"""JSONL-based storage for user feedback (learn_spam / learn_ham).

Feedback records are appended to a JSONL file.  Each record contains the
anonymised text, label, source MTA, and a unique identifier.  The store
reuses :func:`tobira.monitoring.store.append_record` /
:func:`tobira.monitoring.store.read_records` for I/O.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tobira.monitoring.store import append_record, read_records
from tobira.preprocessing.anonymizer import anonymize

DEFAULT_FEEDBACK_PATH = "/var/lib/tobira/feedback.jsonl"


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
