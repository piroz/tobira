"""Email header analysis for spam risk scoring.

Extracts structured features from email headers (SPF, DKIM, DMARC,
Reply-To mismatch, etc.) and computes a header-based spam risk score.
The score is independent of the ML text classification and can be
combined with it at the server level for improved accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Authentication result values
AUTH_PASS = "pass"
AUTH_FAIL = "fail"
AUTH_SOFTFAIL = "softfail"
AUTH_NONE = "none"

_VALID_AUTH_RESULTS = {AUTH_PASS, AUTH_FAIL, AUTH_SOFTFAIL, AUTH_NONE}

# Default weights for each header feature contribution to spam risk
DEFAULT_WEIGHTS: dict[str, float] = {
    "spf": 0.20,
    "dkim": 0.25,
    "dmarc": 0.15,
    "reply_to_mismatch": 0.15,
    "hop_count": 0.10,
    "content_type_anomaly": 0.15,
}


@dataclass(frozen=True)
class HeaderFeatures:
    """Extracted header features as numeric risk scores (0.0 = safe, 1.0 = risky)."""

    spf: float = 0.0
    dkim: float = 0.0
    dmarc: float = 0.0
    reply_to_mismatch: float = 0.0
    hop_count: float = 0.0
    content_type_anomaly: float = 0.0


def _auth_risk(value: str | None) -> float:
    """Convert an authentication result string to a risk score."""
    if value is None or value == AUTH_NONE:
        return 0.5
    v = value.lower().strip()
    if v == AUTH_PASS:
        return 0.0
    if v == AUTH_FAIL:
        return 1.0
    if v == AUTH_SOFTFAIL:
        return 0.7
    return 0.5


def _extract_domain(address: str) -> str:
    """Extract the domain part from an email address."""
    address = address.strip()
    if "<" in address and ">" in address:
        address = address.split("<")[1].split(">")[0]
    if "@" in address:
        return address.rsplit("@", 1)[1].lower().strip()
    return address.lower().strip()


def _reply_to_mismatch_risk(
    from_addr: str | None, reply_to: str | None
) -> float:
    """Score Reply-To vs From domain mismatch."""
    if not reply_to or not from_addr:
        return 0.0
    from_domain = _extract_domain(from_addr)
    reply_domain = _extract_domain(reply_to)
    if not from_domain or not reply_domain:
        return 0.0
    if from_domain == reply_domain:
        return 0.0
    return 1.0


def _hop_count_risk(received: list[str] | None) -> float:
    """Score based on Received header hop count.

    Typical legitimate mail has 3-8 hops. Very few or very many hops
    are slightly suspicious.
    """
    if not received:
        return 0.0
    count = len(received)
    if count <= 8:
        return 0.0
    if count <= 15:
        return 0.3
    return 0.6


def _content_type_anomaly_risk(content_type: str | None) -> float:
    """Score Content-Type anomalies."""
    if not content_type:
        return 0.0
    ct = content_type.lower()
    # Deeply nested multipart or unusual charsets are suspicious
    if ct.count("multipart") > 1:
        return 0.7
    if "charset" in ct:
        for suspicious in ("koi8", "gb2312", "big5"):
            if suspicious in ct:
                return 0.3
    return 0.0


def extract_features(headers: dict[str, object]) -> HeaderFeatures:
    """Extract header features from a header dictionary.

    Args:
        headers: Dictionary of header fields. Expected keys (all optional):
            - ``spf``: SPF result ("pass", "fail", "softfail", "none")
            - ``dkim``: DKIM result ("pass", "fail", "none")
            - ``dmarc``: DMARC result ("pass", "fail", "none")
            - ``from``: From address
            - ``reply_to``: Reply-To address
            - ``received``: List of Received header values
            - ``content_type``: Content-Type header value

    Returns:
        HeaderFeatures with numeric risk scores for each dimension.
    """
    spf_val = headers.get("spf")
    dkim_val = headers.get("dkim")
    dmarc_val = headers.get("dmarc")
    from_addr = headers.get("from")
    reply_to = headers.get("reply_to")
    received = headers.get("received")
    content_type = headers.get("content_type")

    return HeaderFeatures(
        spf=_auth_risk(str(spf_val) if spf_val is not None else None),
        dkim=_auth_risk(str(dkim_val) if dkim_val is not None else None),
        dmarc=_auth_risk(str(dmarc_val) if dmarc_val is not None else None),
        reply_to_mismatch=_reply_to_mismatch_risk(
            str(from_addr) if from_addr is not None else None,
            str(reply_to) if reply_to is not None else None,
        ),
        hop_count=_hop_count_risk(
            list(received) if isinstance(received, list) else None
        ),
        content_type_anomaly=_content_type_anomaly_risk(
            str(content_type) if content_type is not None else None
        ),
    )


def compute_header_score(
    features: HeaderFeatures,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted spam risk score from header features.

    Args:
        features: Extracted header features.
        weights: Optional custom weights. Defaults to ``DEFAULT_WEIGHTS``.

    Returns:
        A float in [0.0, 1.0] representing the header-based spam risk.
    """
    w = weights or DEFAULT_WEIGHTS
    total_weight = sum(w.values())
    if total_weight == 0:
        return 0.0

    score = (
        features.spf * w.get("spf", 0.0)
        + features.dkim * w.get("dkim", 0.0)
        + features.dmarc * w.get("dmarc", 0.0)
        + features.reply_to_mismatch * w.get("reply_to_mismatch", 0.0)
        + features.hop_count * w.get("hop_count", 0.0)
        + features.content_type_anomaly * w.get("content_type_anomaly", 0.0)
    )
    return max(0.0, min(1.0, score / total_weight))


def analyze_headers(
    headers: dict[str, object],
    weights: dict[str, float] | None = None,
) -> float:
    """Convenience function: extract features and compute score in one call.

    Args:
        headers: Raw header dictionary (see ``extract_features`` for keys).
        weights: Optional custom weights.

    Returns:
        Header-based spam risk score in [0.0, 1.0].
    """
    features = extract_features(headers)
    return compute_header_score(features, weights)
