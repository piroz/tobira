"""Concept drift detection algorithms.

Provides PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests
for detecting score distribution shifts between a baseline and current window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_PSI_WARNING = 0.1
_PSI_ALERT = 0.25
_DEFAULT_BINS = 10


@dataclass(frozen=True)
class DriftResult:
    """Result of a combined drift detection analysis.

    Attributes:
        psi: Population Stability Index value.
        psi_level: Severity level (``ok``, ``warning``, or ``alert``).
        ks_statistic: Kolmogorov-Smirnov statistic.
        ks_pvalue: Approximate p-value from the KS test.
        drifted: Whether drift was detected (PSI exceeds threshold or KS
            is significant at alpha=0.05).
    """

    psi: float
    psi_level: str
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


def compute_psi(
    baseline: list[float],
    current: list[float],
    bins: int = _DEFAULT_BINS,
) -> float:
    """Compute the Population Stability Index between two score distributions.

    Args:
        baseline: Reference score distribution.
        current: Current score distribution.
        bins: Number of histogram bins.

    Returns:
        The PSI value.  A value below 0.1 indicates no significant change,
        0.1--0.25 indicates moderate change, and above 0.25 indicates
        significant change.

    Raises:
        ValueError: If either distribution is empty.
    """
    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Both distributions must be non-empty.")

    lo = min(min(baseline), min(current))
    hi = max(max(baseline), max(current))
    if lo == hi:
        return 0.0

    bin_width = (hi - lo) / bins
    eps = 1e-6

    def _bin_proportions(values: list[float]) -> list[float]:
        counts = [0] * bins
        for v in values:
            idx = min(int((v - lo) / bin_width), bins - 1)
            counts[idx] += 1
        total = len(values)
        return [(c / total) + eps for c in counts]

    ref_props = _bin_proportions(baseline)
    cur_props = _bin_proportions(current)

    psi = sum(
        (cur_props[i] - ref_props[i]) * math.log(cur_props[i] / ref_props[i])
        for i in range(bins)
    )
    return round(psi, 6)


def compute_ks_test(
    baseline: list[float],
    current: list[float],
) -> tuple[float, float]:
    """Compute the Kolmogorov-Smirnov statistic and approximate p-value.

    Uses the asymptotic approximation for the two-sample KS test.

    Args:
        baseline: Reference score distribution.
        current: Current score distribution.

    Returns:
        A tuple of ``(statistic, p_value)``.

    Raises:
        ValueError: If either distribution is empty.
    """
    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Both distributions must be non-empty.")

    n1 = len(baseline)
    n2 = len(current)
    combined = sorted(set(baseline + current))

    def _cdf(values: list[float], point: float) -> float:
        return sum(1 for v in values if v <= point) / len(values)

    max_diff = 0.0
    for x in combined:
        diff = abs(_cdf(baseline, x) - _cdf(current, x))
        if diff > max_diff:
            max_diff = diff

    # Asymptotic p-value: P(D > d) ≈ 2 * exp(-2 * n_eff * d^2)
    n_eff = (n1 * n2) / (n1 + n2)
    if max_diff == 0.0:
        p_value = 1.0
    else:
        exponent = -2.0 * n_eff * max_diff * max_diff
        p_value = min(1.0, 2.0 * math.exp(exponent))

    return round(max_diff, 6), round(p_value, 6)


def _classify_psi(psi: float) -> str:
    """Classify a PSI value into a severity level."""
    if psi >= _PSI_ALERT:
        return "alert"
    if psi >= _PSI_WARNING:
        return "warning"
    return "ok"


def detect_drift(
    baseline: list[float],
    current: list[float],
    psi_threshold: float = 0.2,
) -> DriftResult:
    """Run combined PSI and KS drift detection.

    Args:
        baseline: Reference score distribution.
        current: Current score distribution.
        psi_threshold: PSI value above which drift is considered detected.

    Returns:
        A :class:`DriftResult` summarising the analysis.

    Raises:
        ValueError: If either distribution is empty.
    """
    psi = compute_psi(baseline, current)
    ks_stat, ks_pvalue = compute_ks_test(baseline, current)
    psi_level = _classify_psi(psi)
    drifted = psi >= psi_threshold or ks_pvalue < 0.05

    return DriftResult(
        psi=psi,
        psi_level=psi_level,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        drifted=drifted,
    )
