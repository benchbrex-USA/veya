"""Minimum sample sizes that keep a Wilson upper bound at or below epsilon.

Protocol Section 7 reference values (epsilon = 0.02, 95% two-sided Wilson):

    errors = 0 -> 189 accepted cases
    errors = 1 -> 280 accepted cases
    errors = 2 -> 361 accepted cases
    errors = 3 -> 437 accepted cases

``required_n`` searches for the smallest number of accepted cases whose Wilson
upper bound stays at or below ``epsilon`` when exactly ``max_errors`` errors
occur, then scales to the total sample size ``N = accepted /
anticipated_coverage`` (rounded up, so the expected accepted count still
clears the requirement). Pure stdlib math: no model, no data, no I/O.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

_MAX_DOUBLINGS = 1024


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error


def _z_score(confidence: float) -> float:
    """Two-sided normal quantile for ``confidence`` (1.96 for 95%)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a number in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _wilson_ucb(errors: int, accepted: int, z: float) -> float:
    """Wilson score-interval upper bound on the rate ``errors / accepted``."""
    p = errors / accepted
    spread = p * (1.0 - p) / accepted + z * z / (4.0 * accepted * accepted)
    margin = z * math.sqrt(spread)
    return (p + z * z / (2.0 * accepted) + margin) / (1.0 + z * z / accepted)


def wilson_upper_bound(errors: int, accepted: int, confidence: float = 0.95) -> float:
    """Wilson ``confidence``-level upper bound on the error rate errors/accepted."""
    if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted < 1:
        raise ValueError("accepted must be a positive integer")
    if isinstance(errors, bool) or not isinstance(errors, int):
        raise ValueError("errors must be an integer")
    if not 0 <= errors <= accepted:
        raise ValueError("errors must be between 0 and accepted")
    confidence = _as_float(confidence, "confidence")
    return _wilson_ucb(errors, accepted, _z_score(confidence))


def _minimum_accepted(epsilon: float, errors: int, z: float) -> int:
    """Smallest ``n`` with ``wilson_ucb(errors, n) <= epsilon``.

    The upper bound shrinks monotonically as ``n`` grows at fixed ``errors``,
    so an exponential bracket followed by a binary search finds the minimum.
    """

    def ucb(n: int) -> float:
        return _wilson_ucb(errors, n, z)

    low = max(1, errors)
    if ucb(low) <= epsilon:
        return low
    high = 2 * low
    doublings = 0
    while ucb(high) > epsilon:
        high *= 2
        doublings += 1
        if doublings > _MAX_DOUBLINGS:
            raise ValueError("epsilon is too small to resolve with a bounded search")
    # Invariant: ucb(low) > epsilon and ucb(high) <= epsilon.
    while high - low > 1:
        middle = (low + high) // 2
        if ucb(middle) <= epsilon:
            high = middle
        else:
            low = middle
    return high


def required_n(
    epsilon: float,
    anticipated_coverage: float,
    max_errors: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Minimum sample sizes for an epsilon-bounded automation claim.

    Finds the smallest ``accepted`` count such that the Wilson ``confidence``
    upper bound on the error rate stays at or below ``epsilon`` when exactly
    ``max_errors`` errors are observed, then returns the total sample size
    ``total = ceil(accepted / anticipated_coverage)``.

    Returns a dict with ``epsilon``, ``confidence``, ``z``, ``max_errors``,
    ``anticipated_coverage``, ``accepted``, ``total``, and
    ``wilson_ucb_at_accepted`` (the bound actually achieved).
    """
    epsilon = _as_float(epsilon, "epsilon")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be a number in (0, 1)")
    coverage = _as_float(anticipated_coverage, "anticipated_coverage")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("anticipated_coverage must be a number in (0, 1]")
    if isinstance(max_errors, bool) or not isinstance(max_errors, int):
        raise ValueError("max_errors must be a non-negative integer")
    if max_errors < 0:
        raise ValueError("max_errors must be a non-negative integer")
    confidence = _as_float(confidence, "confidence")
    z = _z_score(confidence)
    accepted = _minimum_accepted(epsilon, max_errors, z)
    total = math.ceil(round(accepted / coverage, 9))
    return {
        "epsilon": epsilon,
        "confidence": confidence,
        "z": z,
        "max_errors": max_errors,
        "anticipated_coverage": coverage,
        "accepted": accepted,
        "total": total,
        "wilson_ucb_at_accepted": _wilson_ucb(max_errors, accepted, z),
    }
