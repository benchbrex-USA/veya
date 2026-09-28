"""Selective automation thresholds.

A threshold is chosen on a validation split and applied unchanged to a later
split. The score is higher when the model is more confident. This module does
not train a policy.
"""
import json
import math

import numpy as np


def entropy(probs) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def score_probabilities(probs, kind: str) -> float:
    """Map a probability vector to a confidence score. Higher means more confident."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    if kind == "max_probability":
        return float(p.max())
    if kind == "neg_entropy":
        return float(-entropy(p))
    if kind == "margin":
        if len(p) < 2:
            return float(p.max())
        top = np.partition(p, -2)
        return float(top[-1] - top[-2])
    raise ValueError("unknown selective score %r" % kind)


def select_threshold(scores, correct, epsilon: float):
    """Lowest score still inside the longest high-confidence prefix with risk <= epsilon.

    Returns None when no prefix qualifies. Ties keep the earlier row, matching
    ``safe_automation_coverage``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = int(len(scores))
    if n == 0 or len(correct) != n:
        raise ValueError("scores and correct must be the same non-empty length")
    order = np.argsort(-scores, kind="mergesort")
    wrong = 0.0
    best_threshold = None
    for i, idx in enumerate(order):
        wrong += 1.0 - float(correct[idx])
        is_group_end = i == n - 1 or scores[order[i + 1]] < scores[idx] - 1e-12
        if is_group_end and wrong / (i + 1) <= float(epsilon) + 1e-12:
            best_threshold = float(scores[idx])
    return best_threshold


def apply_threshold(scores, threshold):
    """Boolean mask. A missing threshold accepts nothing."""
    scores = np.asarray(scores, dtype=np.float64)
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    return scores >= float(threshold) - 1e-12


def wilson_interval(incorrect: int, accepted: int, z: float = 1.96):
    if accepted <= 0:
        return None
    p = incorrect / accepted
    den = 1.0 + z * z / accepted
    center = (p + z * z / (2 * accepted)) / den
    margin = z * math.sqrt(max(0.0, p * (1.0 - p) / accepted + z * z / (4 * accepted * accepted))) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def risk_coverage_curve(scores, correct):
    """Descriptive curve. Sorting uses scores only; labels are counted after the sort."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    wrong = 0.0
    curve = []
    n = len(order)
    for i, idx in enumerate(order):
        wrong += 1.0 - float(correct[idx])
        accepted = i + 1
        curve.append({
            "accepted": accepted,
            "coverage": accepted / n,
            "incorrect": int(wrong),
            "risk": wrong / accepted,
        })
    return curve


def apply_report(scores, correct, threshold):
    mask = apply_threshold(scores, threshold)
    accepted = int(mask.sum())
    incorrect = int(accepted - np.asarray(correct, dtype=np.float64)[mask].sum()) if accepted else 0
    n = int(len(scores))
    interval = wilson_interval(incorrect, accepted)
    return {
        "n": n,
        "threshold": threshold,
        "accepted": accepted,
        "deferred": n - accepted,
        "incorrect": incorrect,
        "coverage": (accepted / n) if n else None,
        "observed_risk": (incorrect / accepted) if accepted else None,
        "wilson95": interval,
        "supports_1pct_claim": bool(interval is not None and interval[1] <= 0.01 and incorrect == 0),
    }


def load_policy(path, route, checkpoint_hash, shortlist_sha256, scorer_id):
    """Load a validation-selected policy and reject artifact mixing."""
    with open(path, encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("format") != "veya-selective-policy-v1":
        raise ValueError("unsupported selective policy format %r" % policy.get("format"))
    if policy.get("route") != route:
        raise ValueError("selective policy belongs to a different route")
    if policy.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("selective policy belongs to a different checkpoint")
    if policy.get("shortlist_sha256") != shortlist_sha256:
        raise ValueError("selective policy belongs to a different shortlist")
    if policy.get("scorer_id") != scorer_id:
        raise ValueError("selective policy belongs to a different scorer")
    if policy.get("chosen_on") != "validation" or not policy.get("validation_hash"):
        raise ValueError("selective policy must be chosen on a hashed validation split")
    score = policy.get("score")
    if score not in ("max_probability", "neg_entropy", "margin"):
        raise ValueError("unsupported selective score %r" % score)
    thresholds = policy.get("thresholds")
    operating = policy.get("operating_epsilon")
    if not isinstance(thresholds, dict) or operating not in thresholds:
        raise ValueError("selective policy is missing its operating threshold")
    for key, value in thresholds.items():
        if value is not None and not np.isfinite(float(value)):
            raise ValueError("selective threshold %r is not finite" % key)
    return policy
