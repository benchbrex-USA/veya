"""Per-slice robustness gates (G1-G4) for selective automation.

Consumes per-case evaluation results -- one prediction per case, each case
tagged with a slice (``clean``, ``short_prefix``, ``long_suffix``, ...) -- and
applies the frozen selective policy unchanged to every slice. The gates ask
whether the policy's validation-selected epsilon claim survives the shift:

    G1 clean safety        clean risk UCB95 <= epsilon
    G2 shift safety        every non-clean slice risk UCB95 <= epsilon
    G3 coverage retention  every non-clean slice keeps >= coverage_retention_min
                           of the clean slice's coverage
    G4 collapse floor      no slice falls below the absolute minimums

Every case is a mapping with keys ``prediction``, ``gold_label``, ``correct``,
``probabilities``, and ``slice_tag``. The threshold comes from the frozen
policy (``thresholds[operating_epsilon]``); this module never selects a
threshold and never retunes the policy.

The gates are fail-closed: a statistic that cannot support its claim fails the
gate it feeds. A missing clean slice, no non-clean slices, or a slice that
accepts no rows (``risk_ucb95`` undefined) all fail rather than passing
vacuously, because the protocol claim would be unsupported.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from veya.common import ece_score
from veya.selective import apply_report, score_probabilities

CLEAN_SLICE = "clean"


def _max_confidence(cases: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Normalized top-class probability per case.

    ECE confidence is max probability regardless of the gating score kind
    (``neg_entropy`` scores are negative and cannot be binned in [0, 1]).
    """
    values = []
    for case in cases:
        p = np.asarray(case["probabilities"], dtype=np.float64)
        p = p / p.sum()
        values.append(float(p.max()))
    return np.asarray(values, dtype=np.float64)


def evaluate_slice(cases, policy, epsilon, slice_tag: Optional[str] = None) -> dict:
    """Selective-automation statistics for one slice under the frozen policy.

    Returns the ``apply_report`` fields (accepted, deferred, incorrect,
    coverage, observed_risk, wilson95, risk_ucb95, supports_epsilon_claim, ...)
    plus ``slice_tag``, overall ``accuracy``, and ``ece`` (binned on max
    probability). An empty slice reports ``accuracy``/``ece`` as None.
    """
    cases = list(cases)
    seen = {str(case["slice_tag"]) for case in cases}
    if len(seen) > 1:
        raise ValueError("evaluate_slice expects one slice; got %s" % sorted(seen))
    if slice_tag is None:
        slice_tag = next(iter(seen)) if seen else None
    operating = policy["operating_epsilon"]
    thresholds = policy.get("thresholds") or {}
    if operating not in thresholds:
        raise ValueError(
            "selective policy is missing its operating threshold for %r" % operating
        )
    threshold = thresholds[operating]
    scores = np.asarray(
        [score_probabilities(case["probabilities"], policy["score"]) for case in cases],
        dtype=np.float64,
    )
    correct = np.asarray(
        [1.0 if case["correct"] else 0.0 for case in cases], dtype=np.float64
    )
    result = dict(apply_report(scores, correct, threshold, epsilon=epsilon))
    result["slice_tag"] = slice_tag
    result["accuracy"] = float(correct.mean()) if cases else None
    result["ece"] = (
        float(ece_score(_max_confidence(cases), correct)) if cases else None
    )
    return result


def evaluate_all_slices(cases, policy, epsilon) -> dict[str, dict]:
    """Group cases by ``slice_tag`` and evaluate each slice under the same policy."""
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(str(case["slice_tag"]), []).append(case)
    return {
        tag: evaluate_slice(group, policy, epsilon, slice_tag=tag)
        for tag, group in groups.items()
    }


def evaluate_gates(slice_results, epsilon, coverage_retention_min, collapse_floor) -> dict:
    """Evaluate gates G1-G4 over per-slice selective-automation statistics.

    ``slice_results`` maps slice tag to an ``evaluate_slice``-shaped dict.
    ``collapse_floor`` maps a result metric name to its absolute minimum
    (lower-is-worse metrics only, e.g. ``{"coverage": 0.05, "accuracy": 0.30}``);
    a missing or unknown metric counts as below its floor.

    Returns ``{"gates": {name: {"passed": bool, ...details}}, "failed_gates":
    [...], "passed": bool}``.
    """
    epsilon = float(epsilon)
    retention = float(coverage_retention_min)
    floors = dict(collapse_floor or {})
    clean = slice_results.get(CLEAN_SLICE)
    shifted = {tag: res for tag, res in slice_results.items() if tag != CLEAN_SLICE}

    clean_risk = clean.get("risk_ucb95") if clean else None
    if clean is None:
        g1_reason = "no clean slice in the results"
    elif clean_risk is None:
        g1_reason = "clean slice accepted no rows; risk_ucb95 is undefined"
    elif float(clean_risk) > epsilon:
        g1_reason = "clean risk_ucb95 %.6f exceeds epsilon %.6f" % (clean_risk, epsilon)
    else:
        g1_reason = "clean risk_ucb95 %.6f within epsilon %.6f" % (clean_risk, epsilon)
    gates = {
        "G1_clean_safety": {
            "passed": bool(
                clean is not None and clean_risk is not None and float(clean_risk) <= epsilon
            ),
            "risk_ucb95": clean_risk,
            "epsilon": epsilon,
            "observed_risk": clean.get("observed_risk") if clean else None,
            "coverage": clean.get("coverage") if clean else None,
            "reason": g1_reason,
        }
    }

    per_slice = {}
    for tag, res in shifted.items():
        risk = res.get("risk_ucb95")
        ok = risk is not None and float(risk) <= epsilon
        per_slice[tag] = {
            "risk_ucb95": risk,
            "passed": ok,
            "reason": (
                "accepted no rows; risk_ucb95 is undefined"
                if risk is None
                else "risk_ucb95 %.6f %s epsilon %.6f" % (risk, "<=" if ok else ">", epsilon)
            ),
        }
    failing = [tag for tag, row in per_slice.items() if not row["passed"]]
    if not shifted:
        g2_reason = "no non-clean slices; the shift claim is unsupported"
    elif failing:
        g2_reason = "risk_ucb95 above epsilon in %s" % ", ".join(failing)
    else:
        g2_reason = "risk_ucb95 within epsilon on all %d non-clean slices" % len(shifted)
    gates["G2_shift_safety"] = {
        "passed": bool(shifted) and not failing,
        "failing_slices": failing,
        "per_slice": per_slice,
        "reason": g2_reason,
    }

    clean_coverage = clean.get("coverage") if clean else None
    required = (
        float(clean_coverage) * retention if clean_coverage is not None else None
    )
    per_slice = {}
    for tag, res in shifted.items():
        coverage = res.get("coverage")
        ok = bool(
            clean_coverage is not None
            and float(clean_coverage) > 0.0
            and coverage is not None
            and float(coverage) >= required
        )
        per_slice[tag] = {
            "coverage": coverage,
            "required": required,
            "retention": (
                float(coverage) / float(clean_coverage)
                if coverage is not None and clean_coverage
                else None
            ),
            "passed": ok,
        }
    failing = [tag for tag, row in per_slice.items() if not row["passed"]]
    if clean_coverage is None:
        g3_reason = "no clean coverage to retain (clean slice missing or empty)"
    elif float(clean_coverage) <= 0.0:
        g3_reason = "clean coverage is zero; retention is undefined"
    elif not shifted:
        g3_reason = "no non-clean slices; coverage retention is unsupported"
    elif failing:
        g3_reason = "coverage below %.6f of clean in %s" % (retention, ", ".join(failing))
    else:
        g3_reason = "coverage retained on all %d non-clean slices" % len(shifted)
    gates["G3_coverage_retention"] = {
        "passed": bool(
            shifted
            and clean_coverage is not None
            and float(clean_coverage) > 0.0
            and not failing
        ),
        "coverage_retention_min": retention,
        "clean_coverage": clean_coverage,
        "required_coverage": required,
        "failing_slices": failing,
        "per_slice": per_slice,
        "reason": g3_reason,
    }

    violations = []
    for tag, res in slice_results.items():
        for metric, floor in floors.items():
            value = res.get(metric)
            if value is None or float(value) < float(floor):
                violations.append(
                    {
                        "slice_tag": tag,
                        "metric": metric,
                        "value": value,
                        "floor": float(floor),
                    }
                )
    gates["G4_collapse_floor"] = {
        "passed": not violations,
        "floors": floors,
        "violations": violations,
        "reason": (
            "all slices at or above the collapse floor"
            if not violations
            else "below floor: %s"
            % ", ".join("%s.%s" % (v["slice_tag"], v["metric"]) for v in violations)
        ),
    }

    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {"gates": gates, "failed_gates": failed, "passed": not failed}


def gate_report(cases, policy, epsilon, coverage_retention_min, collapse_floor) -> dict:
    """Full gate report: per-slice statistics, G1-G4 verdicts, and provenance echo."""
    cases = list(cases)
    slices = evaluate_all_slices(cases, policy, epsilon)
    outcome = evaluate_gates(slices, epsilon, coverage_retention_min, collapse_floor)
    operating = policy.get("operating_epsilon")
    thresholds = policy.get("thresholds") if isinstance(policy.get("thresholds"), dict) else {}
    return {
        "format": "veya-selective-gates-v1",
        "n_cases": len(cases),
        "policy": {
            "score": policy.get("score"),
            "operating_epsilon": operating,
            "threshold": thresholds.get(operating),
        },
        "protocol": {
            "epsilon": float(epsilon),
            "coverage_retention_min": float(coverage_retention_min),
            "collapse_floor": dict(collapse_floor or {}),
        },
        "slices": slices,
        "gates": outcome["gates"],
        "failed_gates": outcome["failed_gates"],
        "passed": outcome["passed"],
    }
