"""Robustness gates G1-G4 for per-slice selective automation.

Hand-crafted synthetic cases with known correct/incorrect patterns pin each
gate's pass/fail boundary and the fail-closed behavior for unsupported claims.

Run: python -m pytest tests/test_selective_gates.py -xvs
"""
import json

import pytest

from research.veyabench.selective_gates import (
    evaluate_all_slices,
    evaluate_gates,
    evaluate_slice,
    gate_report,
)

EPSILON = 0.02
RETENTION_MIN = 0.95
FLOOR = {"coverage": 0.05, "accuracy": 0.30}


def _policy(score="max_probability", operating="0.02", threshold=0.9):
    return {
        "format": "veya-selective-policy-v1",
        "score": score,
        "operating_epsilon": operating,
        "thresholds": {operating: threshold},
    }


def _case(correct, confidence, tag):
    return {
        "prediction": "a" if correct else "b",
        "gold_label": "a",
        "correct": bool(correct),
        "probabilities": [confidence, 1.0 - confidence],
        "slice_tag": tag,
    }


def _cases(tag, n, confidence, wrong=0):
    """``n`` cases; the last ``wrong`` are incorrect, the rest correct."""
    return [_case(index < n - wrong, confidence, tag) for index in range(n)]


def _slice(tag, risk_ucb95=0.0, coverage=1.0, accuracy=1.0, observed_risk=0.0):
    """Hand-built ``evaluate_slice``-shaped row for direct gate unit checks."""
    return {
        "slice_tag": tag,
        "risk_ucb95": risk_ucb95,
        "coverage": coverage,
        "accuracy": accuracy,
        "observed_risk": observed_risk,
    }


def test_g1_passes_when_clean_risk_is_low():
    cases = _cases("clean", 300, 0.95) + _cases("long_suffix", 300, 0.95)
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    clean = slices["clean"]
    assert clean["n"] == 300
    assert clean["accepted"] == 300 and clean["deferred"] == 0
    assert clean["incorrect"] == 0 and clean["observed_risk"] == 0.0
    assert clean["coverage"] == 1.0
    assert clean["accuracy"] == 1.0
    assert clean["ece"] == pytest.approx(0.05, abs=1e-9)
    assert clean["risk_ucb95"] == pytest.approx(0.01264, abs=1e-4)
    assert clean["supports_epsilon_claim"] is True
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    assert outcome["gates"]["G1_clean_safety"]["passed"] is True
    assert outcome["passed"] is True
    assert outcome["failed_gates"] == []


def test_epsilon_flows_from_argument_and_can_flip_g1():
    cases = _cases("clean", 300, 0.95) + _cases("long_suffix", 300, 0.95)
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    ucb = slices["clean"]["risk_ucb95"]
    assert 0.01 < ucb <= EPSILON
    assert evaluate_gates(slices, 0.01, RETENTION_MIN, FLOOR)["gates"]["G1_clean_safety"]["passed"] is False
    assert evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)["gates"]["G1_clean_safety"]["passed"] is True


def test_g1_fails_when_clean_risk_exceeds_epsilon():
    cases = (
        _cases("clean", 150, 0.95, wrong=15)
        + _cases("clean", 150, 0.5)
        + _cases("long_suffix", 300, 0.95)
    )
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    assert slices["clean"]["observed_risk"] == pytest.approx(0.1)
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    g1 = outcome["gates"]["G1_clean_safety"]
    assert g1["passed"] is False
    assert g1["risk_ucb95"] > EPSILON
    assert "exceeds epsilon" in g1["reason"]
    assert outcome["failed_gates"] == ["G1_clean_safety"]


def test_g1_fails_when_clean_accepts_nothing():
    cases = _cases("clean", 100, 0.5) + _cases("long_suffix", 300, 0.95)
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    clean = slices["clean"]
    assert clean["accepted"] == 0
    assert clean["coverage"] == 0.0
    assert clean["risk_ucb95"] is None
    assert clean["observed_risk"] is None
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    g1 = outcome["gates"]["G1_clean_safety"]
    assert g1["passed"] is False
    assert "undefined" in g1["reason"]
    assert outcome["passed"] is False


def test_g2_fails_when_a_shift_slice_has_high_risk():
    cases = (
        _cases("clean", 300, 0.95)
        + _cases("long_suffix", 96, 0.95, wrong=4)
        + _cases("long_suffix", 4, 0.5)
    )
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    shifted = slices["long_suffix"]
    assert shifted["accepted"] == 96
    assert shifted["coverage"] == 0.96
    assert shifted["risk_ucb95"] > EPSILON
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    g2 = outcome["gates"]["G2_shift_safety"]
    assert outcome["gates"]["G1_clean_safety"]["passed"] is True
    assert g2["passed"] is False
    assert g2["failing_slices"] == ["long_suffix"]
    assert g2["per_slice"]["long_suffix"]["passed"] is False
    # coverage retention still holds, and no slice collapsed
    assert outcome["gates"]["G3_coverage_retention"]["passed"] is True
    assert outcome["failed_gates"] == ["G2_shift_safety"]


def test_g3_fails_when_shift_coverage_drops_too_far():
    cases = (
        _cases("clean", 300, 0.95)
        + _cases("short_prefix", 250, 0.95)
        + _cases("short_prefix", 50, 0.5, wrong=50)
    )
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    shifted = slices["short_prefix"]
    assert shifted["coverage"] == 250 / 300
    assert shifted["risk_ucb95"] <= EPSILON
    assert shifted["accuracy"] == 250 / 300  # overall, including deferred rows
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    g3 = outcome["gates"]["G3_coverage_retention"]
    assert outcome["gates"]["G2_shift_safety"]["passed"] is True
    assert g3["passed"] is False
    assert g3["clean_coverage"] == 1.0
    assert g3["required_coverage"] == pytest.approx(RETENTION_MIN)
    assert g3["failing_slices"] == ["short_prefix"]
    assert outcome["failed_gates"] == ["G3_coverage_retention"]


def test_g4_fails_when_a_slice_collapses_below_the_floor():
    cases = (
        _cases("clean", 300, 0.95)
        + _cases("long_suffix", 6, 0.95)
        + _cases("long_suffix", 294, 0.5)
    )
    slices = evaluate_all_slices(cases, _policy(), EPSILON)
    assert slices["long_suffix"]["coverage"] == 6 / 300
    outcome = evaluate_gates(slices, EPSILON, RETENTION_MIN, FLOOR)
    g4 = outcome["gates"]["G4_collapse_floor"]
    assert g4["passed"] is False
    assert g4["violations"] == [
        {"slice_tag": "long_suffix", "metric": "coverage", "value": 6 / 300, "floor": 0.05}
    ]
    assert outcome["failed_gates"] == [
        "G2_shift_safety", "G3_coverage_retention", "G4_collapse_floor",
    ]


def test_g4_catches_metric_floor_on_precomputed_results():
    slice_results = {
        "clean": _slice("clean", risk_ucb95=0.005),
        "meta_suffix": _slice("meta_suffix", risk_ucb95=0.005, accuracy=0.25),
    }
    outcome = evaluate_gates(slice_results, EPSILON, RETENTION_MIN, FLOOR)
    g4 = outcome["gates"]["G4_collapse_floor"]
    assert g4["passed"] is False
    assert g4["violations"] == [
        {"slice_tag": "meta_suffix", "metric": "accuracy", "value": 0.25, "floor": 0.30}
    ]
    assert outcome["gates"]["G1_clean_safety"]["passed"] is True
    assert outcome["gates"]["G2_shift_safety"]["passed"] is True
    assert outcome["gates"]["G3_coverage_retention"]["passed"] is True
    assert outcome["failed_gates"] == ["G4_collapse_floor"]


def test_shift_gates_fail_closed_without_shift_slices():
    outcome = evaluate_gates(
        {"clean": _slice("clean", risk_ucb95=0.005)}, EPSILON, RETENTION_MIN, FLOOR
    )
    assert outcome["gates"]["G1_clean_safety"]["passed"] is True
    assert outcome["gates"]["G2_shift_safety"]["passed"] is False
    assert outcome["gates"]["G3_coverage_retention"]["passed"] is False
    assert outcome["gates"]["G4_collapse_floor"]["passed"] is True
    assert outcome["failed_gates"] == ["G2_shift_safety", "G3_coverage_retention"]
    # an empty result set cannot support any claim
    empty = evaluate_gates({}, EPSILON, RETENTION_MIN, FLOOR)
    assert empty["gates"]["G1_clean_safety"]["passed"] is False
    assert "no clean slice" in empty["gates"]["G1_clean_safety"]["reason"]


def test_score_kind_and_operating_threshold_are_read_from_policy():
    policy = {
        "format": "veya-selective-policy-v1",
        "score": "margin",
        "operating_epsilon": "0.05",
        "thresholds": {"0.02": 0.9, "0.05": 0.4},
    }
    cases = [
        _case(False, 0.75, "clean"),  # margin 0.5 -> accepted, wrong
        _case(True, 0.60, "clean"),   # margin 0.2 -> deferred
    ]
    result = evaluate_slice(cases, policy, 0.05)
    assert result["slice_tag"] == "clean"  # inferred from the cases
    assert result["threshold"] == 0.4  # the 0.05 row, not the 0.02 row
    assert result["accepted"] == 1 and result["deferred"] == 1
    assert result["observed_risk"] == 1.0
    assert result["risk_ucb95"] > 0.05
    assert result["supports_epsilon_claim"] is False
    assert result["accuracy"] == 0.5
    assert result["ece"] == pytest.approx(0.575, abs=1e-9)  # max-prob confidence


def test_missing_operating_threshold_raises():
    policy = {
        "score": "max_probability",
        "operating_epsilon": "0.03",
        "thresholds": {"0.02": 0.9},
    }
    with pytest.raises(ValueError, match="missing its operating threshold"):
        evaluate_slice(_cases("clean", 4, 0.95), policy, EPSILON)


def test_gate_report_structure_and_pass():
    cases = (
        _cases("clean", 250, 0.95)
        + _cases("clean", 50, 0.55, wrong=50)
        + _cases("long_suffix", 245, 0.95)
        + _cases("long_suffix", 55, 0.55, wrong=55)
        + _cases("meta_suffix", 240, 0.95)
        + _cases("meta_suffix", 60, 0.55, wrong=60)
    )
    report = gate_report(cases, _policy(), EPSILON, RETENTION_MIN, FLOOR)
    assert report["format"] == "veya-selective-gates-v1"
    assert report["n_cases"] == len(cases)
    assert report["policy"] == {
        "score": "max_probability",
        "operating_epsilon": "0.02",
        "threshold": 0.9,
    }
    assert report["protocol"] == {
        "epsilon": EPSILON,
        "coverage_retention_min": RETENTION_MIN,
        "collapse_floor": FLOOR,
    }
    assert set(report["slices"]) == {"clean", "long_suffix", "meta_suffix"}
    assert report["slices"]["long_suffix"]["slice_tag"] == "long_suffix"
    assert report["slices"]["clean"]["accuracy"] == pytest.approx(250 / 300)
    assert set(report["gates"]) == {
        "G1_clean_safety", "G2_shift_safety", "G3_coverage_retention", "G4_collapse_floor",
    }
    for gate in report["gates"].values():
        assert isinstance(gate["passed"], bool)
        assert isinstance(gate["reason"], str)
    assert report["passed"] is True
    assert report["failed_gates"] == []
    assert json.loads(json.dumps(report)) == report
