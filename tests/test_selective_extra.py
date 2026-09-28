"""Regression tests for selective-automation helpers (veya/selective.py).

Complements tests/test_release_bundle.py, which covers select_threshold,
apply_threshold, and apply_report. This file covers entropy, score_probabilities,
wilson_interval, risk_coverage_curve, and load_policy. No checkpoint load.

Run: python tests/test_selective_extra.py
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from veya.selective import (  # noqa: E402
    entropy,
    load_policy,
    risk_coverage_curve,
    score_probabilities,
    wilson_interval,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def test_entropy():
    check("entropy/all-zero -> 0", entropy([0.0, 0.0]) == 0.0)
    check("entropy/empty -> 0", entropy([]) == 0.0)
    check("entropy/uniform pair", abs(entropy([0.5, 0.5]) - math.log(2)) < 1e-9)
    check("entropy/peaked -> 0", entropy([1.0, 0.0]) == 0.0)
    check("entropy/zeros are filtered", abs(entropy([0.5, 0.5, 0.0]) - math.log(2)) < 1e-9)
    check("entropy/uniform four", abs(entropy([0.25] * 4) - math.log(4)) < 1e-9)


def test_score_probabilities():
    check("score/max_probability", abs(score_probabilities([0.7, 0.2, 0.1], "max_probability") - 0.7) < 1e-9)
    check("score/normalises first", abs(score_probabilities([2.0, 1.0], "max_probability") - 2 / 3) < 1e-9)
    check("score/neg_entropy uniform", abs(score_probabilities([0.5, 0.5], "neg_entropy") + math.log(2)) < 1e-9)
    check("score/neg_entropy peaked", score_probabilities([1.0, 0.0], "neg_entropy") == 0.0)
    check("score/margin", abs(score_probabilities([0.5, 0.3, 0.2], "margin") - 0.2) < 1e-9)
    check("score/margin single normalises to 1", score_probabilities([0.9], "margin") == 1.0)
    try:
        score_probabilities([0.5, 0.5], "nonsense")
        check("score/unknown kind raises", False)
    except ValueError:
        check("score/unknown kind raises", True)


def test_wilson_interval():
    check("wilson/accepted 0 -> None", wilson_interval(0, 0) is None)
    check("wilson/negative accepted -> None", wilson_interval(3, -1) is None)
    low, high = wilson_interval(0, 100)
    check("wilson/bounded", 0.0 <= low <= high <= 1.0, "%s %s" % (low, high))
    check("wilson/zero errors stay low", low < 0.001 and high < 0.04, "%s %s" % (low, high))
    low, high = wilson_interval(5, 100)
    check("wilson/covers point estimate", low <= 0.05 <= high, "%s %s" % (low, high))
    low, high = wilson_interval(50, 100)
    check("wilson/z=1.96 default width", 0.36 <= low <= 0.41 and 0.59 <= high <= 0.64,
          "%s %s" % (low, high))


def test_risk_coverage_curve():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    correct = np.array([1, 1, 0, 1])
    curve = risk_coverage_curve(scores, correct)
    check("curve/one row per prefix", len(curve) == 4)
    check("curve/accepted counts up", [row["accepted"] for row in curve] == [1, 2, 3, 4])
    check("curve/coverage monotonic", all(
        curve[i]["coverage"] < curve[i + 1]["coverage"] for i in range(3)))
    check("curve/first row risk", curve[0]["risk"] == 0.0)
    check("curve/third row counts", curve[2]["incorrect"] == 1 and abs(curve[2]["risk"] - 1 / 3) < 1e-9)
    check("curve/full coverage", abs(curve[3]["coverage"] - 1.0) < 1e-9)
    check("curve/full risk", abs(curve[3]["risk"] - 0.25) < 1e-9)
    # ties: equal scores counted after a stable sort, order preserved
    tied = risk_coverage_curve(np.array([0.5, 0.5]), np.array([0, 1]))
    check("curve/ties keep order", tied[0]["incorrect"] == 1 and tied[1]["incorrect"] == 1)


def _write_policy(directory, **overrides):
    policy = {
        "format": "veya-selective-policy-v1",
        "route": "shortlist_k8_packed_rerank",
        "checkpoint_hash": "hash-1",
        "shortlist_sha256": "sha-1",
        "scorer_id": "base_decision_head",
        "chosen_on": "validation",
        "validation_hash": "val-hash-1",
        "score": "max_probability",
        "thresholds": {"0.05": 0.8, "0.02": 0.9},
        "operating_epsilon": "0.05",
    }
    policy.update(overrides)
    path = Path(directory) / "policy.json"
    path.write_text(json.dumps(policy))
    return str(path)


def test_load_policy_accepts_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_policy(tmp)
        policy = load_policy(path, "shortlist_k8_packed_rerank", "hash-1", "sha-1", "base_decision_head")
        check("policy/accepted", policy["score"] == "max_probability")
        check("policy/thresholds kept", policy["thresholds"]["0.02"] == 0.9)


def test_load_policy_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("wrong format", dict(format="other"), "unsupported selective policy format"),
            ("wrong route", dict(route="packed"), "different route"),
            ("wrong checkpoint", dict(checkpoint_hash="other"), "different checkpoint"),
            ("wrong shortlist", dict(shortlist_sha256="other"), "different shortlist"),
            ("wrong scorer", dict(scorer_id="other_scorer"), "different scorer"),
            ("chosen on test", dict(chosen_on="test"), "hashed validation split"),
            ("missing validation hash", dict(validation_hash=""), "hashed validation split"),
            ("bad score kind", dict(score="weird"), "unsupported selective score"),
            ("missing thresholds", dict(thresholds=None), "missing its operating threshold"),
            ("epsilon not in thresholds", dict(thresholds={"0.02": 0.9}), "missing its operating threshold"),
        ]
        for name, overrides, needle in cases:
            path = _write_policy(tmp, **overrides)
            try:
                load_policy(path, "shortlist_k8_packed_rerank", "hash-1", "sha-1", "base_decision_head")
                check("policy/%s refused" % name, False)
            except ValueError as exc:
                check("policy/%s refused" % name, needle in str(exc), str(exc))
        # a non-finite threshold is refused by name
        path = Path(tmp) / "policy_inf.json"
        policy = {
            "format": "veya-selective-policy-v1",
            "route": "shortlist_k8_packed_rerank",
            "checkpoint_hash": "hash-1",
            "shortlist_sha256": "sha-1",
            "scorer_id": "base_decision_head",
            "chosen_on": "validation",
            "validation_hash": "val-hash-1",
            "score": "max_probability",
            "thresholds": {"0.05": float("inf")},
            "operating_epsilon": "0.05",
        }
        path.write_text(json.dumps(policy))
        try:
            load_policy(str(path), "shortlist_k8_packed_rerank", "hash-1", "sha-1", "base_decision_head")
            check("policy/non-finite threshold refused", False)
        except ValueError as exc:
            check("policy/non-finite threshold refused", "not finite" in str(exc), str(exc))


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_entropy()
    test_score_probabilities()
    test_wilson_interval()
    test_risk_coverage_curve()
    test_load_policy_accepts_valid()
    test_load_policy_refusals()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
