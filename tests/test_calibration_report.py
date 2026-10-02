"""Synthetic verification for research/veyabench/calibration_report.py.

The calibration report is descriptive: it bins the confidence values it is handed,
so synthetic rows with a known number of errors pin the structure and arithmetic
without a checkpoint.

Run: python -m pytest tests/test_calibration_report.py -xvs
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from research.veyabench.calibration_report import (  # noqa: E402
    COVERAGE_DECILES,
    ERROR_HISTOGRAM_BINS,
    RELIABILITY_BINS,
    calibration_report,
    calibration_summary,
)

SEED = 13
N = 200
K = 5
THRESHOLD = 0.5

SLICE_FIELDS = {
    "n",
    "temperature",
    "ece",
    "brier",
    "reliability_diagram",
    "risk_coverage_deciles",
    "accepted_vs_deferred",
    "error_confidence_histogram",
    "error_count",
}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _make_slice(rng, accuracy, gold_boost, wrong_boost, temperature, noise=0.3):
    """Rows where the gold class wins (high confidence) mixed with rows where a rival wins."""
    labels = rng.integers(0, K, size=N)
    logits = rng.normal(0.0, noise, size=(N, K))
    rows = np.arange(N)
    picked = rng.random(N) < accuracy
    logits[rows[picked], labels[picked]] += gold_boost
    missed = rows[~picked]
    rivals = (labels[missed] + 1 + rng.integers(0, K - 1, size=len(missed))) % K
    logits[missed, rivals] += wrong_boost
    confidence = _softmax(logits / temperature).max(axis=1)
    correct = (logits.argmax(axis=1) == labels).astype(np.float64)
    return {
        "logits": logits,
        "labels": labels,
        "confidence": confidence,
        "correct": correct,
        "temperature": temperature,
        "scores": confidence,
    }


def _two_slices():
    rng = np.random.default_rng(SEED)
    return {
        "clean": _make_slice(rng, accuracy=0.8, gold_boost=3.0, wrong_boost=1.0, temperature=1.0),
        "shifted": _make_slice(rng, accuracy=0.5, gold_boost=1.2, wrong_boost=1.4, temperature=2.0),
    }


def test_report_structure_has_all_fields():
    slices = _two_slices()
    reports = calibration_report(slices, threshold=THRESHOLD)
    assert set(reports) == set(slices)
    for tag, report in reports.items():
        assert SLICE_FIELDS <= set(report), tag
        assert report["n"] == N
        assert report["temperature"] == slices[tag]["temperature"]
        assert len(report["reliability_diagram"]) == RELIABILITY_BINS
        assert len(report["error_confidence_histogram"]) == ERROR_HISTOGRAM_BINS
        accepted = report["accepted_vs_deferred"]
        assert accepted["accepted"] + accepted["deferred"] == N
        for row in report["reliability_diagram"]:
            assert {"bin_low", "bin_high", "bin_count", "mean_confidence", "mean_accuracy"} <= set(row)
        for row in report["error_confidence_histogram"]:
            assert {"bin_low", "bin_high", "count"} <= set(row)


def test_ece_and_brier_are_finite_floats():
    reports = calibration_report(_two_slices(), threshold=THRESHOLD)
    for report in reports.values():
        assert isinstance(report["ece"], float) and np.isfinite(report["ece"]) and report["ece"] >= 0.0
        assert isinstance(report["brier"], float) and np.isfinite(report["brier"]) and report["brier"] >= 0.0


def test_reliability_diagram_sums_to_n():
    reports = calibration_report(_two_slices(), threshold=THRESHOLD)
    for report in reports.values():
        bins = report["reliability_diagram"]
        assert sum(row["bin_count"] for row in bins) == report["n"] == N
        for i, row in enumerate(bins):
            assert row["bin_low"] == pytest.approx(i / RELIABILITY_BINS)
            assert row["bin_high"] == pytest.approx((i + 1) / RELIABILITY_BINS)
            if row["bin_count"]:
                assert row["bin_low"] <= row["mean_confidence"] <= row["bin_high"]
                assert 0.0 <= row["mean_accuracy"] <= 1.0
            else:
                assert row["mean_confidence"] is None and row["mean_accuracy"] is None


def test_error_histogram_counts_sum_to_errors():
    slices = _two_slices()
    reports = calibration_report(slices, threshold=THRESHOLD)
    for tag, report in reports.items():
        expected = int((slices[tag]["correct"] == 0).sum())
        assert expected > 0, tag
        assert report["error_count"] == expected
        assert sum(row["count"] for row in report["error_confidence_histogram"]) == expected


def test_risk_coverage_deciles():
    reports = calibration_report(_two_slices(), threshold=THRESHOLD)
    for report in reports.values():
        deciles = report["risk_coverage_deciles"]
        assert len(deciles) == COVERAGE_DECILES
        coverages = [row["coverage"] for row in deciles]
        assert coverages == sorted(coverages)
        assert deciles[0]["coverage"] == pytest.approx(0.1)
        assert coverages[-1] == pytest.approx(1.0)
        for row in deciles:
            assert {"accepted", "coverage", "incorrect", "risk"} <= set(row)


def test_accepted_vs_deferred_accuracy():
    reports = calibration_report(_two_slices(), threshold=THRESHOLD)
    clean = reports["clean"]["accepted_vs_deferred"]
    assert clean["accepted"] > 0 and clean["deferred"] > 0
    assert 0.0 <= clean["accepted_accuracy"] <= 1.0
    assert 0.0 <= clean["deferred_accuracy"] <= 1.0
    for report in reports.values():
        accepted = report["accepted_vs_deferred"]
        assert accepted["threshold"] == THRESHOLD
        assert accepted["accepted"] + accepted["deferred"] == report["n"]
        assert accepted["accepted_wrong"] + accepted["deferred_wrong"] == report["error_count"]
        # the shifted slice's softened confidence stays below the threshold: nothing is
        # accepted, and an empty side must carry None rather than a fabricated accuracy
        assert (accepted["accepted_accuracy"] is None) == (accepted["accepted"] == 0)
        assert (accepted["deferred_accuracy"] is None) == (accepted["deferred"] == 0)


def test_summary_two_slices():
    slices = _two_slices()
    reports = calibration_report(slices, threshold=THRESHOLD)
    document = calibration_summary(reports)
    assert document["slices"] == reports
    top = document["summary"]
    assert top["n_slices"] == 2

    def worst(field):
        values = {tag: report[field] for tag, report in reports.items()}
        tag = max(values, key=values.get)
        return {"slice": tag, "value": values[tag]}

    assert top["worst_ece"] == worst("ece")
    assert top["worst_brier"] == worst("brier")

    # the frozen threshold must leave the clean slice's accepted side measurably purer
    clean = reports["clean"]["accepted_vs_deferred"]
    assert clean["accepted_accuracy"] > clean["deferred_accuracy"]
    assert top["clean_slice"] == "clean"
    assert top["separation_by_slice"]["clean"] is True
    assert top["confidence_separates_accepted_from_deferred"] is True


def test_missing_threshold_defers_everything():
    slices = _two_slices()
    reports = calibration_report(slices)
    document = calibration_summary(reports)
    assert document["summary"]["confidence_separates_accepted_from_deferred"] is None
    for tag, report in reports.items():
        accepted = report["accepted_vs_deferred"]
        assert accepted["threshold"] is None
        assert accepted["accepted"] == 0 and accepted["deferred"] == report["n"]
        assert accepted["accepted_accuracy"] is None
        expected = float(np.asarray(slices[tag]["correct"], dtype=np.float64).mean())
        assert accepted["deferred_accuracy"] == pytest.approx(expected)


def test_slice_threshold_overrides_argument():
    slices = _two_slices()
    slices["clean"] = dict(slices["clean"], threshold=0.9)
    reports = calibration_report(slices, threshold=0.1)
    assert reports["clean"]["accepted_vs_deferred"]["threshold"] == 0.9
    assert reports["shifted"]["accepted_vs_deferred"]["threshold"] == 0.1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
