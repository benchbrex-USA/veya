"""Per-slice calibration evaluation reports (protocol Section 10).

Descriptive only: every number is computed with the frozen temperature each slice
carries in, and nothing here fits or refits one. Slices are reported independently
so a clean distribution and a shifted one can be compared side by side, and
``calibration_summary`` lifts the worst-case values and the accepted/deferred
separation check into one document. Synthetic rows verify the harness; a real
caller supplies logits, confidence, correctness, and selective scores per slice.
"""
from typing import Dict, Optional

import numpy as np

from veya.calibration import brier
from veya.common import ece_score
from veya.selective import apply_threshold, risk_coverage_curve

RELIABILITY_BINS = 15
ERROR_HISTOGRAM_BINS = 10
COVERAGE_DECILES = 10


def _histogram_edges(bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, bins + 1)


def _reliability_diagram(confidence: np.ndarray, correct: np.ndarray, bins: int = RELIABILITY_BINS) -> list:
    """Equal-width confidence bins. The top bin closes on 1.0 so counts cover every row.

    An empty bin reports None means rather than 0.0: no rows is not the same as zero.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = _histogram_edges(bins)
    rows = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (confidence >= lo) & (confidence <= hi) if i == bins - 1 else (confidence >= lo) & (confidence < hi)
        count = int(sel.sum())
        rows.append({
            "bin_low": lo,
            "bin_high": hi,
            "bin_count": count,
            "mean_confidence": float(confidence[sel].mean()) if count else None,
            "mean_accuracy": float(correct[sel].mean()) if count else None,
        })
    return rows


def _error_confidence_histogram(confidence: np.ndarray, correct: np.ndarray, bins: int = ERROR_HISTOGRAM_BINS) -> list:
    """Confidence histogram over the rows the model got wrong. Counts sum to the error count."""
    errors = np.asarray(confidence, dtype=np.float64)[np.asarray(correct, dtype=np.float64) == 0]
    edges = _histogram_edges(bins)
    rows = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (errors >= lo) & (errors <= hi) if i == bins - 1 else (errors >= lo) & (errors < hi)
        rows.append({"bin_low": lo, "bin_high": hi, "count": int(sel.sum())})
    return rows


def _coverage_deciles(scores: np.ndarray, correct: np.ndarray, points: int = COVERAGE_DECILES) -> list:
    """Every 10% coverage point of the risk-coverage curve.

    Same decile indexing as run_release_eval's ``risk_coverage_deciles``.
    """
    curve = risk_coverage_curve(scores, correct)
    if not curve:
        return []
    deciles = []
    for step in range(1, points + 1):
        index = min(len(curve) - 1, max(0, int(round(step / points * len(curve))) - 1))
        deciles.append(curve[index])
    return deciles


def _accepted_vs_deferred(scores: np.ndarray, correct: np.ndarray, threshold: Optional[float]) -> dict:
    """Accuracy on each side of the selective threshold.

    No threshold accepts nothing (``apply_threshold`` semantics), so the accepted
    side is empty and its accuracy is None rather than fabricated.
    """
    mask = apply_threshold(scores, threshold)
    accepted = int(mask.sum())
    deferred = int(len(mask) - accepted)
    return {
        "threshold": None if threshold is None else float(threshold),
        "accepted": accepted,
        "deferred": deferred,
        "accepted_accuracy": float(correct[mask].mean()) if accepted else None,
        "deferred_accuracy": float(correct[~mask].mean()) if deferred else None,
        "accepted_wrong": int(round(float(np.sum(1.0 - correct[mask])))) if accepted else 0,
        "deferred_wrong": int(round(float(np.sum(1.0 - correct[~mask])))) if deferred else 0,
    }


def calibration_report(results_by_slice: Dict[str, dict], threshold: Optional[float] = None) -> Dict[str, dict]:
    """Build one calibration report per slice tag.

    Each slice payload carries ``logits`` (N x K), ``labels`` (N,), ``confidence``
    (N,) after the frozen temperature, ``correct`` (N,) as 1/0, ``temperature``,
    and ``scores`` (N,) for the risk-coverage curve. A payload may carry its own
    ``threshold``, which wins over the ``threshold`` argument; without either, the
    accepted side is empty.
    """
    reports: Dict[str, dict] = {}
    for tag, payload in results_by_slice.items():
        logits = np.asarray(payload["logits"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        confidence = np.asarray(payload["confidence"], dtype=np.float64)
        correct = np.asarray(payload["correct"], dtype=np.float64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        temperature = float(payload["temperature"])
        n = int(labels.size)
        if logits.ndim != 2 or logits.shape[0] != n:
            raise ValueError("slice %r logits must be N x K with one row per label" % tag)
        if len(confidence) != n or len(correct) != n or len(scores) != n:
            raise ValueError("slice %r arrays must share one length" % tag)
        if n and (labels.min() < 0 or labels.max() >= logits.shape[1]):
            raise ValueError("slice %r labels fall outside the logits' class range" % tag)
        slice_threshold = payload.get("threshold")
        if slice_threshold is None:
            slice_threshold = threshold
        reports[tag] = {
            "n": n,
            "temperature": temperature,
            "ece": ece_score(confidence, correct, bins=RELIABILITY_BINS),
            "brier": brier(logits, labels, temperature),
            "reliability_diagram": _reliability_diagram(confidence, correct),
            "risk_coverage_deciles": _coverage_deciles(scores, correct),
            "accepted_vs_deferred": _accepted_vs_deferred(scores, correct, slice_threshold),
            "error_confidence_histogram": _error_confidence_histogram(confidence, correct),
            "error_count": int(np.count_nonzero(correct == 0)),
        }
    return reports


def calibration_summary(all_slice_reports: Dict[str, dict], clean_slice: str = "clean") -> dict:
    """Wrap per-slice reports with a top-level summary. The input dict is not modified.

    ``worst_ece`` and ``worst_brier`` name the slice holding the largest finite
    value (None when no slice has one). ``confidence_separates_accepted_from_
    deferred`` is the clean slice's "accepted accuracy above deferred accuracy"
    flag, or None when that slice or one of its two accuracies is absent.
    """
    worst_ece: Optional[tuple] = None
    worst_brier: Optional[tuple] = None
    separation: Dict[str, Optional[bool]] = {}
    for tag, report in all_slice_reports.items():
        value = report.get("ece")
        if value is not None and np.isfinite(value) and (worst_ece is None or value > worst_ece[1]):
            worst_ece = (tag, float(value))
        value = report.get("brier")
        if value is not None and np.isfinite(value) and (worst_brier is None or value > worst_brier[1]):
            worst_brier = (tag, float(value))
        comparison = report.get("accepted_vs_deferred") or {}
        accepted, deferred = comparison.get("accepted_accuracy"), comparison.get("deferred_accuracy")
        separation[tag] = None if accepted is None or deferred is None else bool(accepted > deferred)
    return {
        "slices": all_slice_reports,
        "summary": {
            "n_slices": len(all_slice_reports),
            "worst_ece": {"slice": worst_ece[0], "value": worst_ece[1]} if worst_ece
            else {"slice": None, "value": None},
            "worst_brier": {"slice": worst_brier[0], "value": worst_brier[1]} if worst_brier
            else {"slice": None, "value": None},
            "clean_slice": clean_slice,
            "separation_by_slice": separation,
            "confidence_separates_accepted_from_deferred": separation.get(clean_slice),
        },
    }
