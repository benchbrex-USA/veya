"""Offline comparison harness for quantum-calibration research rows.

The functions score records the caller already has. They do not call a model
and they do not fit a threshold on the held-out group.
"""

from __future__ import annotations

import math
from typing import Any

MODES = (
    "structured_veya",
    "ising_alone",
    "ising_veya",
    "ising_icl_veya",
    "ising_critic_veya",
)


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = _accuracy(rows, "status")
    diagnosis = _accuracy(rows, "diagnosis")
    action = _accuracy(rows, "next_action")
    brier, ece, n_prob = _probability(rows)
    abs_error, rel_error, n_param = _parameters(rows)
    deferred = [row for row in rows if row.get("deferred") is True]
    executed = [row for row in rows if row.get("policy_decision") == "execute"]
    false_auto = [
        row for row in executed if row.get("gold_execute") is False or row.get("correct") is False
    ]
    covered = [row for row in rows if row.get("deferred") is False]
    selective_errors = [row for row in covered if row.get("correct") is False]
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    calls = [int(row["external_calls"]) for row in rows if isinstance(row.get("external_calls"), int)]
    costs = [float(row["cost"]) for row in rows if isinstance(row.get("cost"), (int, float))]
    return {
        "n": len(rows),
        "experiment_status_accuracy": status,
        "diagnosis_accuracy": diagnosis,
        "next_action_accuracy": action,
        "fit_reliability_brier": brier,
        "fit_reliability_ece": ece,
        "probability_rows": n_prob,
        "parameter_absolute_error": abs_error,
        "parameter_relative_error": rel_error,
        "parameter_rows": n_param,
        "defer_rate": (len(deferred) / len(rows)) if rows else None,
        "coverage": (len(covered) / len(rows)) if rows else None,
        "selective_risk": (len(selective_errors) / len(covered)) if covered else None,
        "false_auto_execution_rate": (len(false_auto) / len(executed)) if executed else 0.0,
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "external_model_calls": sum(calls) if calls else 0,
        "cost": sum(costs) if costs else None,
    }


def summarize_modes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for row in rows:
        mode = row.get("mode")
        if mode in grouped:
            grouped[str(mode)].append(row)
    return {mode: score_rows(items) for mode, items in grouped.items()}


def holdout_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Score each group only on rows from that group. The other rows are withheld."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row.get(key)
        if not isinstance(label, str):
            continue
        groups.setdefault(label, []).append(row)
    folds = []
    for name, test_rows in sorted(groups.items()):
        train_rows = [row for group, items in groups.items() if group != name for row in items]
        test_ids = {id(row) for row in test_rows}
        if any(id(row) in test_ids for row in train_rows):
            raise RuntimeError("holdout overlap")
        folds.append(
            {
                "held_out": name,
                "group_key": key,
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "metrics": score_rows(test_rows),
                "trained_on_holdout": False,
            }
        )
    return folds


def _accuracy(rows: list[dict[str, Any]], field: str) -> float | None:
    paired = [
        row for row in rows if isinstance(row.get("gold_" + field), str) and isinstance(row.get(field), str)
    ]
    if not paired:
        return None
    hits = sum(1 for row in paired if row[field] == row["gold_" + field])
    return hits / len(paired)


def _probability(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, int]:
    paired = []
    for row in rows:
        prob = row.get("fit_reliable_probability")
        gold = row.get("gold_fit_reliable")
        if isinstance(prob, (int, float)) and not isinstance(prob, bool) and isinstance(gold, bool):
            if 0.0 <= float(prob) <= 1.0:
                paired.append((float(prob), 1.0 if gold else 0.0))
    if not paired:
        return None, None, 0
    brier = sum((prob - gold) ** 2 for prob, gold in paired) / len(paired)
    return brier, _ece(paired), len(paired)


def _ece(paired: list[tuple[float, float]], bins: int = 10) -> float:
    total = len(paired)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [item for item in paired if (low <= item[0] < high) or (index == bins - 1 and item[0] == 1.0)]
        if not bucket:
            continue
        confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - confidence)
    return error


def _parameters(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, int]:
    absolute = []
    relative = []
    for row in rows:
        gold = row.get("gold_parameter")
        pred = row.get("parameter")
        if isinstance(gold, bool) or isinstance(pred, bool):
            continue
        if not isinstance(gold, (int, float)) or not isinstance(pred, (int, float)):
            continue
        if not math.isfinite(float(gold)) or not math.isfinite(float(pred)):
            continue
        absolute.append(abs(float(pred) - float(gold)))
        if float(gold) != 0.0:
            relative.append(abs(float(pred) - float(gold)) / abs(float(gold)))
    if not absolute:
        return None, None, 0
    abs_mean = sum(absolute) / len(absolute)
    rel_mean = (sum(relative) / len(relative)) if relative else None
    return abs_mean, rel_mean, len(absolute)
