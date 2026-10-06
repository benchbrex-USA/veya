"""Offline comparison of Veya, experts, and cascades.

Rows are caller-supplied records. This module does not call a model.
"""

from __future__ import annotations

from typing import Any


def score_system(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "coverage": None, "latency_ms_mean": None, "cost_mean": None}
    correct = [row for row in rows if row.get("correct") is True]
    answered = [row for row in rows if row.get("deferred") is not True]
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    costs = [float(row["cost"]) for row in rows if isinstance(row.get("cost"), (int, float))]
    return {
        "n": len(rows),
        "accuracy": len(correct) / len(rows),
        "coverage": len(answered) / len(rows),
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "cost_mean": (sum(costs) / len(costs)) if costs else None,
        "external_calls": sum(int(row.get("external_calls") or 0) for row in rows),
    }


def by_system(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("system")
        if isinstance(name, str):
            grouped.setdefault(name, []).append(row)
    return {name: score_system(items) for name, items in grouped.items()}


def pareto(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Systems not dominated on accuracy, latency, and cost. Higher accuracy is better."""
    names = []
    items = list(summaries.items())
    for name, stats in items:
        dominated = False
        for other_name, other in items:
            if other_name == name:
                continue
            if _dominates(other, stats):
                dominated = True
                break
        if not dominated:
            names.append(name)
    return names


def disagreement_priority(flags: list[str]) -> dict[str, str]:
    return {
        "review_priority": "high" if flags else "normal",
        "automation": "defer",
    }


def _dominates(challenger: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = ("accuracy", "latency_ms_mean", "cost_mean")
    better = False
    for key in keys:
        left = challenger.get(key)
        right = current.get(key)
        if left is None or right is None:
            return False
        if key == "accuracy":
            if left < right:
                return False
            better = better or left > right
        else:
            if left > right:
                return False
            better = better or left < right
    return better
