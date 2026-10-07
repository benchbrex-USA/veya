"""Promotion matrix. Nothing is promoted unless every gate is measured and true."""

from __future__ import annotations

import json
from typing import Any

GATES = (
    "full_tests_pass",
    "static_checks_pass",
    "security_checks_pass",
    "fresh_checkpoint_load_pass",
    "benchmark_reproducible",
    "no_data_leakage",
    "multi_seed_stable",
    "calibration_held_out",
    "routing_quality_measured",
    "latency_measured",
    "cost_measured",
    "robustness_measured",
    "diff_reviewed",
)


def classify(record: dict[str, Any]) -> str:
    if record.get("rejected") is True:
        return "REJECTED"
    if all(record.get(gate) is True for gate in GATES):
        return "PROMOTED"
    if record.get("measured_finalist") is True:
        return "CANDIDATE"
    return "RESEARCH"


def matrix() -> dict[str, Any]:
    """Current evidence only. A measured failure is REJECTED. Nothing is PROMOTED."""
    items = [
        _item("veya_checkpoint", "veya-core", "0.3.5-audit-baseline.1", "previously_measured", "Audited research release. Not re-promoted."),
        _item("retriever", "production-shortlist-k8", "manifest", "previously_measured", "Unchanged production retriever. Schema-held-out clean Recall@8 was 0.349 on naturalized candidates in the robustness run."),
        _item("retriever", "phase10-domain-general-k8", "research", "measured", "Clean Recall@8 0.431. Full-string long-suffix Recall@8 0.077. Not promoted."),
        _item(
            "retriever",
            "query-view-max-fusion",
            "research-eval",
            "measured",
            "Phase-10 projection with deterministic views: long-suffix Recall@8 0.328 and clean Recall@8 unchanged at 0.431. Not promoted; end-to-end scorer and adversarial suffixes were not measured.",
            status="RESEARCH",
            robustness={"long_suffix_recall_at_8": 0.328, "clean_recall_at_8": 0.431},
        ),
        _item(
            "retriever",
            "noise-augmented-projection",
            "one-seed",
            "measured",
            "Train-only suffix augmentation plus one hard-negative refresh. Clean Recall@8 fell to 0.306 and long-suffix Recall@8 stayed at 0.074.",
            status="REJECTED",
            robustness={"clean_recall_at_8": 0.306, "long_suffix_recall_at_8": 0.074},
        ),
        _item("calibration", "packed-and-hierarchical", "manifest", "previously_measured", "Not refit."),
        _item("model_profile", "provider-baselines", "experts-profile-v1", "blocked", "NIM credential was not configured. No live expert benchmark."),
        _item("routing_policy", "experts-router", "hypothesis", "not_tested", "Routes remain capability constraints, not measured winners."),
        _item("action_head", "act_weight", "0.0", "not_tested", "No held-out defer labels were found. act_weight stays 0."),
        _item("training_sweep", "veya-train-config", "reference", "not_tested", "The decision-model learning-rate sweep was not run."),
    ]
    return {
        "format": "veya-promotion-matrix-v1",
        "promoted_count": sum(1 for item in items if item["status"] == "PROMOTED"),
        "items": items,
    }


def _item(
    kind: str,
    name: str,
    version: str,
    measurement: str,
    reason: str,
    status: str | None = None,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "kind": kind,
        "name": name,
        "version": version,
        "measurement": measurement,
        "reason": reason,
        "metrics": None,
        "latency": None,
        "cost": None,
        "robustness": robustness,
        "artifact_sha256": None,
        "rejected": status == "REJECTED",
    }
    record["status"] = status or classify(record)
    return record


def write_matrix(path: str) -> dict[str, Any]:
    payload = matrix()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload
