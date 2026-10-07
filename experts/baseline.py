"""Read the audited baseline. This does not rescore the sealed set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experts.sweep import ACT_HEAD, REFERENCE

# Cited from docs/phase10/final_report.md. Not remeasured here.
PHASE10 = {
    "source": "docs/phase10/final_report.md",
    "status": "previously_measured_not_rerun",
    "decision": "KEEP AS RESEARCH",
    "clinc_train_recall_at_8": {"current": 0.3553, "phase10": 0.4310},
    "massive_blind_recall_at_8": {"current": 0.2851, "phase10": 0.3497},
    "massive_blind_end_to_end": {"current": 0.2293, "phase10": 0.2734},
    "suffix_recall_at_8": {"clean": 0.4267, "short_irrelevant": 0.1400, "long_irrelevant": 0.0533},
    "k": 8,
    "production_retriever_changed": False,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def audited_choice_baseline(root: Path | None = None) -> dict[str, Any]:
    """Load sealed aggregate metrics. Per-example rows are not in the artifact."""
    path = (root or repo_root()) / "artifacts/release/blind_clinc150_val_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    choice = payload["choice"]
    return {
        "source": "artifacts/release/blind_clinc150_val_result.json",
        "status": "previously_measured_not_rerun",
        "dataset": payload.get("dataset"),
        "n": payload.get("n"),
        "candidate_count": payload.get("candidate_count"),
        "recall_at_8": choice["retrieval_recall_at_8"]["rate"],
        "conditional_accuracy_given_hit": choice["conditional_accuracy_given_hit"]["rate"],
        "end_to_end_accuracy": choice["end_to_end_accuracy"]["rate"],
        "conditional_nll": choice["conditional_nll_on_hits"],
        "conditional_brier": choice["conditional_brier_on_hits"],
        "end_to_end_nll": choice["honest_end_to_end_nll_with_miss_floor_1e-12"],
        "end_to_end_brier": choice["honest_end_to_end_brier_over_150_classes"],
        "ece": choice["ece_end_to_end"],
        "retrieval_misses": choice["retrieval_misses"],
        "reranker_misses": choice["reranker_misses_with_target_present"],
        "recall_at_1": None,
        "recall_at_4": None,
        "recall_at_16": None,
        "missing_reason": "the sealed aggregate file does not publish Recall@1, Recall@4, or Recall@16",
    }


def snapshot(root: Path | None = None) -> dict[str, Any]:
    return {
        "audited_choice": audited_choice_baseline(root),
        "phase10": PHASE10,
        "trainer_reference": REFERENCE,
        "act_head": ACT_HEAD,
        "optimal": False,
        "production_ready": False,
    }
