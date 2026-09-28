"""Freeze the causal input mode from the three matched SGD development controls."""
from __future__ import annotations

import json

from research.veyabench.phase12_sgd_inventory import ROOT, file_hash


BASE = ROOT / "artifacts" / "experiments" / "phase12"
EARLY = BASE / "context_baselines.json"
FULL = BASE / "context_baseline_selected_rerun.json"
OLD_INVENTORY = BASE / "sgd_inventory_before_selection_rule_correction.json"
CURRENT_INVENTORY = BASE / "sgd_inventory.json"
OUTPUT = BASE / "context_selection.json"


def main() -> None:
    early = json.loads(EARLY.read_text())
    full = json.loads(FULL.read_text())
    old_inventory = json.loads(OLD_INVENTORY.read_text())
    current_inventory = json.loads(CURRENT_INVENTORY.read_text())
    if set(early["modes"]) != {"utterance_only", "previous_turn"}:
        raise ValueError("first controls incomplete")
    if set(full["modes"]) != {"full_context"}:
        raise ValueError("full-history control incomplete")
    if early["inventory_sha256"] != file_hash(OLD_INVENTORY):
        raise ValueError("early inventory hash mismatch")
    if full["inventory_sha256"] != file_hash(CURRENT_INVENTORY):
        raise ValueError("full-history inventory hash mismatch")
    for field in ("source_files_sha256", "schema_intent_count"):
        if old_inventory[field] != current_inventory[field]:
            raise ValueError(f"source lineage changed: {field}")
    for field in ("train", "validation", "sealed_test"):
        if old_inventory["phase12_split"][field] != current_inventory["phase12_split"][field]:
            raise ValueError(f"service split changed: {field}")
    for field in ("source_commit", "release_manifest_sha256", "n_matched", "device", "batch_size"):
        if early[field] != full[field]:
            raise ValueError(f"baseline controls are unmatched: {field}")
    if early["test_dialogues_opened"] or full["test_dialogues_opened"]:
        raise ValueError("sealed test was opened")
    previous = early["modes"]["previous_turn"]
    complete = full["modes"]["full_context"]

    def rank(result: dict) -> tuple[float, int, float]:
        return (
            result["accuracy"],
            -result["context"]["truncated_decisions"],
            -result["context"]["raw_tokens_p95"],
        )

    winner = max((("previous_turn", previous), ("full_context", complete)),
                 key=lambda pair: rank(pair[1]))
    report = {
        "format": "veya-phase12-context-selection-v1",
        "workflow": "schema_guided_service_intent_routing",
        "selection_rule": "Highest raw all-turn development accuracy among the two context-bearing modes; ties favor fewer dropped contexts, then shorter input. Current-only is diagnostic.",
        "chosen_mode": winner[0],
        "chosen_accuracy": winner[1]["accuracy"],
        "n_matched": early["n_matched"],
        "source_commit": early["source_commit"],
        "release_manifest_sha256": early["release_manifest_sha256"],
        "control_artifact_sha256": {
            "utterance_and_previous": file_hash(EARLY),
            "full_context": file_hash(FULL),
        },
        "control_metrics": {
            "utterance_only": early["modes"]["utterance_only"]["accuracy"],
            "previous_turn": previous["accuracy"],
            "full_context": complete["accuracy"],
        },
        "test_dialogues_opened": False,
        "note": "This freezes the input formulation for subsequent scorer work; it does not qualify a finalist or authorize sealed evaluation.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"chosen_mode": winner[0], "accuracy": winner[1]["accuracy"]}))


if __name__ == "__main__":
    main()
