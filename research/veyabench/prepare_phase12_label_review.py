"""Prepare a blind, development-only SGD intent-label review queue."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from research.veyabench.phase12_context import load_examples, render_turns, turns_for
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash


BASE = ROOT / "artifacts" / "experiments" / "phase12"
DETAILS = BASE / "sparse_scorer_previous_turn_158042_3_details.jsonl"
BLIND = BASE / "label_review_blind.jsonl"
KEY = BASE / "label_review_key.jsonl"
MANIFEST = BASE / "label_review_manifest.json"


def stable_rank(item: dict) -> str:
    return hashlib.sha256(item["id"].encode("utf-8")).hexdigest()


def select(details: list[dict]) -> list[tuple[dict, str]]:
    by_service: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        by_service[item["service"]].append(item)
    selected = []
    for service, items in sorted(by_service.items()):
        errors = [item for item in items if not item["correct"]]
        controls = [item for item in items if item["correct"]]
        high_confidence = sorted(errors, key=lambda item: (-item["confidence"], stable_rank(item)))[:8]
        high_ids = {item["id"] for item in high_confidence}
        diverse = sorted((item for item in errors if item["id"] not in high_ids), key=stable_rank)[:8]
        controls = sorted(controls, key=stable_rank)[:5]
        selected.extend((item, "high_confidence_error") for item in high_confidence)
        selected.extend((item, "sampled_error") for item in diverse)
        selected.extend((item, "blind_control") for item in controls)
        if len(high_confidence) < 8 or len(diverse) < 8 or len(controls) < 5:
            raise ValueError(f"insufficient review examples for {service}")
    return sorted(selected, key=lambda pair: stable_rank(pair[0]))


def main() -> None:
    inventory = json.loads(OUTPUT.read_text())
    if inventory["phase12_split"]["test_dialogues_parsed"]:
        raise ValueError("sealed test was opened")
    hashes = inventory["source_files_sha256"]
    rows = load_examples("dev", set(inventory["phase12_split"]["validation"]), hashes)
    by_id = {row["id"]: row for row in rows}
    details = [json.loads(line) for line in DETAILS.read_text().splitlines()]
    if len(details) != len(rows) or {item["id"] for item in details} != set(by_id):
        raise ValueError("scorer details are not matched to the development population")
    schema_path = SOURCE / "dev" / "schema.json"
    if file_hash(schema_path) != hashes["dev/schema.json"]:
        raise ValueError("official development schema changed")
    schema = {item["service_name"]: item for item in json.loads(schema_path.read_text())}
    chosen = select(details)
    with BLIND.open("w", encoding="utf-8") as blind_file, KEY.open("w", encoding="utf-8") as key_file:
        for item, stratum in chosen:
            row = by_id[item["id"]]
            service = schema[row["service"]]
            blind = {
                "id": row["id"],
                "service_name": row["service"],
                "service_description": service["description"],
                "dialogue_through_current_user": render_turns(turns_for(row, "full_context")),
                "candidates": [
                    {"name": intent["name"], "description": intent["description"]}
                    for intent in service["intents"]
                ],
                "no_route_option": "NONE",
                "review_options": [intent["name"] for intent in service["intents"]]
                + ["NONE", "AMBIGUOUS", "INSUFFICIENT_INFORMATION"],
            }
            key = {
                "id": row["id"], "service_name": row["service"],
                "official_gold": row["gold"], "model_prediction": item["prediction"],
                "model_confidence_uncalibrated": item["confidence"], "stratum": stratum,
            }
            blind_file.write(json.dumps(blind, sort_keys=True) + "\n")
            key_file.write(json.dumps(key, sort_keys=True) + "\n")
    report = {
        "format": "veya-phase12-blind-label-review-v1",
        "source_commit": inventory["commit"],
        "inventory_sha256": file_hash(OUTPUT),
        "scorer_details_sha256": file_hash(DETAILS),
        "blind_file_sha256": file_hash(BLIND),
        "key_file_sha256": file_hash(KEY),
        "n": len(chosen),
        "per_service": 21,
        "high_confidence_errors_per_service": 8,
        "sampled_errors_per_service": 8,
        "blind_controls_per_service": 5,
        "test_dialogues_opened": False,
        "instructions": "Give annotators only the blind file. Obtain two independent intent/none/ambiguous judgments with evidence from the displayed dialogue; adjudicate disagreements before opening the key. The service is given, not inferred.",
    }
    MANIFEST.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n": report["n"], "services": len({item["service"] for item, _ in chosen}),
                      "test_dialogues_opened": False}))


if __name__ == "__main__":
    main()
