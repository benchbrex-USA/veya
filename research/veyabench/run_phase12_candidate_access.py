"""Measure exact candidate access for the frozen SGD development workflow."""
from __future__ import annotations

import json
from collections import Counter

from research.veyabench.phase12_context import load_examples
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash


RESULT = ROOT / "artifacts" / "experiments" / "phase12" / "candidate_access.json"


def main() -> None:
    inventory = json.loads(OUTPUT.read_text())
    source_hashes = inventory["source_files_sha256"]
    path = SOURCE / "dev" / "schema.json"
    if file_hash(path) != source_hashes["dev/schema.json"]:
        raise ValueError("official dev schema changed")
    schemas = {item["service_name"]: item for item in json.loads(path.read_text())}
    rows = load_examples("dev", set(inventory["phase12_split"]["validation"]), source_hashes)
    active = [row for row in rows if row["gold"] != "NONE"]
    candidate_counts = Counter()
    accessible = {k: 0 for k in (4, 8, 16, 32)}
    for row in active:
        names = [intent["name"] for intent in schemas[row["service"]]["intents"]]
        candidate_counts[len(names)] += 1
        if row["gold"] not in names:
            raise ValueError(f"gold intent absent from official schema: {row['id']}")
        for k in accessible:
            accessible[k] += int(len(names) <= k)
    report = {
        "format": "veya-phase12-candidate-access-v1",
        "workflow": "schema_guided_service_intent_routing",
        "source_commit": inventory["commit"],
        "inventory_sha256": file_hash(OUTPUT),
        "n_all_user_frames": len(rows),
        "n_active_intent": len(active),
        "n_none": len(rows) - len(active),
        "candidate_count_distribution_active": dict(sorted(candidate_counts.items())),
        "recall_at_1": None,
        "recall_at_1_note": "No retrieval ranker is used; top-1 is the scorer's accuracy, not candidate access.",
        "recall_at_4": accessible[4] / len(active),
        "recall_at_8": accessible[8] / len(active),
        "recall_at_16": accessible[16] / len(active),
        "recall_at_32": accessible[32] / len(active),
        "recall_at_all": 1.0,
        "selected_architecture": "score every official intent; model NONE as a separate no-route outcome",
        "test_dialogues_opened": False,
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
