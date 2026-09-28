"""Aggregate contextual feasibility and real TRAIN regression pairs for Phase 12."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict

from research.veyabench.phase12_context import load_examples, render_turns, turns_for
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash


REPORT = ROOT / "artifacts" / "experiments" / "phase12" / "context_audit.json"
REGRESSION = ROOT / "artifacts" / "experiments" / "phase12" / "context_regression.json"


def norm(value: str) -> str:
    return " ".join(value.casefold().split())


def ceiling(rows: list[dict], mode: str) -> dict:
    groups = defaultdict(Counter)
    for row in rows:
        key = (row["service"], norm(render_turns(turns_for(row, mode))))
        groups[key][row["gold"]] += 1
    n = len(rows)
    unavoidable = sum(sum(labels.values()) - max(labels.values()) for labels in groups.values())
    repeated = sum(count - 1 for labels in groups.values() for count in labels.values())
    return {
        "n": n,
        "repeated_same_label_input_rows": repeated,
        "conflicting_input_groups": sum(len(labels) > 1 for labels in groups.values()),
        "minimum_exact_input_conflict_errors": unavoidable,
        "in_sample_deterministic_ceiling": round(1 - unavoidable / n, 6) if n else None,
    }


def summarize(rows: list[dict]) -> dict:
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["service"]][row["gold"]] += 1
    return {
        "n": len(rows),
        "none_rows": sum(row["gold"] == "NONE" for row in rows),
        "initial_rows": sum(row["initial_user_turn"] for row in rows),
        "service_transitions": sum(row["service_transition"] for row in rows),
        "by_service": {
            service: {"rows": sum(labels.values()), "labels": dict(sorted(labels.items()))}
            for service, labels in sorted(counts.items())
        },
        "exact_input_ceiling": {
            mode: ceiling(rows, mode)
            for mode in ("utterance_only", "previous_turn", "full_context")
        },
    }


def regression_pairs(rows: list[dict], limit: int = 32) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["gold"] != "NONE":
            groups[(row["service"], row["current"])].append(row)
    pairs = []
    for (service, utterance), variants in sorted(groups.items()):
        by_gold = defaultdict(list)
        for row in variants:
            by_gold[row["gold"]].append(row)
        if len(by_gold) < 2:
            continue
        labels = sorted(by_gold)
        left = min(by_gold[labels[0]], key=lambda row: row["id"])
        right = min(by_gold[labels[1]], key=lambda row: row["id"])
        left_context = render_turns(turns_for(left, "previous_turn"))
        right_context = render_turns(turns_for(right, "previous_turn"))
        if norm(left_context) == norm(right_context):
            continue
        pairs.append({
            "service": service,
            "current_utterance": left["current"],
            "current_utterance_sha256": hashlib.sha256(utterance.encode()).hexdigest(),
            "left_id": left["id"],
            "left_gold": left["gold"],
            "left_previous_turn_input": left_context,
            "left_previous_turn_sha256": hashlib.sha256(left_context.encode()).hexdigest(),
            "right_id": right["id"],
            "right_gold": right["gold"],
            "right_previous_turn_input": right_context,
            "right_previous_turn_sha256": hashlib.sha256(right_context.encode()).hexdigest(),
        })
    ranked = sorted(pairs, key=lambda pair: hashlib.sha256(
        (pair["service"] + pair["current_utterance_sha256"]).encode()
    ).hexdigest())
    return ranked[:limit]


def schema_quality(source_hashes: dict[str, str]) -> dict:
    schema = {}
    for split in ("train", "dev"):
        path = SOURCE / split / "schema.json"
        if file_hash(path) != source_hashes[f"{split}/schema.json"]:
            raise ValueError(f"{split} schema changed")
        schema[split] = {item["service_name"]: item for item in json.loads(path.read_text())}
    common = set(schema["train"]) & set(schema["dev"])
    changed = sorted(name for name in common if schema["train"][name] != schema["dev"][name])
    near_pairs = {}
    for split, services in schema.items():
        pairs = []
        for service, definition in services.items():
            for left, right in itertools.combinations(definition["intents"], 2):
                a = set(re.findall(r"[a-z]+", left["description"].casefold()))
                b = set(re.findall(r"[a-z]+", right["description"].casefold()))
                overlap = len(a & b) / len(a | b)
                if overlap >= 0.5:
                    pairs.append({
                        "service": service,
                        "left": left["name"], "right": right["name"],
                        "token_jaccard": round(overlap, 3),
                    })
        near_pairs[split] = sorted(pairs, key=lambda pair: (
            pair["service"], pair["left"], pair["right"]
        ))
    return {
        "shared_service_definitions_train_dev": len(common),
        "changed_shared_definitions_train_dev": changed,
        "lexically_similar_intent_pairs_jaccard_ge_0_5": near_pairs,
        "caveat": "Lexical similarity flags pairs for review; it does not measure expert label ambiguity.",
    }


def main() -> None:
    inventory = json.loads(OUTPUT.read_text())
    source_hashes = inventory["source_files_sha256"]
    split = inventory["phase12_split"]
    train = load_examples("train", set(split["train"]), source_hashes)
    dev = load_examples("dev", set(split["validation"]), source_hashes)
    pairs = regression_pairs(train)
    if len(pairs) < 16:
        raise ValueError("too few real context regression pairs")
    report = {
        "format": "veya-phase12-context-audit-v1",
        "inventory_sha256": file_hash(OUTPUT),
        "source_commit": inventory["commit"],
        "train": summarize(train),
        "development": summarize(dev),
        "schema_quality": schema_quality(source_hashes),
        "regression_pair_count": len(pairs),
        "test_dialogues_opened": False,
        "caveat": "Exact-input collision ceiling is an in-sample feasibility bound, not a model score or human agreement estimate.",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REGRESSION.write_text(json.dumps({
        "format": "veya-phase12-context-regression-v1",
        "source_partition": "official SGD train only",
        "source_commit": inventory["commit"],
        "pairs": pairs,
        "test_dialogues_opened": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "train": {"n": len(train), "ceiling": report["train"]["exact_input_ceiling"]},
        "development": {"n": len(dev), "ceiling": report["development"]["exact_input_ceiling"]},
        "regression_pair_count": len(pairs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
