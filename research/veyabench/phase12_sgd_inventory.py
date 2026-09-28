"""Aggregate-only SGD provenance and label audit for Phase 12.

The official test dialogues are hashed but never parsed here. Their services
are identified from the public test schema, before any model selection.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "external" / "sgd"
OUTPUT = ROOT / "artifacts" / "experiments" / "phase12" / "sgd_inventory.json"
EXPECTED_COMMIT = "e852981ae34990f4358979625854259302feaa78"
SOURCE_URL = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schemas(split: str) -> dict[str, dict]:
    path = SOURCE / split / "schema.json"
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    by_name = {item["service_name"]: item for item in items}
    if len(by_name) != len(items):
        raise ValueError(f"duplicate service in {path}")
    for service in items:
        intents = service["intents"]
        names = [intent["name"] for intent in intents]
        if len(names) != len(set(names)) or not service.get("description"):
            raise ValueError(f"invalid service schema {service['service_name']}")
        if any(not intent.get("description") for intent in intents):
            raise ValueError(f"missing official intent definition in {service['service_name']}")
    return by_name


def audit_dialogues(split: str, by_service: dict[str, dict]) -> dict:
    initial = defaultdict(Counter)
    all_active = defaultdict(Counter)
    initial_labels = defaultdict(Counter)
    missing = 0
    unknown = 0
    none_initial = 0
    multi_frame_initial = 0
    dialogue_count = 0
    for raw_path in sorted(glob.glob(str(SOURCE / split / "dialogues_*.json"))):
        with open(raw_path, encoding="utf-8") as handle:
            dialogues = json.load(handle)
        for dialogue in dialogues:
            dialogue_count += 1
            first_user_seen = False
            for turn in dialogue["turns"]:
                if turn["speaker"] != "USER":
                    continue
                is_initial = not first_user_seen
                first_user_seen = True
                if is_initial and len(turn["frames"]) > 1:
                    multi_frame_initial += 1
                utterance = " ".join(turn.get("utterance", "").casefold().split())
                if not utterance:
                    missing += len(turn["frames"])
                    continue
                for frame in turn["frames"]:
                    service = frame["service"]
                    intent = frame.get("state", {}).get("active_intent")
                    if service not in by_service or intent is None:
                        missing += 1
                        continue
                    known = {item["name"] for item in by_service[service]["intents"]}
                    if intent != "NONE" and intent not in known:
                        unknown += 1
                        continue
                    if intent != "NONE":
                        all_active[(service, utterance)][intent] += 1
                    if is_initial:
                        initial[(service, utterance)][intent] += 1
                        initial_labels[service][intent] += 1
                        if intent == "NONE":
                            none_initial += 1

    def collisions(groups: dict) -> dict:
        n = sum(sum(labels.values()) for labels in groups.values())
        errors = sum(sum(labels.values()) - max(labels.values()) for labels in groups.values())
        duplicate_rows = sum(sum(labels.values()) - len(labels) for labels in groups.values())
        conflict_rows = sum(sum(labels.values()) for labels in groups.values() if len(labels) > 1)
        return {
            "rows": n,
            "normalized_duplicate_rows": duplicate_rows,
            "conflicting_text_rows": conflict_rows,
            "minimum_errors_from_exact_text_conflicts": errors,
            "in_sample_deterministic_ceiling": round(1 - errors / n, 6) if n else None,
        }

    return {
        "dialogues": dialogue_count,
        "missing_utterance_or_label": missing,
        "unknown_intent": unknown,
        "initial_multi_frame_dialogues": multi_frame_initial,
        "initial_none_rows": none_initial,
        "initial": collisions(initial),
        "all_active_turns": collisions(all_active),
        "initial_by_service": {
            service: {"rows": sum(counts.values()), "labels": dict(sorted(counts.items()))}
            for service, counts in sorted(initial_labels.items())
        },
    }


def build() -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"unexpected SGD commit: {commit}")
    if subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("official SGD checkout is modified")

    schema = {split: schemas(split) for split in ("train", "dev", "test")}
    phase10 = json.loads((ROOT / "artifacts/experiments/phase10/splits.json").read_text())
    phase10_train = {item.removeprefix("sgd:") for item in phase10["train"]["schemas"] if item.startswith("sgd:")}
    phase10_dev = {item.removeprefix("sgd:") for item in phase10["development"]["schemas"]}
    if (phase10_train | phase10_dev) != set(schema["train"]):
        raise ValueError("Phase 10 exposure map does not cover SGD train schemas")
    if len(phase10_train) != 21 or len(phase10_dev) != 5:
        raise ValueError("Phase 10 exposure count changed")

    train = sorted(s for s, item in schema["train"].items() if len(item["intents"]) >= 2)
    validation = sorted(
        s for s, item in schema["dev"].items()
        if s not in schema["train"] and len(item["intents"]) >= 2
    )
    sealed = sorted(
        s for s, item in schema["test"].items()
        if s not in schema["train"] and s not in schema["dev"] and len(item["intents"]) >= 2
    )
    if not set(train).isdisjoint(validation) or not set(train).isdisjoint(sealed) or not set(validation).isdisjoint(sealed):
        raise ValueError("service split overlap")
    if not validation or not sealed:
        raise ValueError("no unseen validation or sealed services")

    source_files = [SOURCE / "LICENSE.txt"]
    source_files += [SOURCE / split / "schema.json" for split in ("train", "dev", "test")]
    source_files += sorted(SOURCE.glob("train/dialogues_*.json"))
    source_files += sorted(SOURCE.glob("dev/dialogues_*.json"))
    source_files += sorted(SOURCE.glob("test/dialogues_*.json"))
    report = {
        "format": "veya-phase12-sgd-inventory-v1",
        "source": SOURCE_URL,
        "commit": commit,
        "license": "CC-BY-SA-4.0",
        "source_files_sha256": {str(path.relative_to(SOURCE)): file_hash(path) for path in source_files},
        "phase10_exposure": {
            "trained_services": sorted(phase10_train),
            "development_services": sorted(phase10_dev),
            "base_checkpoint_pretraining_exposure": "unknown",
        },
        "phase12_split": {
            "train": train,
            "validation": validation,
            "sealed_test": sealed,
            "excluded_one_intent_services": sorted({
                s for split in schema.values() for s, item in split.items() if len(item["intents"]) < 2
            }),
            "split_unit": "whole service_name; service_name is a schema ID, not a customer ID",
            "selection_rule": "every USER frame for a known service with at least two official intents; entire dialogue remains causally available through that turn",
            "test_dialogues_parsed": False,
        },
        "schema_intent_count": {
            split: {s: len(item["intents"]) for s, item in sorted(services.items())}
            for split, services in schema.items()
        },
        "label_audit": {
            split: audit_dialogues(split, schema[split]) for split in ("train", "dev")
        },
        "unmeasured": [
            "human inter-annotator agreement",
            "semantic ambiguity beyond exact text collisions",
            "commercial customer representativeness",
            "trained-weight compatibility with CC-BY-SA-4.0",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "commit": report["commit"],
        "services": {key: len(value) for key, value in report["phase12_split"].items() if isinstance(value, list)},
        "train_initial": report["label_audit"]["train"]["initial"],
        "train_all_active_turns": report["label_audit"]["train"]["all_active_turns"],
        "test_dialogues_parsed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
