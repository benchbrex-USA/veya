"""Phase-10 data registry and contamination-aware split construction.

Only Banking77 ``train``, SGD ``train``, and CLINC150 ``train`` are read.  The
previously consumed Banking77 test and CLINC150 validation/test rows are never
loaded by this module.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from collections import defaultdict

from veya.experimental.domain_retriever import naturalize_label, text_sha256


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE10 = os.path.join(ROOT, "artifacts", "experiments", "phase10")
DATA_OUT = os.path.join(ROOT, "research", "data", "phase10")
SEED = 20260923
TRAIN_CAP_PER_LABEL = 40
VALIDATION_CAP_PER_LABEL = 20
DEVELOPMENT_SERVICES = frozenset(
    {"Calendar_1", "Flights_1", "Hotels_3", "Media_1", "Services_3"}
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_cap(rows: list[dict], per_label: int) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["candidate_id"]].append(row)
    out = []
    for label in sorted(by_label):
        ranked = sorted(
            by_label[label],
            key=lambda row: hashlib.sha256(
                (str(SEED) + "\0" + row["id"] + "\0" + row["text"]).encode("utf-8")
            ).hexdigest(),
        )
        out.extend(ranked[:per_label])
    return sorted(out, key=lambda row: row["id"])


def _candidate(
    candidate_id: str,
    label: str,
    definition: str | None,
    schema: str,
) -> dict:
    naturalized = naturalize_label(label)
    definition = (definition or "").strip()
    return {
        "candidate_id": candidate_id,
        "schema": schema,
        "raw": label,
        "naturalized": naturalized,
        "definition": definition or None,
        "name_definition": "%s. %s" % (naturalized, definition) if definition else naturalized,
    }


def load_banking_train() -> tuple[list[dict], list[dict]]:
    path = os.path.join(ROOT, "research", "data", "banking77", "train.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            label = row["category"]
            rows.append(
                {
                    "id": "banking77:train:%d" % index,
                    "dataset": "banking77",
                    "schema": "banking77",
                    "text": row["text"],
                    "candidate_id": "banking77:%s" % label,
                }
            )
    labels = sorted({row["candidate_id"].split(":", 1)[1] for row in rows})
    candidates = [_candidate("banking77:" + label, label, None, "banking77") for label in labels]
    return deterministic_cap(rows, TRAIN_CAP_PER_LABEL), candidates


def load_sgd_train() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    base = os.path.join(ROOT, "research", "data", "sgd", "train")
    with open(os.path.join(base, "schema.json"), encoding="utf-8") as handle:
        schemas = json.load(handle)
    candidates = []
    known = set()
    for service in schemas:
        schema = "sgd:%s" % service["service_name"]
        for intent in service["intents"]:
            candidate_id = "%s:%s" % (schema, intent["name"])
            known.add(candidate_id)
            candidates.append(
                _candidate(
                    candidate_id,
                    intent["name"],
                    intent.get("description"),
                    schema,
                )
            )
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "dialogues_*.json"))):
        with open(path, encoding="utf-8") as handle:
            dialogues = json.load(handle)
        for dialogue in dialogues:
            dialogue_id = str(dialogue["dialogue_id"])
            for turn_index, turn in enumerate(dialogue["turns"]):
                if turn.get("speaker") != "USER":
                    continue
                for frame_index, frame in enumerate(turn.get("frames", [])):
                    intent = frame.get("state", {}).get("active_intent", "NONE")
                    if intent == "NONE":
                        continue
                    candidate_id = "sgd:%s:%s" % (frame["service"], intent)
                    if candidate_id not in known:
                        raise ValueError("SGD row references unknown candidate %s" % candidate_id)
                    rows.append(
                        {
                            "id": "sgd:train:%s:%d:%d" % (dialogue_id, turn_index, frame_index),
                            "dataset": "sgd",
                            "schema": "sgd:%s" % frame["service"],
                            "text": turn["utterance"],
                            "candidate_id": candidate_id,
                        }
                    )
    development = [
        row for row in rows if row["schema"].split(":", 1)[1] in DEVELOPMENT_SERVICES
    ]
    train = [
        row for row in rows if row["schema"].split(":", 1)[1] not in DEVELOPMENT_SERVICES
    ]
    train_candidates = [
        row for row in candidates if row["schema"].split(":", 1)[1] not in DEVELOPMENT_SERVICES
    ]
    development_candidates = [
        row for row in candidates if row["schema"].split(":", 1)[1] in DEVELOPMENT_SERVICES
    ]
    return (
        deterministic_cap(train, TRAIN_CAP_PER_LABEL),
        train_candidates,
        deterministic_cap(development, VALIDATION_CAP_PER_LABEL),
        development_candidates,
    )


def load_clinc_train() -> tuple[list[dict], list[dict]]:
    path = os.path.join(ROOT, "research", "data", "clinc150", "data_full.json")
    with open(path, encoding="utf-8") as handle:
        train_rows = json.load(handle)["train"]
    rows = []
    for index, pair in enumerate(train_rows):
        text, label = pair
        rows.append(
            {
                "id": "clinc150:train:%d" % index,
                "dataset": "clinc150",
                "schema": "clinc150",
                "text": text,
                "candidate_id": "clinc150:%s" % label,
            }
        )
    labels = sorted({row["candidate_id"].split(":", 1)[1] for row in rows})
    candidates = [_candidate("clinc150:" + label, label, None, "clinc150") for label in labels]
    return deterministic_cap(rows, VALIDATION_CAP_PER_LABEL), candidates


def split_digest(rows: list[dict]) -> str:
    return text_sha256(
        "%s\0%s\0%s" % (row["id"], row["candidate_id"], row["text"]) for row in rows
    )


def build() -> dict:
    banking_rows, banking_candidates = load_banking_train()
    sgd_rows, sgd_candidates, dev_rows, dev_candidates = load_sgd_train()
    validation_rows, validation_candidates = load_clinc_train()
    e2e_rows = sorted(
        validation_rows,
        key=lambda row: hashlib.sha256(("e2e\0" + row["id"]).encode("utf-8")).hexdigest(),
    )[:1000]
    train_rows = banking_rows + sgd_rows
    train_candidates = banking_candidates + sgd_candidates

    train_schemas = sorted({row["schema"] for row in train_rows})
    development_schemas = sorted({row["schema"] for row in dev_rows})
    validation_schemas = sorted({row["schema"] for row in validation_rows})
    if set(train_schemas) & set(development_schemas):
        raise ValueError("development schema leaked into training")
    if set(train_schemas) & set(validation_schemas):
        raise ValueError("validation schema leaked into training")

    os.makedirs(DATA_OUT, exist_ok=True)
    os.makedirs(PHASE10, exist_ok=True)
    for name, rows in (
        ("train", train_rows),
        ("development", dev_rows),
        ("schema_held_out_validation", validation_rows),
    ):
        with open(os.path.join(DATA_OUT, name + ".jsonl"), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    candidates = train_candidates + dev_candidates + validation_candidates
    with open(os.path.join(DATA_OUT, "candidates.json"), "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    registry = {
        "format": "veya-phase10-dataset-registry-v1",
        "seed": SEED,
        "taint_rule": "Whole candidate schemas, not only rows, are excluded from retriever training when used for validation or blind evaluation.",
        "datasets": [
            {
                "name": "banking77",
                "version": "PolyAI task-specific-datasets local train.csv",
                "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
                "source_sha256": sha256_file(os.path.join(ROOT, "research/data/banking77/train.csv")),
                "license": "CC-BY-4.0",
                "domain": "online banking support",
                "language": "en",
                "task_type": "single-label intent classification",
                "num_examples": len(banking_rows),
                "num_labels": len(banking_candidates),
                "candidate_representation": "raw and deterministic naturalized labels; no definitions",
                "training_role": "TRAIN",
                "validation_role": None,
                "blind_role": None,
                "previous_exposure": "all train rows are development data; production retriever previously used a subset",
            },
            {
                "name": "schema_guided_dialogue",
                "version": "GitHub master archive f703636bd043057c7f492b78f82ea8a80d004ff49d60b9d66d5593b417ccc708",
                "source": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
                "source_sha256": {
                    "train_schema": sha256_file(os.path.join(ROOT, "research/data/sgd/train/schema.json")),
                    "license": sha256_file(os.path.join(ROOT, "research/data/sgd/LICENSE.txt")),
                },
                "license": "CC-BY-SA-4.0",
                "domain": "20 task-oriented service/API domains",
                "language": "en",
                "task_type": "active-intent retrieval by service schema",
                "num_examples": len(sgd_rows) + len(dev_rows),
                "num_labels": len(sgd_candidates) + len(dev_candidates),
                "candidate_representation": "intent name plus authoritative dataset definition",
                "training_role": "TRAIN except frozen development services",
                "validation_role": "DEVELOPMENT for %s" % sorted(DEVELOPMENT_SERVICES),
                "blind_role": None,
                "previous_exposure": "new in Phase 10",
            },
            {
                "name": "clinc150",
                "version": "local data_full.json sha256:%s" % sha256_file(os.path.join(ROOT, "research/data/clinc150/data_full.json")),
                "source": "https://github.com/clinc/oos-eval",
                "source_sha256": sha256_file(os.path.join(ROOT, "research/data/clinc150/data_full.json")),
                "license": "dataset repository license recorded locally",
                "domain": "150 assistant intents across 10 domains",
                "language": "en",
                "task_type": "single-label intent classification",
                "num_examples": len(validation_rows),
                "num_labels": len(validation_candidates),
                "candidate_representation": "raw and deterministic naturalized labels; no authoritative definitions",
                "training_role": None,
                "validation_role": "SCHEMA-HELD-OUT VALIDATION using train partition only",
                "blind_role": None,
                "previous_exposure": "validation/test partitions consumed in Phase 1-9; their examples and errors are prohibited; train partition newly designated development data",
            },
            {
                "name": "amazon_massive_en_us",
                "version": "UNFROZEN",
                "source": "https://huggingface.co/datasets/AmazonScience/massive",
                "source_sha256": None,
                "license": "CC-BY-4.0",
                "domain": "general voice-assistant intents",
                "language": "en-US",
                "task_type": "single-label intent classification",
                "num_examples": None,
                "num_labels": 60,
                "candidate_representation": "not inspected before finalist freeze",
                "training_role": None,
                "validation_role": None,
                "blind_role": "FUTURE BLIND CANDIDATE; data absent and untouched",
                "previous_exposure": "none known; base-checkpoint provenance cannot independently exclude pretraining exposure",
            },
        ],
    }
    with open(os.path.join(DATA_OUT, "registry.json"), "w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    freeze = {
        "format": "veya-phase10-split-freeze-v1",
        "seed": SEED,
        "train": {
            "examples": len(train_rows),
            "candidates": len(train_candidates),
            "schemas": train_schemas,
            "sha256": split_digest(train_rows),
        },
        "development": {
            "examples": len(dev_rows),
            "candidates": len(dev_candidates),
            "schemas": development_schemas,
            "sha256": split_digest(dev_rows),
        },
        "schema_held_out_validation": {
            "examples": len(validation_rows),
            "candidates": len(validation_candidates),
            "schemas": validation_schemas,
            "source_partition": "CLINC150 train only",
            "sha256": split_digest(validation_rows),
            "matched_end_to_end_subset_examples": len(e2e_rows),
            "matched_end_to_end_subset_ids_sha256": text_sha256(row["id"] for row in e2e_rows),
        },
        "prohibited": [
            "research/data/banking77/test.csv",
            "CLINC150 val partition",
            "CLINC150 test partition",
        ],
        "future_blind": "unfrozen and absent until finalist selection",
    }
    with open(os.path.join(PHASE10, "splits.json"), "w", encoding="utf-8") as handle:
        json.dump(freeze, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"registry": registry, "freeze": freeze}


if __name__ == "__main__":
    print(json.dumps(build()["freeze"], indent=2))
