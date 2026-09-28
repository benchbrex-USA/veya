"""Read-only Phase 12 packed Choice baseline on unseen SGD dev services.

This never opens official test dialogues and writes aggregate metrics only.
It does not train, calibrate, or select a threshold.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from research.veyabench.phase12_sgd_inventory import EXPECTED_COMMIT, OUTPUT, ROOT, SOURCE, file_hash
from veya.release_bundle import load_release_manifest


RESULT = ROOT / "artifacts" / "experiments" / "phase12" / "sgd_dev_baseline.json"
MANIFEST = ROOT / "artifacts" / "release" / "manifest.json"


def wilson95(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [center - margin, center + margin]


def load_validation(inventory: dict) -> tuple[list[dict], dict]:
    if inventory["commit"] != EXPECTED_COMMIT or inventory["phase12_split"]["test_dialogues_parsed"]:
        raise ValueError("Phase 12 source protocol changed")
    schema_path = SOURCE / "dev" / "schema.json"
    expected = inventory["source_files_sha256"]["dev/schema.json"]
    if file_hash(schema_path) != expected:
        raise ValueError("SGD dev schema changed")
    schema = {item["service_name"]: item for item in json.loads(schema_path.read_text())}
    chosen = set(inventory["phase12_split"]["validation"])
    rows = []
    for path in sorted(glob.glob(str(SOURCE / "dev" / "dialogues_*.json"))):
        relative = str(Path(path).relative_to(SOURCE))
        if file_hash(Path(path)) != inventory["source_files_sha256"][relative]:
            raise ValueError(f"SGD source changed: {relative}")
        with open(path, encoding="utf-8") as handle:
            dialogues = json.load(handle)
        for dialogue in dialogues:
            first = next((turn for turn in dialogue["turns"] if turn["speaker"] == "USER"), None)
            if first is None:
                raise ValueError("dialogue has no USER turn")
            for frame in first["frames"]:
                service = frame["service"]
                if service not in chosen:
                    continue
                gold = frame["state"]["active_intent"]
                rows.append({"service": service, "text": first["utterance"], "gold": gold})
    rows.sort(key=lambda row: hashlib.sha256(
        (row["service"] + "\0" + row["text"] + "\0" + row["gold"]).encode()
    ).hexdigest())
    return rows, schema


def score() -> dict:
    inventory = json.loads(OUTPUT.read_text())
    rows, schema = load_validation(inventory)
    bundle, verified = load_release_manifest(str(MANIFEST), device="cpu", root=str(ROOT))
    results = defaultdict(lambda: {"n": 0, "correct": 0, "confusion": Counter()})
    latencies = []
    probs = []
    correctness = []
    nll = []
    brier = []
    n_none = 0
    for index, row in enumerate(rows, 1):
        service = schema[row["service"]]
        criteria = {intent["name"]: intent["description"] for intent in service["intents"]}
        instructions = (
            "Select the active intent of this initial user request within the known service. "
            "Service: " + service["description"]
        )
        if row["gold"] == "NONE":
            n_none += 1
            continue
        started = time.perf_counter()
        scored = bundle.score_choice(row["text"], instructions, criteria)
        latencies.append((time.perf_counter() - started) * 1000)
        if scored["path"] != "packed" or scored["shortlist"] is not None:
            raise ValueError("expected all-candidate packed Choice path")
        labels = scored["labels"]
        probabilities = scored["probabilities"]
        target = labels.index(row["gold"])
        correct = scored["prediction"] == row["gold"]
        entry = results[row["service"]]
        entry["n"] += 1
        entry["correct"] += int(correct)
        entry["confusion"][(row["gold"], scored["prediction"])] += 1
        correctness.append(int(correct))
        probs.append(float(max(probabilities)))
        nll.append(float(-math.log(max(float(probabilities[target]), 1e-12))))
        brier.append(float(sum(
            (float(probability) - int(i == target)) ** 2
            for i, probability in enumerate(probabilities)
        )))
        if index % 100 == 0:
            print(f"Phase 12 SGD dev baseline {index}/{len(rows)}", flush=True)
    n = len(correctness)
    confidence = np.asarray(probs)
    hits = np.asarray(correctness)
    overall = {
        "n": n,
        "correct": int(hits.sum()),
        "accuracy": float(hits.mean()),
        "wilson95": wilson95(int(hits.sum()), n),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "mean_confidence": float(confidence.mean()),
        "high_confidence_errors_at_0_9": int(((hits == 0) & (confidence >= 0.9)).sum()),
        "latency_ms": {f"p{q}": float(np.percentile(latencies, q)) for q in (50, 95, 99)},
        "throughput_sequential_per_second": n / (sum(latencies) / 1000),
    }
    by_service = {}
    for service, entry in sorted(results.items()):
        by_service[service] = {
            "n": entry["n"],
            "correct": entry["correct"],
            "accuracy": entry["correct"] / entry["n"],
            "wilson95": wilson95(entry["correct"], entry["n"]),
            "confusion": {
                f"{gold} -> {predicted}": count
                for (gold, predicted), count in sorted(entry["confusion"].items())
            },
        }
    return {
        "format": "veya-phase12-sgd-dev-baseline-v1",
        "source_commit": EXPECTED_COMMIT,
        "inventory_sha256": file_hash(OUTPUT),
        "release_manifest_sha256": file_hash(MANIFEST),
        "verified_release": verified.get("release") if isinstance(verified, dict) else None,
        "population": "initial USER turn; whole dev-only services; >=2 official intents",
        "candidate_representation": "official intent descriptions; official service description in instruction",
        "device": "cpu",
        "platform": platform.platform(),
        "test_dialogues_opened": False,
        "none_rows": n_none,
        "candidate_access": {"recall_at_4": 1.0, "recall_at_8": 1.0, "recall_at_16": 1.0, "recall_at_32": 1.0, "recall_at_all": 1.0},
        "overall": overall,
        "by_service": by_service,
        "limitations": [
            "No training, calibration, threshold fitting, or sealed test was performed.",
            "Six of seven validation services have one observed first-turn intent; majority-by-service scores 97.1%. This baseline cannot support a 98% commercial claim.",
            "Local CPU scoring latency excludes HTTP, concurrency, and cold load.",
            "Public SGD services are not commercial customers.",
        ],
    }


def main() -> None:
    report = score()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": report["overall"], "by_service": {
        service: {"n": value["n"], "accuracy": value["accuracy"]}
        for service, value in report["by_service"].items()
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
