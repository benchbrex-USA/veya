"""Paired development comparison for the two saved sparse context models."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from research.veyabench.phase12_sgd_inventory import ROOT, file_hash


BASE = ROOT / "artifacts" / "experiments" / "phase12"
PREVIOUS = BASE / "sparse_scorer_previous_turn_158042_3_details.jsonl"
FULL = BASE / "sparse_scorer_full_context_158042_3_details.jsonl"
RESULT = BASE / "sparse_context_comparison.json"
SEED = 12012


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    previous = read(PREVIOUS)
    full = read(FULL)
    if len(previous) != len(full):
        raise ValueError("context models did not score matched decisions")
    groups = defaultdict(lambda: [0, 0])
    by_service = defaultdict(lambda: [0, 0, 0])
    either_correct = 0
    for left, right in zip(previous, full):
        if left["id"] != right["id"] or left["gold"] != right["gold"]:
            raise ValueError("context model decisions are not paired")
        dialogue_id = ":".join(left["id"].split(":")[:2])
        groups[dialogue_id][0] += left["correct"] - right["correct"]
        groups[dialogue_id][1] += 1
        service = by_service[left["service"]]
        service[0] += 1
        service[1] += left["correct"]
        service[2] += right["correct"]
        either_correct += int(left["correct"] or right["correct"])
    values = np.asarray(list(groups.values()), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    bootstrap = []
    for _ in range(2000):
        sample = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        bootstrap.append(sample[0] / sample[1])
    n = len(previous)
    classes = sorted({(row["service"], row["gold"]) for row in previous})
    class_metrics = {}
    for service, label in classes:
        tp = sum(row["service"] == service and row["gold"] == label
                 and row["prediction"] == label for row in previous)
        fn = sum(row["service"] == service and row["gold"] == label
                 and row["prediction"] != label for row in previous)
        fp = sum(row["service"] == service and row["gold"] != label
                 and row["prediction"] == label for row in previous)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[f"{service}/{label}"] = {
            "n": tp + fn, "precision": precision, "recall": recall, "f1": f1,
        }
    report = {
        "format": "veya-phase12-sparse-context-comparison-v1",
        "n_decisions": n,
        "n_dialogues": len(values),
        "source_detail_sha256": {"previous_turn": file_hash(PREVIOUS), "full_context": file_hash(FULL)},
        "previous_turn_accuracy": sum(row["correct"] for row in previous) / n,
        "full_context_accuracy": sum(row["correct"] for row in full) / n,
        "previous_minus_full_delta": sum(row["correct"] for row in previous)
            / n - sum(row["correct"] for row in full) / n,
        "dialogue_cluster_bootstrap_delta_ci95": [float(np.quantile(bootstrap, q)) for q in (0.025, 0.975)],
        "oracle_either_correct": either_correct,
        "oracle_either_accuracy": either_correct / n,
        "oracle_caveat": "Uses gold to choose per row and is not a deployable ensemble.",
        "previous_turn_service_class_macro_f1": sum(item["f1"] for item in class_metrics.values()) / len(class_metrics),
        "previous_turn_service_class_metrics": class_metrics,
        "by_service": {
            name: {"n": value[0], "previous_turn_accuracy": value[1] / value[0],
                   "full_context_accuracy": value[2] / value[0]}
            for name, value in sorted(by_service.items())
        },
        "test_dialogues_opened": False,
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"delta": report["previous_minus_full_delta"],
                      "ci95": report["dialogue_cluster_bootstrap_delta_ci95"],
                      "oracle_either_accuracy": report["oracle_either_accuracy"]}))


if __name__ == "__main__":
    main()
