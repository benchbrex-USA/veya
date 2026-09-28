"""Matched all-turn SGD development baselines for three causal context modes."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase12_context import MODES, load_examples, pack_context
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash
from research.veyabench.run_phase12_sgd_baseline import wilson95
from veya.common import QTYPES, build_sequence, collate_items
from veya.release_bundle import load_release_manifest

RESULT = ROOT / "artifacts" / "experiments" / "phase12" / "context_baselines.json"
MANIFEST = ROOT / "artifacts" / "release" / "manifest.json"
BATCH_SIZE = 4


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def summarize(records: list[dict], audits: list[dict], elapsed: float) -> dict:
    n = len(records)
    correct = sum(record["correct"] for record in records)
    raw_token_counts = [audit["raw_tokens"] for audit in audits if "raw_tokens" in audit]

    def token_percentile(percentile: int) -> float | None:
        return float(np.percentile(raw_token_counts, percentile)) if raw_token_counts else None

    by_service = defaultdict(lambda: [0, 0])
    by_slice = defaultdict(lambda: [0, 0])
    for record in records:
        for key, group in ((record["service"], by_service),
                           ("initial" if record["initial"] else "later", by_slice),
                           ("transition" if record["transition"] else "same_service", by_slice)):
            group[key][0] += 1
            group[key][1] += record["correct"]
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "wilson95": wilson95(correct, n),
        "none_rows": sum(record["gold"] == "NONE" for record in records),
        "none_correct": sum(record["correct"] for record in records if record["gold"] == "NONE"),
        "over_budget_deferred": sum(record["over_budget"] for record in records),
        "by_service": {key: {"n": pair[0], "accuracy": pair[1] / pair[0]}
                       for key, pair in sorted(by_service.items())},
        "by_slice": {key: {"n": pair[0], "accuracy": pair[1] / pair[0]}
                     for key, pair in sorted(by_slice.items())},
        "context": {
            "dropped_prior_turns_total": sum(audit.get("dropped_prior_turns", 0) for audit in audits),
            "truncated_decisions": sum(audit.get("dropped_prior_turns", 0) > 0 for audit in audits),
            "raw_tokens_p50": token_percentile(50),
            "raw_tokens_p95": token_percentile(95),
            "raw_tokens_p99": token_percentile(99),
        },
        "elapsed_s_including_preprocessing": elapsed,
        "batch_throughput_decisions_per_second": n / elapsed,
    }


def evaluate_mode(bundle, rows: list[dict], schema: dict[str, dict], mode: str) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    agent = bundle.agent
    tok = agent.tok
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    prepared = []
    audits = []
    records_by_index: dict[int, dict] = {}
    for index, row in enumerate(rows):
        service = schema[row["service"]]
        criteria = {intent["name"]: intent["description"] for intent in service["intents"]}
        question = {
            "t": "choice",
            "ins": "Select the current active intent within this known service. Service: "
                   + service["description"],
            "crit": criteria,
        }
        packed, audit = pack_context(
            tok, row, mode, question, max_len=max_len, head_max_len=head_max_len
        )
        audits.append(audit)
        if packed is None:
            records_by_index[index] = {
                "id": row["id"], "prediction": None, "confidence": None,
                "margin": None,
                "service": row["service"], "gold": row["gold"], "correct": 0,
                "initial": row["initial_user_turn"],
                "transition": row["service_transition"], "over_budget": True,
            }
            continue
        ids, markers = build_sequence(tok, packed, question, max_len, head_max_len)
        if len(markers) != len(criteria):
            raise ValueError("candidate disappeared during tokenization")
        prepared.append((index, row, list(criteria), {
            "ids": ids, "markers": markers, "qtype": QTYPES["choice"],
        }))
    print(f"{mode}: prepared {len(prepared)}/{len(rows)} decisions", flush=True)

    with torch.inference_mode():
        for start in range(0, len(prepared), BATCH_SIZE):
            chunk = prepared[start:start + BATCH_SIZE]
            batch = collate_items([[item] for _, _, _, item in chunk], tok.pad_token_id)
            logits, _actions = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )
            values = logits.float().cpu().numpy()
            for offset, (index, row, labels, _item) in enumerate(chunk):
                temperature = bundle.temperature_for("packed", len(labels))
                probabilities = softmax(values[offset, :len(labels)], temperature)
                prediction = labels[int(np.argmax(probabilities))]
                sorted_probabilities = np.sort(probabilities)[::-1]
                records_by_index[index] = {
                    "id": row["id"], "prediction": prediction,
                    "confidence": float(sorted_probabilities[0]),
                    "margin": float(sorted_probabilities[0] - sorted_probabilities[1]),
                    "service": row["service"], "gold": row["gold"],
                    "correct": int(prediction == row["gold"]),
                    "initial": row["initial_user_turn"],
                    "transition": row["service_transition"], "over_budget": False,
                }
            if (start + BATCH_SIZE) % 1000 == 0:
                print(f"{mode}: scored {min(start + BATCH_SIZE, len(prepared))}/{len(prepared)}", flush=True)
    records = [records_by_index[index] for index in range(len(rows))]
    return summarize(records, audits, time.perf_counter() - started), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--write-details", action="store_true")
    args = parser.parse_args()
    result_path = RESULT if tuple(args.modes) == MODES else RESULT.with_name(
        "context_baseline_selected_rerun.json"
    )
    inventory = json.loads(OUTPUT.read_text())
    if inventory["phase12_split"]["test_dialogues_parsed"]:
        raise ValueError("test dialogue seal was broken")
    chosen = set(inventory["phase12_split"]["validation"])
    rows = load_examples("dev", chosen, inventory["source_files_sha256"])
    schema_path = SOURCE / "dev" / "schema.json"
    if file_hash(schema_path) != inventory["source_files_sha256"]["dev/schema.json"]:
        raise ValueError("dev schema changed")
    schema = {item["service_name"]: item for item in json.loads(schema_path.read_text())}
    bundle, _verified = load_release_manifest(str(MANIFEST), device="cpu", root=str(ROOT))
    report = {
        "format": "veya-phase12-context-baselines-v1",
        "source_commit": inventory["commit"],
        "inventory_sha256": file_hash(OUTPUT),
        "release_manifest_sha256": file_hash(MANIFEST),
        "device": "cpu",
        "platform": platform.platform(),
        "n_matched": len(rows),
        "batch_size": BATCH_SIZE,
        "test_dialogues_opened": False,
        "modes": {},
    }
    for mode in args.modes:
        aggregate, records = evaluate_mode(bundle, rows, schema, mode)
        report["modes"][mode] = aggregate
        if args.write_details:
            details = RESULT.with_name(f"context_baseline_{mode}_details.jsonl")
            with details.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({mode: {"n": item["n"], "accuracy": item["accuracy"],
                             "truncated": item["context"]["truncated_decisions"]}
                      for mode, item in report["modes"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
