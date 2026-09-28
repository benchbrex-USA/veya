"""Train a small schema-conditioned lexical scorer on SGD train services only.

This is a workflow-specific research candidate, not a served Veya model. It
uses current/prior utterances and official intent descriptions, never states,
future turns, or sealed test dialogues.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import zlib
from collections import defaultdict

import numpy as np

from research.veyabench.phase12_context import MODES, load_examples
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash
from research.veyabench.run_phase12_sgd_baseline import wilson95


RESULT = ROOT / "artifacts" / "experiments" / "phase12" / "sparse_scorer_development.json"
DIM = 1 << 19
OVERLAP_DIM = 8
SEED = 12012
TOKEN = re.compile(r"[a-z0-9]+")
CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


def terms(text: str) -> set[str]:
    words = TOKEN.findall(text.casefold())
    return set(words) | {left + "_" + right for left, right in zip(words, words[1:])}


def candidate_terms(name: str, description: str) -> tuple[str, set[str], set[str]]:
    name_words = {word.casefold() for word in CAMEL.findall(name)}
    return name, name_words, terms(description)


def context_fields(row: dict, mode: str) -> list[tuple[str, set[str]]]:
    fields = [("current", terms(row["current"]))]
    history = row["history"]
    if mode == "utterance_only" or not history:
        return fields
    if mode == "previous_turn":
        speaker, text = history[-1]
        return fields + [("previous_" + speaker.casefold(), terms(text))]
    if mode != "full_context":
        raise ValueError(f"unknown mode {mode}")
    for speaker in ("USER", "SYSTEM"):
        prior = [text for who, text in history if who == speaker]
        if prior:
            fields.append(("recent_" + speaker.casefold(), terms(prior[-1])))
            fields.append(("all_" + speaker.casefold(), terms(" ".join(prior))))
    speaker, text = history[-1]
    fields.append(("previous_" + speaker.casefold(), terms(text)))
    return fields


def hashed(value: str) -> int:
    return OVERLAP_DIM + zlib.crc32(value.encode("utf-8")) % (DIM - OVERLAP_DIM)


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def features(fields: list[tuple[str, set[str]]], candidate: tuple[str, set[str], set[str]]) -> tuple[np.ndarray, np.ndarray]:
    name, name_words, description_words = candidate
    action = next(iter(CAMEL.findall(name)), name).casefold()
    values = np.zeros(OVERLAP_DIM, dtype=np.float32)
    for tag, words in fields:
        if tag == "current":
            values[0] = overlap(words, name_words)
            values[1] = overlap(words, description_words)
        elif tag.startswith("previous_"):
            values[2] = overlap(words, name_words)
            values[3] = overlap(words, description_words)
        elif tag.startswith("recent_"):
            values[4] += overlap(words, name_words)
            values[5] += overlap(words, description_words)
        elif tag.startswith("all_"):
            values[6] += overlap(words, name_words)
            values[7] += overlap(words, description_words)
    indices = {hashed("bias:name:" + name), hashed("bias:action:" + action)}
    for tag, words in fields:
        for word in words:
            indices.add(hashed(tag + "|" + word + "|name:" + name))
            indices.add(hashed(tag + "|" + word + "|action:" + action))
    return np.fromiter(sorted(indices), dtype=np.int32), values


def schema_candidates(schema: dict[str, dict]) -> dict[str, list[tuple[str, set[str], set[str]]]]:
    output = {}
    for service, definition in schema.items():
        options = [candidate_terms(item["name"], item["description"]) for item in definition["intents"]]
        options.append(("NONE", {"none"}, set()))
        output[service] = options
    return output


def item_features(row: dict, options: dict, mode: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    fields = context_fields(row, mode)
    candidates = options[row["service"]]
    return [features(fields, candidate) for candidate in candidates], [candidate[0] for candidate in candidates]


def scores(weights: np.ndarray, item: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.asarray([
        float(weights[indices].sum() / math.sqrt(len(indices)) + weights[:OVERLAP_DIM] @ values)
        for indices, values in item
    ], dtype=np.float32)


def fit(rows: list[dict], options: dict, mode: str, *, epochs: int, lr: float) -> tuple[np.ndarray, list[dict]]:
    weights = np.zeros(DIM, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        order = rng.permutation(len(rows))
        correct = 0
        loss_total = 0.0
        for step, position in enumerate(order):
            row = rows[int(position)]
            item, names = item_features(row, options, mode)
            gold = names.index(row["gold"])
            logits = scores(weights, item)
            correct += int(np.argmax(logits) == gold)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            loss_total -= math.log(max(float(probabilities[gold]), 1e-9))
            probabilities[gold] -= 1.0
            for gradient, (indices, values) in zip(probabilities, item):
                scale = -lr * float(gradient)
                np.add.at(weights, indices, scale / math.sqrt(len(indices)))
                weights[:OVERLAP_DIM] += scale * values
            if (step + 1) % 10000 == 0:
                print(f"epoch {epoch + 1}: {step + 1}/{len(rows)}", flush=True)
        summary = {"epoch": epoch + 1, "train_online_accuracy": correct / len(rows),
                   "train_online_nll": loss_total / len(rows),
                   "seconds": time.perf_counter() - started}
        history.append(summary)
        print(json.dumps(summary), flush=True)
    return weights, history


def evaluate(rows: list[dict], options: dict, mode: str, weights: np.ndarray) -> tuple[dict, list[dict]]:
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    by_service = defaultdict(lambda: [0, 0])
    by_slice = defaultdict(lambda: [0, 0])
    details = []
    for row in rows:
        item, names = item_features(row, options, mode)
        logits = scores(weights, item)
        prediction = names[int(np.argmax(logits))]
        normalized = np.exp(logits - logits.max())
        normalized /= normalized.sum()
        hit = int(prediction == row["gold"])
        correct += hit
        confusion[row["gold"]][prediction] += 1
        details.append({
            "id": row["id"], "service": row["service"], "gold": row["gold"],
            "prediction": prediction, "correct": hit,
            "confidence": float(normalized.max()),
            "initial": row["initial_user_turn"],
            "transition": row["service_transition"],
        })
        for key, groups in ((row["service"], by_service),
                            ("initial" if row["initial_user_turn"] else "later", by_slice),
                            ("transition" if row["service_transition"] else "same_service", by_slice)):
            groups[key][0] += 1
            groups[key][1] += hit
    return {
        "n": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "wilson95": wilson95(correct, len(rows)),
        "by_service": {name: {"n": pair[0], "accuracy": pair[1] / pair[0]}
                       for name, pair in sorted(by_service.items())},
        "by_slice": {name: {"n": pair[0], "accuracy": pair[1] / pair[0]}
                     for name, pair in sorted(by_slice.items())},
        "confusion": {gold: dict(sorted(predictions.items()))
                      for gold, predictions in sorted(confusion.items())},
    }, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(OUTPUT.read_text())
    if inventory["phase12_split"]["test_dialogues_parsed"]:
        raise ValueError("test seal broken")
    hashes = inventory["source_files_sha256"]
    train = load_examples("train", set(inventory["phase12_split"]["train"]), hashes)
    dev = load_examples("dev", set(inventory["phase12_split"]["validation"]), hashes)
    schemas = {}
    for split in ("train", "dev"):
        path = SOURCE / split / "schema.json"
        if file_hash(path) != hashes[f"{split}/schema.json"]:
            raise ValueError("schema changed")
        schemas[split] = schema_candidates({item["service_name"]: item for item in json.loads(path.read_text())})
    if len(train) > args.max_train:
        rng = np.random.default_rng(SEED)
        chosen = sorted(rng.choice(len(train), size=args.max_train, replace=False))
        train = [train[int(index)] for index in chosen]
    weights, history = fit(train, schemas["train"], args.mode, epochs=args.epochs, lr=args.lr)
    started = time.perf_counter()
    dev_result, details = evaluate(dev, schemas["dev"], args.mode, weights)
    report = {
        "format": "veya-phase12-sparse-scorer-diagnostic-v1",
        "source_commit": inventory["commit"],
        "inventory_sha256": file_hash(OUTPUT),
        "mode": args.mode,
        "train_services": inventory["phase12_split"]["train"],
        "dev_services": inventory["phase12_split"]["validation"],
        "n_train_sampled": len(train),
        "n_dev": len(dev),
        "epochs": args.epochs,
        "lr": args.lr,
        "feature_dimension": DIM,
        "training_history": history,
        "development": dev_result,
        "development_evaluation_seconds": time.perf_counter() - started,
        "test_dialogues_opened": False,
        "caveat": "Development-only research candidate; no calibration, deferral policy, robustness, or sealed evaluation.",
    }
    if args.save_weights:
        path = RESULT.with_name(f"sparse_scorer_{args.mode}_{len(train)}_{args.epochs}.npz")
        np.savez_compressed(path, weights=weights)
        report["weights_path"] = str(path.relative_to(ROOT))
        report["weights_sha256"] = file_hash(path)
    if args.write_details:
        path = RESULT.with_name(f"sparse_scorer_{args.mode}_{len(train)}_{args.epochs}_details.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        report["details_path"] = str(path.relative_to(ROOT))
        report["details_sha256"] = file_hash(path)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"development_accuracy": dev_result["accuracy"],
                      "n": dev_result["n"]}), flush=True)


if __name__ == "__main__":
    main()
