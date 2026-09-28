"""WF98-G11: schema-conditioned contextual scorer with sibling hard negatives.

Trains on official SGD TRAIN services only, inside the frozen `previous_turn`
input representation chosen by `context_selection.json`. Relative to the
released Phase 12 sparse scorer this adds:

1. Explicit speaker/turn-position tokens (`stagepos_*`) as per-candidate
   features. Only positions derivable from the frozen rendered input are used:
   no prior USER turn, and the prior turn's speaker prefix. The prior turn's
   service identity is deliberately NOT used, because the frozen contract
   excludes prior-turn frame annotations from model input.
2. Sibling-pair features: one hashed feature per (candidate pair, scored
   candidate) direction, so the model can learn per-service sibling
   adjustments that shared lexical features cannot express.
3. Hard negatives for sibling intents and stage transitions: a second
   training pass over an upsampled copy of the TRAIN rows that the first pass
   misclassified as a sibling intent, or that mark a service transition.

This is a workflow-specific research candidate, not a served Veya model. It
uses current/prior utterances and official intent descriptions, never states,
future turns, or sealed test dialogues.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from research.veyabench.phase12_context import load_examples
from research.veyabench.phase12_sgd_inventory import OUTPUT, ROOT, SOURCE, file_hash
from research.veyabench.run_phase12_sparse_scorer import (
    DIM,
    SEED,
    evaluate,
    features,
    schema_candidates,
    scores,
)

RESULT = ROOT / "artifacts" / "experiments" / "phase12" / "g11_scorer_development.json"


def g11_fields(row: dict, mode: str) -> list[tuple[str, set[str]]]:
    """Frozen `previous_turn` context fields plus stage-position tokens.

    The utterance text visible to the model is exactly the frozen
    representation: the current USER utterance and the immediately preceding
    USER or SYSTEM turn. Stage tokens carry only what the rendered input
    already shows: the absence of a prior USER turn, and the prior turn's
    speaker prefix.
    """
    from research.veyabench.run_phase12_sparse_scorer import context_fields

    fields = context_fields(row, mode)
    extras: set[str] = set()
    if row.get("initial_user_turn"):
        extras.add("stagepos_no_prior_user_turn")
    if row["history"]:
        speaker = row["history"][-1][0].casefold()
        extras.add("stagepos_prior_" + speaker)
    if extras:
        fields.append(("stage", extras))
    return fields


def g11_item_features(
    row: dict, options: dict, mode: str
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Candidate features with sibling-pair indices added per scored candidate."""
    fields = g11_fields(row, mode)
    candidates = options[row["service"]]
    names = [candidate[0] for candidate in candidates]
    items = []
    for candidate in candidates:
        indices, values = features(fields, candidate)
        name = candidate[0]
        expanded = set(int(i) for i in indices)
        for other in names:
            if other != name:
                pair = "|".join(sorted((name, other)))
                expanded.add(int(features_hashed("sib|" + pair + "|scored:" + name)))
        items.append((np.fromiter(sorted(expanded), dtype=np.int32), values))
    return items, names


def features_hashed(value: str) -> int:
    from research.veyabench.run_phase12_sparse_scorer import hashed

    return hashed(value)


def fit_weighted(
    rows: list[dict],
    options: dict,
    mode: str,
    *,
    epochs: int,
    lr: float,
    weights_init: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Online softmax SGD with optional per-row loss weights and warm start."""
    weights = np.zeros(DIM, dtype=np.float32) if weights_init is None else weights_init
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        order = rng.permutation(len(rows))
        correct = 0.0
        weighted_total = 0.0
        loss_total = 0.0
        for step, position in enumerate(order):
            index = int(position)
            row = rows[index]
            weight = 1.0 if row_weights is None else float(row_weights[index])
            item, names = g11_item_features(row, options, mode)
            gold = names.index(row["gold"])
            logits = scores(weights, item)
            correct += weight * int(np.argmax(logits) == gold)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            loss_total -= weight * math.log(max(float(probabilities[gold]), 1e-9))
            probabilities[gold] -= 1.0
            for gradient, (indices, values) in zip(probabilities, item):
                scale = -lr * weight * float(gradient)
                np.add.at(weights, indices, scale / np.sqrt(len(indices)))
                weights[:8] += scale * values
            weighted_total += weight
            if (step + 1) % 20000 == 0:
                print(f"epoch {epoch + 1}: {step + 1}/{len(rows)}", flush=True)
        summary = {
            "epoch": epoch + 1,
            "train_online_accuracy": correct / weighted_total,
            "train_online_nll": loss_total / weighted_total,
            "seconds": time.perf_counter() - started,
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
    return weights, history


def train_errors(rows: list[dict], options: dict, mode: str, weights: np.ndarray) -> list[int]:
    """Positions of TRAIN rows the given weights misclassify as a sibling intent."""
    positions = []
    for position, row in enumerate(rows):
        item, names = g11_item_features(row, options, mode)
        logits = scores(weights, item)
        prediction = names[int(np.argmax(logits))]
        if prediction == row["gold"] or prediction == "NONE" or row["gold"] == "NONE":
            continue
        positions.append(position)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("utterance_only", "previous_turn", "full_context"),
                        default="previous_turn")
    parser.add_argument("--max-train", type=int, default=158042)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hard-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--hard-weight", type=float, default=4.0)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="result filename inside artifacts/experiments/phase12/; "
                             "variants must not overwrite each other")
    args = parser.parse_args()
    result_path = (
        ROOT / "artifacts" / "experiments" / "phase12" / args.output
        if args.output else RESULT
    )
    stem = args.output.rsplit(".", 1)[0] if args.output else "g11_scorer"
    inventory = json.loads(OUTPUT.read_text())
    if inventory["phase12_split"]["test_dialogues_parsed"]:
        raise ValueError("test seal broken")
    if args.mode != "previous_turn":
        raise ValueError("the frozen input representation is previous_turn; "
                         "see artifacts/experiments/phase12/context_selection.json")
    hashes = inventory["source_files_sha256"]
    train = load_examples("train", set(inventory["phase12_split"]["train"]), hashes)
    dev = load_examples("dev", set(inventory["phase12_split"]["validation"]), hashes)
    schemas = {}
    for split in ("train", "dev"):
        path = SOURCE / split / "schema.json"
        if file_hash(path) != hashes[f"{split}/schema.json"]:
            raise ValueError("schema changed")
        schemas[split] = schema_candidates(
            {item["service_name"]: item for item in json.loads(path.read_text())})
    if len(train) > args.max_train:
        rng = np.random.default_rng(SEED)
        chosen = sorted(rng.choice(len(train), size=args.max_train, replace=False))
        train = [train[int(index)] for index in chosen]

    print(f"pass 1: {args.epochs} epochs on {len(train)} TRAIN rows", flush=True)
    weights, pass1 = fit_weighted(train, schemas["train"], args.mode,
                                  epochs=args.epochs, lr=args.lr)
    hard_positions = train_errors(train, schemas["train"], args.mode, weights)
    print(f"hard negatives: {len(hard_positions)} sibling-intent errors", flush=True)

    upsampled = list(train)
    extra_rows = []
    for position in hard_positions:
        for _ in range(int(args.hard_weight) - 1):
            extra_rows.append(train[position])
    # Upsampling via duplicated rows: each duplicate adds one more gradient step,
    # so duplicated rows receive (1 + duplicates) times the updates of a normal row.
    upsampled += extra_rows
    row_weights = np.ones(len(upsampled), dtype=np.float64)
    print(f"pass 2: {args.hard_epochs} epochs on {len(upsampled)} upsampled rows "
          f"({len(extra_rows)} duplicated sibling errors)", flush=True)
    weights, pass2 = fit_weighted(upsampled, schemas["train"], args.mode,
                                  epochs=args.hard_epochs, lr=args.lr,
                                  weights_init=weights, row_weights=row_weights)

    started = time.perf_counter()
    dev_result, details = evaluate(dev, schemas["dev"], args.mode, weights)
    evaluation_seconds = time.perf_counter() - started

    active = [row for row in dev if row["gold"] != "NONE"]
    active_correct = sum(detail["correct"] for detail in details if detail["gold"] != "NONE")
    report = {
        "format": "veya-phase12-g11-scorer-diagnostic-v1",
        "workflow": "schema_guided_service_intent_routing",
        "source_commit": inventory["commit"],
        "inventory_sha256": file_hash(OUTPUT),
        "mode": args.mode,
        "train_services": inventory["phase12_split"]["train"],
        "dev_services": inventory["phase12_split"]["validation"],
        "n_train_sampled": len(train),
        "n_train_upsampled": len(upsampled),
        "n_hard_negatives": len(hard_positions),
        "n_dev": len(dev),
        "epochs": args.epochs,
        "hard_epochs": args.hard_epochs,
        "lr": args.lr,
        "hard_weight": args.hard_weight,
        "feature_dimension": DIM,
        "additions": [
            "stagepos speaker/turn-position tokens (frozen-input derivable only)",
            "sibling-pair per-direction hashed features",
            "hard negatives: duplicated sibling-intent errors from TRAIN",
        ],
        "training_history": {"pass1": pass1, "pass2": pass2},
        "development": dev_result,
        "n_active_intent": len(active),
        "active_intent_accuracy": active_correct / len(active) if active else None,
        "development_evaluation_seconds": evaluation_seconds,
        "test_dialogues_opened": False,
        "caveat": "Development-only research candidate; no calibration, deferral policy, "
                  "robustness, or sealed evaluation. Trained inside the frozen "
                  "previous_turn input representation on TRAIN services only.",
    }
    if args.save_weights:
        path = result_path.with_name(
            f"{stem}_{args.mode}_{len(train)}_{args.epochs}+{args.hard_epochs}.npz")
        np.savez_compressed(path, weights=weights)
        report["weights_path"] = str(path.relative_to(ROOT))
        report["weights_sha256"] = file_hash(path)
    if args.write_details:
        path = result_path.with_name(
            f"{stem}_{args.mode}_{len(train)}_{args.epochs}+{args.hard_epochs}_details.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        report["details_path"] = str(path.relative_to(ROOT))
        report["details_sha256"] = file_hash(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"development_accuracy": dev_result["accuracy"],
                      "active_intent_accuracy": report["active_intent_accuracy"],
                      "n": dev_result["n"]}), flush=True)


if __name__ == "__main__":
    main()
