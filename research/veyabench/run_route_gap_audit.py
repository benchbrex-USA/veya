"""Close the Phase-6 routing evidence gap at 21 and 31 candidates.

The original clean training-validation sweep measured 20, 24, 28, and 32
candidates on Banking77 train rows 9300:9600.  This audit reuses those frozen
aggregates and evaluates only the two missing cardinalities with the identical
model, shortlist, row range, seed rule, and scorer.
"""
import hashlib
import json
import os
import platform
import random
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.run_phase3_eval import ROOT, SEED, load_banking, subset_criteria
from research.veyabench.run_release_eval import softmax_row
from veya.release_bundle import INSTRUCTIONS, load_bundle, raw_choice_logits

MISSING_GRID = (21, 31)
EXPECTED_EXISTING_GRID = (20, 24, 28, 32)
ROW_RANGE = (9300, 9600)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(bundle, rows, labels, criteria, n):
    packed_ok = []
    hierarchical_ok = []
    started = time.perf_counter()
    for index, (text, truth) in enumerate(rows):
        rng = random.Random(SEED + n * 10007 + index)
        subset, _gold = subset_criteria(labels, truth, n, rng)
        logits, _tokens = raw_choice_logits(bundle.agent, text, INSTRUCTIONS, subset)
        temperature = bundle.temperature_for("packed", len(subset))
        names = list(subset)
        packed = names[int(np.argmax(softmax_row(logits, temperature)))]
        packed_ok.append(packed == truth)

        picked, _stage = bundle.shortlist(text, list(subset.items()), 8)
        shortlist = {label: subset[label] for label in picked}
        h_logits, _h_tokens = raw_choice_logits(bundle.agent, text, INSTRUCTIONS, shortlist)
        hierarchical = picked[int(np.argmax(softmax_row(h_logits, bundle.route_temperature)))]
        hierarchical_ok.append(hierarchical == truth)
    return {
        "n_candidates": n,
        "examples": len(rows),
        "packed_accuracy": float(np.mean(packed_ok)),
        "hierarchical_accuracy": float(np.mean(hierarchical_ok)),
        "seconds": round(time.perf_counter() - started, 3),
        "source": "fresh_gap_measurement",
    }


def main():
    prior_path = os.path.join(ROOT, "artifacts", "experiments", "phase6", "final_report.json")
    with open(prior_path, encoding="utf-8") as handle:
        prior = json.load(handle)["cardinality_validation"]
    if tuple(prior["rows"]) != ROW_RANGE:
        raise SystemExit("Phase-6 row range changed")
    existing = {int(row["n_candidates"]): dict(row) for row in prior["table"]}
    if tuple(sorted(existing)) != EXPECTED_EXISTING_GRID:
        raise SystemExit("Phase-6 existing cardinality grid changed")
    for row in existing.values():
        if int(row["examples"]) != ROW_RANGE[1] - ROW_RANGE[0]:
            raise SystemExit("Phase-6 example count changed")
        row["source"] = "frozen_phase6_validation"

    checkpoint_hash = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bundle = load_bundle(
        os.path.join(ROOT, "models", "production"),
        os.path.join(ROOT, "models", "production", "choice_router.json"),
        os.path.join(ROOT, "models", "production", "shortlist_k8.pt"),
        os.path.join(ROOT, "models", "production", "calibration_shortlist_k8.json"),
        checkpoint_hash,
        device=device,
    )
    train = load_banking("train")
    rows = train[ROW_RANGE[0]:ROW_RANGE[1]]
    labels = list(bundle.labels)
    criteria = dict(zip(labels, bundle.label_texts))
    fresh = {}
    for n in MISSING_GRID:
        print("route-gap", n, flush=True)
        fresh[n] = evaluate(bundle, rows, labels, criteria, n)

    table = [existing.get(n) or fresh[n] for n in (20, 21, 24, 28, 31, 32)]
    gaps = {
        row["n_candidates"]: row["hierarchical_accuracy"] - row["packed_accuracy"]
        for row in table
    }
    first = min(n for n in gaps if gaps[n] >= 0.05 and all(gaps[m] >= 0.05 for m in gaps if m >= n))
    report = {
        "format": "veya-route-cutoff-audit-v1",
        "status": "DEVELOPMENT-EXPOSED / ROUTING-SELECTION ONLY",
        "protocol": {
            "dataset": "Banking77 train",
            "rows": list(ROW_RANGE),
            "seed": SEED,
            "checkpoint_hash": checkpoint_hash,
            "shortlist_hash": bundle.shortlist_sha256,
            "route_calibration_hash": file_sha256(os.path.join(ROOT, "models", "production", "calibration_shortlist_k8.json")),
            "packed_scorer": "base_decision_head",
            "hierarchical_scorer": "shortlist_k8_packed_rerank/base_decision_head",
            "device": device,
            "platform": platform.platform(),
        },
        "table": table,
        "decision_rule": "use shortlist when its accuracy lead is >=0.05 at this and every larger measured count",
        "first_supported_shortlist_count": first,
        "recommended_packed_max_candidates": first - 1,
        "decision": "shortlist at 20+; packed at <=19",
    }
    output = os.path.join(ROOT, "artifacts", "release", "route_cutoff_audit.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
