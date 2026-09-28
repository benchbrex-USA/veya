"""One-shot aggregate-only CLINC150 blind evaluation for the frozen release.

This harness intentionally never writes predictions, per-example scores, error
rows, or class-level breakdowns.  It refuses to run if a result already exists.
"""
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from research.veyabench.run_phase3_eval import ROOT  # noqa: E402
from veya.common import ece_score  # noqa: E402
from veya.release_bundle import INSTRUCTIONS, load_release_manifest  # noqa: E402
from veya.selective import apply_report, wilson_interval  # noqa: E402

RESULT_PATH = os.path.join(ROOT, "artifacts", "release", "blind_clinc150_val_result.json")
PERMUTATION_EXAMPLES = 64
PERMUTATIONS_PER_EXAMPLE = 4
NLL_FLOOR = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def rss_mb():
    try:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return round(int(raw.strip()) / 1024.0, 1)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def latency_summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
    }


def rate(count, n):
    if n == 0:
        return {"count": int(count), "n": 0, "rate": None, "wilson95": None}
    interval = wilson_interval(n - count, n)
    return {
        "count": int(count),
        "n": int(n),
        "rate": count / n,
        "wilson95": None if interval is None else [1.0 - interval[1], 1.0 - interval[0]],
    }


def softmax(logits, temperature):
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max()
    probs = np.exp(values)
    return probs / probs.sum()


def permutation_audit(bundle, rows, criteria):
    rng = random.Random(20260923)
    labels = list(criteria)
    trials = 0
    shortlist_set_changes = 0
    top1_flips = 0
    max_logit_delta = 0.0
    for text, _truth in rows[:PERMUTATION_EXAMPLES]:
        baseline = bundle.score_choice(text, INSTRUCTIONS, criteria)
        baseline_logits = dict(zip(baseline["labels"], baseline["logits"]))
        for _ in range(PERMUTATIONS_PER_EXAMPLE):
            shuffled = list(labels)
            rng.shuffle(shuffled)
            permuted = {label: criteria[label] for label in shuffled}
            moved = bundle.score_choice(text, INSTRUCTIONS, permuted)
            trials += 1
            if set(moved["labels"]) != set(baseline["labels"]):
                shortlist_set_changes += 1
                continue
            moved_logits = dict(zip(moved["labels"], moved["logits"]))
            max_logit_delta = max(
                max_logit_delta,
                max(abs(float(baseline_logits[label]) - float(moved_logits[label])) for label in baseline_logits),
            )
            if moved["prediction"] != baseline["prediction"]:
                top1_flips += 1
    return {
        "examples": min(PERMUTATION_EXAMPLES, len(rows)),
        "permutations_per_example": PERMUTATIONS_PER_EXAMPLE,
        "trials": trials,
        "shortlist_set_changes": shortlist_set_changes,
        "top1_flips": top1_flips,
        "max_remapped_logit_delta": max_logit_delta,
    }


def main():
    if os.path.exists(RESULT_PATH):
        raise SystemExit("blind result already exists; refusing a second run")
    manifest_path = os.path.join(ROOT, "artifacts", "release", "manifest.json")
    freeze_path = os.path.join(ROOT, "docs", "audit", "blind_clinc150_val_freeze.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(freeze_path, encoding="utf-8") as handle:
        freeze = json.load(handle)
    blind = manifest["blind_evaluation"]
    if blind.get("status") != "FROZEN_UNOPENED" or freeze.get("evaluation_runs") != 0:
        raise SystemExit("blind protocol is not in the frozen-unopened state")
    if sha256_file(freeze_path) != blind["freeze_manifest_hash"]:
        raise SystemExit("blind freeze manifest hash mismatch")
    data_path = os.path.join(ROOT, blind["dataset"])
    if sha256_file(data_path) != blind["dataset_hash"] or blind["dataset_hash"] != freeze["source_file_sha256"]:
        raise SystemExit("blind dataset hash mismatch")
    harness_path = os.path.realpath(__file__)
    if sha256_file(harness_path) != blind.get("harness_hash"):
        raise SystemExit("blind harness is not frozen in the release manifest")

    with open(data_path, encoding="utf-8") as handle:
        dataset = json.load(handle)
    rows = dataset[freeze["split"]]
    ids = ["clinc150:%s:%d" % (freeze["split"], index) for index in range(len(rows))]
    if len(rows) != freeze["example_count"] or sha256_lines(ids) != freeze["example_ids_sha256"]:
        raise SystemExit("blind example IDs do not match the freeze")
    labels = sorted({label for _text, label in rows})
    if len(labels) != freeze["candidate_count"]:
        raise SystemExit("blind candidate universe changed")
    criteria = {label: "intent: " + label.replace("_", " ") for label in labels}

    load_started = time.perf_counter()
    bundle, verified_manifest = load_release_manifest(manifest_path, device="cpu", root=ROOT)
    load_s = time.perf_counter() - load_started
    if bundle.hypothesis is not None or bundle.reranker_id != "base_decision_head":
        raise SystemExit("blind run is not using the frozen base scorer")
    if bundle.selective_policy.get("operating_epsilon") != "disabled":
        raise SystemExit("blind run unexpectedly enables automation")

    hits = 0
    rerank_correct = 0
    end_to_end_correct = 0
    conditional_nll = []
    conditional_brier = []
    honest_nll = []
    honest_brier = []
    confidence = []
    correct_flags = []
    latencies = []
    token_counts = []
    started = time.perf_counter()
    for index, (text, truth) in enumerate(rows):
        row_started = time.perf_counter()
        scored = bundle.score_choice(text, INSTRUCTIONS, criteria)
        latencies.append((time.perf_counter() - row_started) * 1000.0)
        picked = scored["labels"]
        probs = np.asarray(scored["probabilities"], dtype=np.float64)
        hit = truth in picked
        correct = scored["prediction"] == truth
        hits += int(hit)
        rerank_correct += int(hit and correct)
        end_to_end_correct += int(correct)
        confidence.append(float(probs.max()))
        correct_flags.append(1.0 if correct else 0.0)
        token_counts.append(int(scored["input_tokens"]))
        if hit:
            gold = picked.index(truth)
            row_nll = -math.log(max(float(probs[gold]), NLL_FLOOR))
            target = np.zeros_like(probs)
            target[gold] = 1.0
            row_brier = float(np.sum((probs - target) ** 2))
            conditional_nll.append(row_nll)
            conditional_brier.append(row_brier)
            honest_nll.append(row_nll)
            honest_brier.append(row_brier)
        else:
            honest_nll.append(-math.log(NLL_FLOOR))
            honest_brier.append(float(1.0 + np.sum(probs ** 2)))
        if (index + 1) % 250 == 0:
            print("blind progress %d/%d" % (index + 1, len(rows)), flush=True)
    elapsed = time.perf_counter() - started

    permutation = permutation_audit(bundle, rows, criteria)
    scores = np.asarray(confidence, dtype=np.float64)
    correct_array = np.asarray(correct_flags, dtype=np.float64)
    thresholds = bundle.selective_policy["thresholds"]
    descriptive_policy = {
        epsilon: apply_report(scores, correct_array, threshold)
        for epsilon, threshold in thresholds.items()
    }
    report = {
        "format": "veya-blind-evaluation-v1",
        "status": "SEALED_AGGREGATES_ONLY",
        "release": verified_manifest["release"],
        "release_manifest_sha256": bundle.manifest_sha256,
        "harness_sha256": sha256_file(harness_path),
        "dataset": "CLINC150 %s (in-scope)" % freeze["split"],
        "dataset_sha256": blind["dataset_hash"],
        "example_ids_sha256": freeze["example_ids_sha256"],
        "candidate_universe_sha256": sha256_lines(labels),
        "n": len(rows),
        "candidate_count": len(labels),
        "architecture": verified_manifest["architecture_version"],
        "scorer_id": bundle.reranker_id,
        "choice": {
            "retrieval_recall_at_8": rate(hits, len(rows)),
            "conditional_accuracy_given_hit": rate(rerank_correct, hits),
            "end_to_end_accuracy": rate(end_to_end_correct, len(rows)),
            "conditional_nll_on_hits": float(np.mean(conditional_nll)),
            "conditional_brier_on_hits": float(np.mean(conditional_brier)),
            "honest_end_to_end_nll_with_miss_floor_1e-12": float(np.mean(honest_nll)),
            "honest_end_to_end_brier_over_150_classes": float(np.mean(honest_brier)),
            "ece_end_to_end": ece_score(scores, correct_array),
            "mean_max_probability": float(scores.mean()),
            "mean_input_tokens": float(np.mean(token_counts)),
            "retrieval_misses": len(rows) - hits,
            "reranker_misses_with_target_present": hits - rerank_correct,
        },
        "selective_policy": {
            "runtime_status": "DISABLED",
            "frozen_validation_thresholds_applied_descriptively": descriptive_policy,
            "note": "No threshold was fit or selected on the blind labels.",
        },
        "permutation": permutation,
        "runtime": {
            "device": str(bundle.agent.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cold_load_s": round(load_s, 3),
            "evaluation_s": round(elapsed, 3),
            "throughput_examples_per_second": len(rows) / elapsed,
            "latency": latency_summary(latencies),
            "rss_mb_at_end": rss_mb(),
        },
        "integrity": {
            "per_example_outputs_written": False,
            "class_level_outputs_written": False,
            "model_selection_after_unblind_permitted": False,
            "evaluation_runs": 1,
        },
    }
    with open(RESULT_PATH, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
