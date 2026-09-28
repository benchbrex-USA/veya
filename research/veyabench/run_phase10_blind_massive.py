"""One-shot aggregate-only MASSIVE en-US test evaluation for Phase 10.

The harness transitions its freeze manifest to RUNNING before model scoring and
refuses any rerun.  It never writes example-level predictions, errors, ranks,
confidences, or class metrics.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow.parquet as pq
import torch

from research.veyabench.phase10_data import PHASE10, ROOT
from research.veyabench.run_phase10_train import ARTIFACT_PATH, MANIFEST_PATH
from veya.experimental.domain_retriever import load_domain_retriever, naturalize_label, text_sha256
from veya.release_bundle import INSTRUCTIONS, load_release_manifest, raw_choice_logits
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


DATA_PATH = os.path.join(ROOT, "research", "data", "massive", "massive-test.parquet")
FREEZE_PATH = os.path.join(ROOT, "docs", "phase10", "blind_massive_freeze.json")
RESULT_PATH = os.path.join(PHASE10, "blind_massive_result.json")
SEED = 20260923


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def binomial_rate(count: int, total: int) -> dict:
    if total == 0:
        return {"count": count, "n": total, "rate": None, "wilson95": None}
    z = 1.96
    p = count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "count": count,
        "n": total,
        "rate": p,
        "wilson95": [max(0.0, center - margin), min(1.0, center + margin)],
    }


def latency(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
    }


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max()
    probs = np.exp(values)
    return probs / probs.sum()


def paired_bootstrap(base: np.ndarray, new: np.ndarray, draws: int = 2000) -> dict:
    delta = new.astype(np.float64) - base.astype(np.float64)
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[sample].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "draws": draws,
    }


def class_names(table, field: str) -> list[str]:
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    return list(metadata["info"]["features"][field]["names"])


def rank_metrics(scores: np.ndarray, gold: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    positions = np.argmax(order == gold[:, None], axis=1) + 1
    return (
        {
            "recall_at_1": binomial_rate(int(np.sum(positions <= 1)), len(gold)),
            "recall_at_4": binomial_rate(int(np.sum(positions <= 4)), len(gold)),
            "recall_at_8": binomial_rate(int(np.sum(positions <= 8)), len(gold)),
            "recall_at_16": binomial_rate(int(np.sum(positions <= 16)), len(gold)),
            "recall_at_32": binomial_rate(int(np.sum(positions <= 32)), len(gold)),
            "mrr": float(np.mean(1.0 / positions)),
            "mean_rank": float(np.mean(positions)),
            "median_rank": float(np.median(positions)),
        },
        positions,
        order,
    )


def score_hits(bundle, texts, gold, labels, order, route_name: str) -> tuple[dict, np.ndarray]:
    hits = 0
    conditional_correct = 0
    end_to_end = np.zeros(len(texts), dtype=bool)
    margins = []
    latency_ms = []
    for index, text in enumerate(texts):
        picked_indices = order[index, :8]
        if gold[index] not in picked_indices:
            continue
        hits += 1
        picked = [labels[int(value)] for value in picked_indices]
        criteria = {label: "intent: " + naturalize_label(label) for label in picked}
        started = time.perf_counter()
        logits, _tokens = raw_choice_logits(bundle.agent, text, INSTRUCTIONS, criteria)
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        probabilities = softmax(logits, bundle.route_temperature)
        prediction = picked[int(np.argmax(probabilities))]
        truth = labels[int(gold[index])]
        correct = prediction == truth
        conditional_correct += int(correct)
        end_to_end[index] = correct
        top = np.sort(probabilities)[-2:]
        margins.append(float(top[-1] - top[-2]))
        if hits % 250 == 0:
            print("blind scorer %s hits %d" % (route_name, hits), flush=True)
    return (
        {
            "retrieval_hits": hits,
            "conditional_accuracy": binomial_rate(conditional_correct, hits),
            "end_to_end_accuracy": binomial_rate(conditional_correct, len(texts)),
            "mean_top1_top2_probability_margin_on_hits": float(np.mean(margins)),
            "scorer_latency_on_hits": latency(latency_ms),
        },
        end_to_end,
    )


def main() -> None:
    if os.path.exists(RESULT_PATH):
        raise SystemExit("blind result already exists; refusing a second run")
    with open(FREEZE_PATH, encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "FROZEN_UNOPENED" or freeze.get("evaluation_runs") != 0:
        raise SystemExit("blind freeze is not unopened; refusing to score")
    pre_run_freeze_sha256 = file_sha256(FREEZE_PATH)
    if file_sha256(DATA_PATH) != freeze["dataset_sha256"]:
        raise SystemExit("blind dataset hash mismatch")
    if file_sha256(ARTIFACT_PATH) != freeze["retriever_sha256"]:
        raise SystemExit("retriever changed after blind freeze")
    if file_sha256(MANIFEST_PATH) != freeze["release_manifest_sha256"]:
        raise SystemExit("release manifest changed after blind freeze")
    if file_sha256(os.path.realpath(__file__)) != freeze["harness_sha256"]:
        raise SystemExit("blind harness changed after freeze")

    table = pq.read_table(DATA_PATH)
    ids = [str(value) for value in table["id"].to_pylist()]
    labels = class_names(table, "intent")
    if len(ids) != freeze["example_count"] or text_sha256(ids) != freeze["example_ids_sha256"]:
        raise SystemExit("blind example IDs changed")
    if len(labels) != freeze["candidate_count"] or text_sha256(labels) != freeze["candidate_universe_sha256"]:
        raise SystemExit("blind candidate universe changed")
    if set(table["partition"].to_pylist()) != {"test"} or set(table["locale"].to_pylist()) != {"en-US"}:
        raise SystemExit("blind source contains an unexpected partition or locale")

    # One-way transition before any model scoring. A crash consumes the run.
    running = dict(freeze)
    running.update(
        {
            "status": "RUNNING_CONSUMED_ON_FAILURE",
            "evaluation_runs": 1,
            "started_at": utc_now(),
            "pre_run_freeze_sha256": pre_run_freeze_sha256,
        }
    )
    write_json(FREEZE_PATH, running)

    texts = [str(value) for value in table["utt"].to_pylist()]
    gold = np.asarray(table["intent"].to_pylist(), dtype=np.int64)
    load_started = time.perf_counter()
    bundle, release_manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    embed = embed_fn_from_agent(
        bundle.agent, max_length=int(bundle.router["embed_max_length"]), batch_size=16
    )
    retriever = load_domain_retriever(
        ARTIFACT_PATH, checkpoint_hash=bundle.checkpoint_hash, embed_fn=embed
    )
    load_seconds = time.perf_counter() - load_started

    query_parts = []
    query_started = time.perf_counter()
    for start in range(0, len(texts), 256):
        stop = min(start + 256, len(texts))
        query_parts.append(embed(texts[start:stop]))
        print("blind retrieval embeddings %d/%d" % (stop, len(texts)), flush=True)
    query_vectors = np.concatenate(query_parts, axis=0).astype(np.float32)
    naturalized_vectors = embed([naturalize_label(label) for label in labels])
    prefixed_vectors = embed(["intent: " + naturalize_label(label) for label in labels])
    query_seconds = time.perf_counter() - query_started

    with torch.inference_mode():
        current_scores = (
            bundle.projection(torch.as_tensor(query_vectors, dtype=torch.float32))
            @ bundle.projection(torch.as_tensor(prefixed_vectors, dtype=torch.float32)).T
        ).cpu().numpy()
    phase10_scores = retriever.project_queries(query_vectors) @ retriever.project_candidates(
        naturalized_vectors
    ).T
    current_retrieval, _current_positions, current_order = rank_metrics(current_scores, gold)
    phase10_retrieval, _phase10_positions, phase10_order = rank_metrics(phase10_scores, gold)

    permutation_trials = 0
    permutation_changes = 0
    rng = random.Random(SEED)
    projected_queries = retriever.project_queries(query_vectors[:64])
    projected_candidates = retriever.project_candidates(naturalized_vectors)
    for row_index in range(64):
        baseline = {labels[index]: float(phase10_scores[row_index, index]) for index in range(len(labels))}
        for _ in range(4):
            order = list(range(len(labels)))
            rng.shuffle(order)
            moved_scores = projected_queries[row_index] @ projected_candidates[order].T
            restored = {
                labels[source]: float(score) for source, score in zip(order, moved_scores)
            }
            permutation_trials += 1
            if any(abs(restored[label] - baseline[label]) > 1e-6 for label in labels):
                permutation_changes += 1

    current_e2e, current_correct = score_hits(
        bundle, texts, gold, labels, current_order, "current"
    )
    phase10_e2e, phase10_correct = score_hits(
        bundle, texts, gold, labels, phase10_order, "phase10"
    )
    report = {
        "format": "veya-phase10-blind-v1",
        "status": "SEALED_AGGREGATES_ONLY",
        "dataset": "Amazon MASSIVE en-US test",
        "dataset_sha256": freeze["dataset_sha256"],
        "source_revision": freeze["source_revision"],
        "license": "CC-BY-4.0",
        "example_count": len(texts),
        "candidate_count": len(labels),
        "pre_run_freeze_sha256": pre_run_freeze_sha256,
        "harness_sha256": freeze["harness_sha256"],
        "retriever_sha256": freeze["retriever_sha256"],
        "release_manifest_sha256": freeze["release_manifest_sha256"],
        "release": release_manifest["release"],
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "model_load_seconds": load_seconds,
            "retrieval_embedding_seconds": query_seconds,
        },
        "current": {"retrieval": current_retrieval, "end_to_end": current_e2e},
        "phase10": {"retrieval": phase10_retrieval, "end_to_end": phase10_e2e},
        "paired_end_to_end_delta": paired_bootstrap(current_correct, phase10_correct),
        "permutation": {
            "trials": permutation_trials,
            "score_map_changes": permutation_changes,
        },
        "frozen_components": {
            "k": 8,
            "scorer": "base_decision_head",
            "temperature": bundle.route_temperature,
            "calibration_refit": False,
            "automation_policy_changed": False,
        },
        "data_retention": "no per-example predictions, errors, scores, ranks, confidence, or class metrics written",
        "completed_at": utc_now(),
    }
    write_json(RESULT_PATH, report)
    consumed = dict(running)
    consumed.update(
        {
            "status": "CONSUMED",
            "completed_at": report["completed_at"],
            "result_sha256": file_sha256(RESULT_PATH),
        }
    )
    write_json(FREEZE_PATH, consumed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
