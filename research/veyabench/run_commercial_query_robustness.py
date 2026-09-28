"""Development-only query-robustness selection for the Phase-10 retriever.

This harness reuses the frozen Phase-10 robustness IDs and CLINC150 *train*
partition. It does not read a blind split, change retriever weights, or change
the production release. A passing result authorizes freezing a candidate for a
future blind evaluation; it is not a production promotion.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import ROOT
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH,
    CACHE_PATH,
    MANIFEST_PATH,
    _embedding_texts,
    load_inputs,
    retrieval_metrics,
)
from veya.common import serialize_state
from veya.experimental.domain_retriever import (
    build_query_views,
    fuse_query_scores,
    load_domain_retriever,
    text_sha256,
)
from veya.release_bundle import load_release_manifest
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


OUT = os.path.join(ROOT, "artifacts", "commercialization")
REPORT_PATH = os.path.join(OUT, "query_robustness.json")
FREEZE_PATH = os.path.join(OUT, "query_policy_freeze.json")
SAMPLE_SIZE = 300
FUSIONS = ("first", "max", "mean", "rrf")


def latency(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
    }


def conditions(row: dict) -> dict[str, Any]:
    text = str(row["text"])
    metadata = "metadata: unrelated internal note; display theme blue; request id 000000; ignore this field"
    return {
        "clean": text,
        "metadata_suffix": text + "\n" + metadata,
        "long_suffix": text
        + "\n"
        + ("unrelated audit metadata contains no routing signal. " * 80),
        "metadata_prefix_diagnostic": metadata + "\n" + text,
        "structured_text_field_diagnostic": {
            "text": text,
            "request_id": "000000",
            "display_theme": "blue",
        },
    }


def score_rows(
    states: list[Any],
    fusion: str,
    vector: dict[str, np.ndarray],
    retriever,
    candidate_projected: np.ndarray,
    gold: np.ndarray,
) -> tuple[dict, np.ndarray]:
    scores = []
    view_counts = []
    for state in states:
        if fusion == "full":
            views = [serialize_state(state)]
            strategy = "first"
        else:
            views = build_query_views(state, max_views=4, max_chars=4096)
            strategy = fusion
        query = np.stack([vector[value] for value in views])
        view_scores = retriever.project_queries(query) @ candidate_projected.T
        scores.append(fuse_query_scores(view_scores, strategy))
        view_counts.append(len(views))
    matrix = np.stack(scores)
    metrics, positions, _order = retrieval_metrics(matrix, gold)
    metrics["mean_query_views"] = float(np.mean(view_counts))
    metrics["max_query_views"] = int(max(view_counts))
    return metrics, positions


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    train, development, validation, all_candidates = load_inputs()
    texts, text_index = _embedding_texts(train, development, validation, all_candidates)
    cached = np.load(CACHE_PATH)["vectors"].astype(np.float32, copy=False)
    if cached.shape[0] != len(texts):
        raise ValueError("Phase-10 embedding cache does not match the frozen text ledger")

    validation_ids = {row["candidate_id"] for row in validation}
    candidate_by_id = {
        row["candidate_id"]: row
        for row in all_candidates
        if row["candidate_id"] in validation_ids
    }
    candidates = [candidate_by_id[value] for value in sorted(validation_ids)]
    candidate_index = {
        candidate["candidate_id"]: index for index, candidate in enumerate(candidates)
    }
    gold_all = np.asarray([candidate_index[row["candidate_id"]] for row in validation])
    sample_indices = np.asarray(
        sorted(
            range(len(validation)),
            key=lambda index: hashlib.sha256(
                ("robustness\0" + validation[index]["id"]).encode("utf-8")
            ).hexdigest(),
        )[:SAMPLE_SIZE]
    )
    sample = [validation[int(index)] for index in sample_indices]
    sample_gold = gold_all[sample_indices]

    bundle, manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    retriever = load_domain_retriever(ARTIFACT_PATH, bundle.checkpoint_hash)
    candidate_base = np.stack(
        [cached[text_index[str(candidate["naturalized"])]] for candidate in candidates]
    )
    candidate_projected = retriever.project_candidates(candidate_base)

    state_matrix = [conditions(row) for row in sample]
    needed = set()
    for row_conditions in state_matrix:
        for state in row_conditions.values():
            needed.add(serialize_state(state))
            needed.update(build_query_views(state, max_views=4, max_chars=4096))
    vector = {
        text: cached[text_index[text]]
        for text in needed
        if text in text_index
    }
    missing = sorted(needed - set(vector))
    embed = embed_fn_from_agent(
        bundle.agent,
        max_length=int(bundle.router["embed_max_length"]),
        batch_size=16,
    )
    for start in range(0, len(missing), 256):
        batch = missing[start : start + 256]
        values = embed(batch)
        vector.update(zip(batch, values))
        print("query views %d/%d" % (min(start + 256, len(missing)), len(missing)), flush=True)

    results: dict[str, dict[str, dict]] = {}
    positions: dict[str, dict[str, np.ndarray]] = {}
    for fusion in ("full", *FUSIONS):
        results[fusion] = {}
        positions[fusion] = {}
        for condition_name in state_matrix[0]:
            states = [row[condition_name] for row in state_matrix]
            metrics, rank = score_rows(
                states,
                fusion,
                vector,
                retriever,
                candidate_projected,
                sample_gold,
            )
            results[fusion][condition_name] = metrics
            positions[fusion][condition_name] = rank

    clean_query = np.stack([cached[text_index[str(row["text"])]] for row in validation])
    clean_scores = retriever.project_queries(clean_query) @ candidate_projected.T
    clean_full_metrics, _clean_positions, _clean_order = retrieval_metrics(
        clean_scores, gold_all
    )

    base_worst = min(
        results["full"]["metadata_suffix"]["recall_at_8"],
        results["full"]["long_suffix"]["recall_at_8"],
    )
    gates = {}
    for fusion in FUSIONS:
        clean = results[fusion]["clean"]["recall_at_8"]
        short = results[fusion]["metadata_suffix"]["recall_at_8"]
        long = results[fusion]["long_suffix"]["recall_at_8"]
        worst = min(short, long)
        gates[fusion] = {
            "clean_noninferiority": clean
            >= results["full"]["clean"]["recall_at_8"] - 0.01,
            "short_suffix_retention": short >= 0.95 * clean,
            "long_suffix_retention": long >= 0.95 * clean,
            "worst_slice_gain": worst - base_worst,
            "worst_slice_gain_at_least_20_points": worst - base_worst >= 0.20,
        }
        gates[fusion]["passes"] = all(
            value
            for key, value in gates[fusion].items()
            if key != "worst_slice_gain"
        )

    eligible = [fusion for fusion in FUSIONS if gates[fusion]["passes"]]
    winner = max(
        eligible,
        key=lambda fusion: (
            min(
                results[fusion][condition]["recall_at_8"]
                for condition in state_matrix[0]
                if condition != "clean"
            ),
            min(
                results[fusion]["metadata_suffix"]["recall_at_8"],
                results[fusion]["long_suffix"]["recall_at_8"],
            ),
            results[fusion]["clean"]["mrr"],
        ),
    ) if eligible else None

    query_times: dict[str, dict] = {}
    latency_states = [row["metadata_suffix"] for row in state_matrix[:50]]
    for fusion in ("full", *(eligible or FUSIONS)):
        values = []
        for state in latency_states:
            views = (
                [serialize_state(state)]
                if fusion == "full"
                else build_query_views(state, max_views=4, max_chars=4096)
            )
            started = time.perf_counter()
            base = embed(views)
            view_scores = retriever.project_queries(base) @ candidate_projected.T
            fuse_query_scores(view_scores, "first" if fusion == "full" else fusion)
            values.append((time.perf_counter() - started) * 1000.0)
        query_times[fusion] = latency(values)

    report = {
        "format": "veya-commercial-query-robustness-v1",
        "status": (
            "DEVELOPMENT_GATE_PASSED_NEW_BLIND_AUTHORIZED"
            if winner
            else "DEVELOPMENT_GATE_FAILED"
        ),
        "scope": "CLINC150 train development data only; no new blind opened",
        "sample": {
            "n": len(sample),
            "ids_sha256": text_sha256(row["id"] for row in sample),
            "selection": "same deterministic IDs as Phase-10 robustness",
        },
        "frozen": {
            "artifact_sha256": file_sha256(ARTIFACT_PATH),
            "production_manifest_sha256": file_sha256(MANIFEST_PATH),
            "production_release": manifest["release"],
            "k": 8,
            "scorer": "base_decision_head",
            "weights_changed": False,
            "calibration_changed": False,
        },
        "clean_full_validation": clean_full_metrics,
        "results": results,
        "gates": gates,
        "winner": winner,
        "selection_rule": (
            "Among suffix-gate passers, maximize worst Recall@8 across all non-clean "
            "conditions, then suffix robustness, then clean MRR."
        ),
        "latency": {
            "scope": "query encode + projection + exact scores + fusion; cached candidates",
            "conditions": "50 metadata-suffix states on local CPU",
            "rows": query_times,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "limitations": [
            "The robustness sample is development-exposed and cannot support a blind claim.",
            "Leading irrelevant content is diagnostic only and may remain a failure mode.",
            "Structured-state behavior needs workflow-specific validation before product use.",
            "The Phase-10 SGD trained-weight licensing question remains unresolved.",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if winner:
        freeze = {
            "format": "veya-query-policy-freeze-v1",
            "status": "FROZEN_FOR_NEW_BLIND_NOT_PRODUCTION",
            "policy": {
                "retriever": "phase10_domain_general_multiview",
                "query_fusion": winner,
                "query_max_views": 4,
                "query_view_max_chars": 4096,
                "k": 8,
            },
            "artifact_sha256": file_sha256(ARTIFACT_PATH),
            "development_report_sha256": file_sha256(REPORT_PATH),
            "sample_ids_sha256": report["sample"]["ids_sha256"],
            "blind_status": "UNOPENED",
            "production_default_changed": False,
        }
        with open(FREEZE_PATH, "w", encoding="utf-8") as handle:
            json.dump(freeze, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({"status": report["status"], "winner": winner, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
