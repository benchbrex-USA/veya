"""Post-selection query-input, cardinality, and detailed latency diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import PHASE10, ROOT
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH,
    CACHE_PATH,
    MANIFEST_PATH,
    _embedding_texts,
    load_inputs,
    retrieval_metrics,
)
from veya.experimental.domain_retriever import load_domain_retriever
from veya.release_bundle import load_release_manifest
from veya.shortlist import embed_fn_from_agent


REPORT_PATH = os.path.join(PHASE10, "diagnostics.json")
QUESTION = "Which label best fits the text?"


def rss_mb() -> float | None:
    try:
        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        )
        return int(raw.strip()) / 1024.0
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def latency(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
    }


def hashed_subset(row: dict, candidates: list[dict], truth_index: int, count: int) -> list[int]:
    others = [index for index in range(len(candidates)) if index != truth_index]
    others.sort(
        key=lambda index: hashlib.sha256(
            (row["id"] + "\0" + candidates[index]["candidate_id"]).encode("utf-8")
        ).hexdigest()
    )
    return [truth_index] + others[: count - 1]


def cardinality_scaling(
    bundle,
    retriever,
    validation: list[dict],
    candidates: list[dict],
    query_vectors: np.ndarray,
    naturalized_vectors: np.ndarray,
    prefixed_vectors: np.ndarray,
) -> list[dict]:
    index = {candidate["candidate_id"]: position for position, candidate in enumerate(candidates)}
    gold = np.asarray([index[row["candidate_id"]] for row in validation])
    with torch.inference_mode():
        base_scores = (
            bundle.projection(torch.as_tensor(query_vectors, dtype=torch.float32))
            @ bundle.projection(torch.as_tensor(prefixed_vectors, dtype=torch.float32)).T
        ).cpu().numpy()
    new_scores = retriever.project_queries(query_vectors) @ retriever.project_candidates(
        naturalized_vectors
    ).T
    rows = []
    for count in (8, 16, 32, 50, 77, 100, 150, 250):
        if count > len(candidates):
            continue
        base_positions = []
        new_positions = []
        for row_index, row in enumerate(validation):
            subset = hashed_subset(row, candidates, int(gold[row_index]), count)
            base_order = np.argsort(-base_scores[row_index, subset], kind="stable")
            new_order = np.argsort(-new_scores[row_index, subset], kind="stable")
            base_positions.append(int(np.flatnonzero(base_order == 0)[0]) + 1)
            new_positions.append(int(np.flatnonzero(new_order == 0)[0]) + 1)
        base_positions = np.asarray(base_positions)
        new_positions = np.asarray(new_positions)
        rows.append(
            {
                "candidates": count,
                "current_recall_at_8": float(np.mean(base_positions <= 8)),
                "phase10_recall_at_8": float(np.mean(new_positions <= 8)),
                "current_mrr": float(np.mean(1.0 / base_positions)),
                "phase10_mrr": float(np.mean(1.0 / new_positions)),
            }
        )
    return rows


def query_ablation(retriever, embed, sample, candidates, naturalized_vectors) -> dict:
    candidate_index = {
        candidate["candidate_id"]: index for index, candidate in enumerate(candidates)
    }
    gold = np.asarray([candidate_index[row["candidate_id"]] for row in sample])
    inputs = {
        "state_only": [row["text"] for row in sample],
        "question_only": [QUESTION for _row in sample],
        "state_plus_question": [QUESTION + "\n" + row["text"] for row in sample],
        "structured_text_field": [json.dumps({"text": row["text"]}) for row in sample],
    }
    candidate_projected = retriever.project_candidates(naturalized_vectors)
    out = {}
    for name, texts in inputs.items():
        vectors = embed(texts)
        scores = retriever.project_queries(vectors) @ candidate_projected.T
        metrics, _positions, _order = retrieval_metrics(scores, gold)
        out[name] = metrics
    return out


def runtime_detail(bundle, artifact_path: str) -> list[dict]:
    embed = embed_fn_from_agent(
        bundle.agent, max_length=int(bundle.router["embed_max_length"]), batch_size=16
    )
    retriever = load_domain_retriever(
        artifact_path, checkpoint_hash=bundle.checkpoint_hash, embed_fn=embed
    )
    rows = []
    for count in (50, 100, 150, 500, 1000):
        candidates = ["candidate %04d account support topic" % index for index in range(count)]
        candidate_times = []
        query_times = []
        scoring_times = []
        for _ in range(3):
            started = time.perf_counter()
            candidate_base = embed(candidates)
            candidate_projected = retriever.project_candidates(candidate_base)
            candidate_times.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            query_base = embed(["I need help with my account"])
            query_projected = retriever.project_queries(query_base)[0]
            query_times.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            np.argsort(-(candidate_projected @ query_projected), kind="stable")[:8]
            scoring_times.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "candidates": count,
                "candidate_encoding_and_projection": latency(candidate_times),
                "query_encoding_and_projection": latency(query_times),
                "exact_score_and_top8": latency(scoring_times),
                "projected_cache_bytes": int(count * retriever.projection.output_dim * 4),
            }
        )
        print("detailed runtime", count, flush=True)
    return rows


def main() -> None:
    train, development, validation, all_candidates = load_inputs()
    all_texts, text_index = _embedding_texts(train, development, validation, all_candidates)
    del all_texts
    vectors = np.load(CACHE_PATH)["vectors"]

    def vector(text: str) -> np.ndarray:
        return vectors[text_index[text]]

    validation_ids = {row["candidate_id"] for row in validation}
    validation_candidate_map = {
        candidate["candidate_id"]: candidate
        for candidate in all_candidates
        if candidate["candidate_id"] in validation_ids
    }
    validation_candidates = [validation_candidate_map[value] for value in sorted(validation_ids)]
    extra_candidates = [
        candidate for candidate in all_candidates if candidate["candidate_id"] not in validation_ids
    ]
    combined_candidates = validation_candidates + extra_candidates
    query_vectors = np.stack([vector(row["text"]) for row in validation])
    naturalized_vectors = np.stack(
        [vector(candidate["naturalized"]) for candidate in combined_candidates]
    )
    prefixed_vectors = np.stack(
        [vector("intent: " + candidate["naturalized"]) for candidate in combined_candidates]
    )
    bundle, _manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    retriever = load_domain_retriever(ARTIFACT_PATH, bundle.checkpoint_hash)
    scaling = cardinality_scaling(
        bundle,
        retriever,
        validation,
        combined_candidates,
        query_vectors,
        naturalized_vectors,
        prefixed_vectors,
    )

    embed = embed_fn_from_agent(
        bundle.agent, max_length=int(bundle.router["embed_max_length"]), batch_size=16
    )
    sample = sorted(
        validation,
        key=lambda row: hashlib.sha256(("query-ablation\0" + row["id"]).encode("utf-8")).hexdigest(),
    )[:600]
    query_inputs = query_ablation(
        retriever,
        embed,
        sample,
        validation_candidates,
        naturalized_vectors[: len(validation_candidates)],
    )
    report = {
        "format": "veya-phase10-diagnostics-v1",
        "status": "POST_SELECTION_DIAGNOSTIC_NO_RETUNING",
        "cardinality": {
            "protocol": "Each CLINC150-train query keeps its truth plus deterministic hashed distractors drawn from the combined CLINC150 and training/development candidate pool.",
            "rows": scaling,
        },
        "query_representation": {
            "n": len(sample),
            "results": query_inputs,
            "decision": "state_only remains frozen; diagnostics did not alter the artifact",
        },
        "runtime": {
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
            },
            "rss_mb_after_model_load": rss_mb(),
            "rows": runtime_detail(bundle, ARTIFACT_PATH),
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
