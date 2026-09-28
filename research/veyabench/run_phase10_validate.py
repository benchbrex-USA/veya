"""Fresh-process reproducibility, frozen-scorer, robustness, and runtime audit."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import PHASE10, ROOT, SEED
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH,
    CACHE_META_PATH,
    CACHE_PATH,
    MANIFEST_PATH,
    _embedding_texts,
    load_inputs,
    paired_bootstrap,
    retrieval_metrics,
)
from veya.common import ece_score
from veya.experimental.domain_retriever import DomainRetriever, load_domain_retriever, naturalize_label
from veya.release_bundle import INSTRUCTIONS, load_release_manifest, raw_choice_logits
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


REPORT_PATH = os.path.join(PHASE10, "validation_report.json")


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max()
    probs = np.exp(values)
    return probs / probs.sum()


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


def embed_chunks(embed, texts: list[str], label: str) -> np.ndarray:
    parts = []
    for start in range(0, len(texts), 256):
        stop = min(start + 256, len(texts))
        parts.append(embed(texts[start:stop]))
        print("fresh embeddings %s %d/%d" % (label, stop, len(texts)), flush=True)
    return np.concatenate(parts, axis=0).astype(np.float32)


def candidate_text(candidate: dict, representation: str) -> str:
    if representation == "production_prefixed":
        return "intent: " + str(candidate["naturalized"])
    return str(candidate[representation])


def evaluate_e2e(bundle, validation, candidates, base_order, new_order) -> dict:
    subset = sorted(
        validation,
        key=lambda row: hashlib.sha256(("e2e\0" + row["id"]).encode("utf-8")).hexdigest(),
    )[:1000]
    row_index = {row["id"]: index for index, row in enumerate(validation)}
    expected_ids_hash = json.load(open(os.path.join(PHASE10, "splits.json"), encoding="utf-8"))[
        "schema_held_out_validation"
    ]["matched_end_to_end_subset_ids_sha256"]
    from veya.experimental.domain_retriever import text_sha256

    if text_sha256(row["id"] for row in subset) != expected_ids_hash:
        raise ValueError("matched end-to-end subset changed after freeze")

    outputs = {}
    predictions: dict[str, list[bool]] = {}
    for name, order in (("current", base_order), ("phase10", new_order)):
        hits = []
        correct = []
        conditional_correct = []
        margins = []
        confidence = []
        latencies = []
        for progress, row in enumerate(subset, start=1):
            index = row_index[row["id"]]
            picked_indices = order[index, :8]
            picked = [candidates[int(value)]["candidate_id"] for value in picked_indices]
            criteria = {
                candidate_id: "intent: "
                + naturalize_label(candidate_id.split(":", 1)[1])
                for candidate_id in picked
            }
            started = time.perf_counter()
            logits, _tokens = raw_choice_logits(bundle.agent, row["text"], INSTRUCTIONS, criteria)
            latencies.append((time.perf_counter() - started) * 1000.0)
            probs = softmax(logits, bundle.route_temperature)
            prediction = picked[int(np.argmax(probs))]
            hit = row["candidate_id"] in picked
            ok = prediction == row["candidate_id"]
            hits.append(hit)
            correct.append(ok)
            if hit:
                conditional_correct.append(ok)
                top = np.sort(probs)[-2:]
                margins.append(float(top[-1] - top[-2]))
            confidence.append(float(probs.max()))
            if progress % 250 == 0:
                print("e2e %s %d/%d" % (name, progress, len(subset)), flush=True)
        predictions[name] = correct
        hits_array = np.asarray(hits, dtype=bool)
        correct_array = np.asarray(correct, dtype=bool)
        outputs[name] = {
            "n": len(subset),
            "recall_at_8": float(hits_array.mean()),
            "conditional_accuracy": float(np.mean(conditional_correct)),
            "conditional_n": len(conditional_correct),
            "end_to_end_accuracy": float(correct_array.mean()),
            "mean_top1_top2_probability_margin_on_hits": float(np.mean(margins)),
            "ece_descriptive_with_frozen_temperature": ece_score(
                np.asarray(confidence), correct_array.astype(np.float64)
            ),
            "scorer_latency": latency(latencies),
        }
    outputs["paired_end_to_end_delta"] = paired_bootstrap(
        np.asarray(predictions["current"], dtype=bool),
        np.asarray(predictions["phase10"], dtype=bool),
    )
    outputs["frozen_components"] = {
        "scorer": "base_decision_head",
        "k": 8,
        "temperature": bundle.route_temperature,
        "calibration_refit": False,
        "automation_policy_changed": False,
    }
    outputs["theoretical_ceiling_approximation"] = (
        outputs["phase10"]["recall_at_8"] * outputs["current"]["conditional_accuracy"]
    )
    return outputs


def robustness(
    retriever: DomainRetriever,
    embed,
    validation: list[dict],
    candidates: list[dict],
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    gold: np.ndarray,
) -> dict:
    scores = retriever.project_queries(query_vectors) @ retriever.project_candidates(candidate_vectors).T
    lengths = np.asarray([len(row["text"]) for row in validation])
    boundaries = np.quantile(lengths, [0.0, 0.33, 0.67, 1.0])
    length_slices = []
    for index in range(3):
        low, high = boundaries[index], boundaries[index + 1]
        mask = (lengths >= low) & (lengths <= high if index == 2 else lengths < high)
        metrics, _positions, _order = retrieval_metrics(scores[mask], gold[mask])
        length_slices.append(
            {"slice": index, "min_chars": float(low), "max_chars": float(high), **metrics}
        )

    sample_indices = np.asarray(
        sorted(
            range(len(validation)),
            key=lambda index: hashlib.sha256(
                ("robustness\0" + validation[index]["id"]).encode("utf-8")
            ).hexdigest(),
        )[:300]
    )
    original = scores[sample_indices]
    distractor_texts = [
        row["text"]
        + "\nmetadata: unrelated internal note; display theme blue; request id 000000; ignore this field"
        for row in (validation[index] for index in sample_indices)
    ]
    long_texts = [
        row["text"]
        + "\n"
        + ("unrelated audit metadata contains no routing signal. " * 80)
        for row in (validation[index] for index in sample_indices)
    ]
    distractor_vectors = embed(distractor_texts)
    long_vectors = embed(long_texts)
    candidate_projected = retriever.project_candidates(candidate_vectors)
    distractor_scores = retriever.project_queries(distractor_vectors) @ candidate_projected.T
    long_scores = retriever.project_queries(long_vectors) @ candidate_projected.T
    sample_gold = gold[sample_indices]
    original_metrics, _op, original_order = retrieval_metrics(original, sample_gold)
    distractor_metrics, _dp, distractor_order = retrieval_metrics(distractor_scores, sample_gold)
    long_metrics, _lp, long_order = retrieval_metrics(long_scores, sample_gold)
    top8_jaccard = []
    for before, after in zip(original_order[:, :8], distractor_order[:, :8]):
        top8_jaccard.append(len(set(before) & set(after)) / len(set(before) | set(after)))

    by_label: dict[str, list[int]] = {}
    for index, row in enumerate(validation):
        by_label.setdefault(row["candidate_id"], []).append(index)
    pairs = []
    for values in by_label.values():
        if len(values) >= 2:
            pairs.append((values[0], values[1]))
    neighborhood = []
    for first, second in pairs:
        left = set(np.argsort(-scores[first], kind="stable")[:8])
        right = set(np.argsort(-scores[second], kind="stable")[:8])
        neighborhood.append(len(left & right) / len(left | right))

    max_similarity = scores.max(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    hit = np.any(np.argsort(-scores, axis=1, kind="stable")[:, :8] == gold[:, None], axis=1)
    return {
        "query_length_slices": length_slices,
        "distractor": {
            "n": len(sample_indices),
            "before": original_metrics,
            "after": distractor_metrics,
            "recall_at_8_delta": distractor_metrics["recall_at_8"] - original_metrics["recall_at_8"],
            "mean_top8_jaccard": float(np.mean(top8_jaccard)),
        },
        "long_structured_state": {
            "n": len(sample_indices),
            "before": original_metrics,
            "after": long_metrics,
            "recall_at_8_delta": long_metrics["recall_at_8"] - original_metrics["recall_at_8"],
            "configured_max_tokens": 96,
            "note": "long suffix intentionally exceeds the configured retriever window",
        },
        "intent_equivalent_utterance_variation": {
            "pairs": len(neighborhood),
            "mean_top8_jaccard": float(np.mean(neighborhood)),
            "note": "pairs are distinct development utterances with the same intent, not generated text",
        },
        "label_surface": {
            "raw_vs_naturalized": "reported in retriever_comparison.json",
            "method": "underscores/camel-case versus deterministic whitespace naturalization",
        },
        "retrieval_score_diagnostic_only": {
            "mean_max_similarity_hit": float(max_similarity[hit].mean()),
            "mean_max_similarity_miss": float(max_similarity[~hit].mean()),
            "mean_margin_hit": float(margins[hit].mean()),
            "mean_margin_miss": float(margins[~hit].mean()),
            "warning": "cosine scores are rankings, not calibrated confidence",
        },
    }


def performance(bundle, artifact_path: str) -> dict:
    embed = embed_fn_from_agent(
        bundle.agent, max_length=int(bundle.router["embed_max_length"]), batch_size=16
    )
    counts = (50, 100, 150, 500, 1000)
    rows = []
    for count in counts:
        candidates = [
            ("candidate_%04d" % index, "candidate %04d account support topic" % index)
            for index in range(count)
        ]
        retriever = load_domain_retriever(
            artifact_path, bundle.checkpoint_hash, embed_fn=embed
        )
        started = time.perf_counter()
        retriever.rank("I need help with my account", candidates, "runtime-%d" % count, k=8)
        uncached_ms = (time.perf_counter() - started) * 1000.0
        cached = []
        for _ in range(10):
            started = time.perf_counter()
            retriever.rank("I need help with my account", candidates, "runtime-%d" % count, k=8)
            cached.append((time.perf_counter() - started) * 1000.0)
        query = np.ones(128, dtype=np.float32)
        matrix = np.ones((count, 128), dtype=np.float32)
        topk = []
        for _ in range(100):
            started = time.perf_counter()
            np.argsort(-(matrix @ query), kind="stable")[:8]
            topk.append((time.perf_counter() - started) * 1000.0)
        cache_bytes = sum(value.nbytes for value in retriever._candidate_cache.values())
        rows.append(
            {
                "candidates": count,
                "uncached_total_ms": uncached_ms,
                "cached_total": latency(cached),
                "topk_exact_only": latency(topk),
                "projected_candidate_cache_bytes": cache_bytes,
                "cache_hits": retriever.cache_hits,
                "cache_misses": retriever.cache_misses,
            }
        )
        print("runtime cardinality", count, flush=True)
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "rss_mb_after": rss_mb(),
        "rows": rows,
        "ann": "not introduced; exact scoring remained sub-millisecond apart from text encoding",
    }


def main() -> None:
    train, development, validation, all_candidates = load_inputs()
    del train, development
    with open(os.path.join(PHASE10, "retriever_comparison.json"), encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "FINALIST_FROZEN_BLIND_NOT_OPENED":
        raise SystemExit("retriever selection is not frozen")
    bundle, manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    embed = embed_fn_from_agent(
        bundle.agent, max_length=int(bundle.router["embed_max_length"]), batch_size=16
    )
    retriever = load_domain_retriever(
        ARTIFACT_PATH, checkpoint_hash=bundle.checkpoint_hash, embed_fn=embed
    )

    texts, text_index = _embedding_texts([], [], validation, all_candidates)
    with open(CACHE_META_PATH, encoding="utf-8") as handle:
        cache_metadata = json.load(handle)
    full_vectors = np.load(CACHE_PATH)["vectors"]
    if cache_metadata["texts_sha256"] == __import__(
        "veya.experimental.domain_retriever", fromlist=["text_sha256"]
    ).text_sha256(texts):
        vectors = full_vectors
    else:
        # The training cache includes training/development text. Reconstruct its
        # sorted text index, then select the validation rows from that source.
        train_rows, dev_rows, _validation_rows, full_candidates = load_inputs()
        full_texts, full_index = _embedding_texts(
            train_rows, dev_rows, validation, full_candidates
        )
        if cache_metadata["texts_sha256"] != __import__(
            "veya.experimental.domain_retriever", fromlist=["text_sha256"]
        ).text_sha256(full_texts):
            raise ValueError("embedding cache text hash mismatch")
        text_index = full_index
        vectors = full_vectors
    def vector(text: str) -> np.ndarray:
        return vectors[text_index[text]]

    validation_ids = {row["candidate_id"] for row in validation}
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in all_candidates}
    candidates = [candidate_by_id[value] for value in sorted(validation_ids)]
    candidate_index = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    cached_query_vectors = np.stack([vector(row["text"]) for row in validation])
    cached_naturalized_vectors = np.stack(
        [vector(candidate["naturalized"]) for candidate in candidates]
    )
    cached_prefixed_vectors = np.stack(
        [vector("intent: " + candidate["naturalized"]) for candidate in candidates]
    )
    query_vectors = embed_chunks(
        embed, [row["text"] for row in validation], "validation queries"
    )
    naturalized_vectors = embed_chunks(
        embed,
        [str(candidate["naturalized"]) for candidate in candidates],
        "naturalized candidates",
    )
    prefixed_vectors = embed_chunks(
        embed,
        ["intent: " + str(candidate["naturalized"]) for candidate in candidates],
        "prefixed candidates",
    )
    embedding_delta = max(
        float(np.max(np.abs(query_vectors - cached_query_vectors))),
        float(np.max(np.abs(naturalized_vectors - cached_naturalized_vectors))),
        float(np.max(np.abs(prefixed_vectors - cached_prefixed_vectors))),
    )
    all_fresh = np.concatenate((query_vectors, naturalized_vectors, prefixed_vectors), axis=0)
    all_cached = np.concatenate(
        (cached_query_vectors, cached_naturalized_vectors, cached_prefixed_vectors), axis=0
    )
    embedding_p99_abs_delta = float(np.percentile(np.abs(all_fresh - all_cached), 99))
    embedding_cosine = np.sum(all_fresh * all_cached, axis=1) / (
        np.linalg.norm(all_fresh, axis=1) * np.linalg.norm(all_cached, axis=1)
    )
    embedding_min_cosine = float(np.min(embedding_cosine))
    print("fresh-process base embedding max delta", embedding_delta, flush=True)
    if embedding_min_cosine < 0.99999:
        raise ValueError("fresh-process base embeddings changed direction beyond tolerance")
    gold = np.asarray([candidate_index[row["candidate_id"]] for row in validation])
    with torch.inference_mode():
        base_scores = (
            bundle.projection(torch.as_tensor(query_vectors, dtype=torch.float32))
            @ bundle.projection(torch.as_tensor(prefixed_vectors, dtype=torch.float32)).T
        ).cpu().numpy()
    new_scores = retriever.project_queries(query_vectors) @ retriever.project_candidates(
        naturalized_vectors
    ).T
    base_metrics, _base_positions, base_order = retrieval_metrics(base_scores, gold)
    new_metrics, _new_positions, new_order = retrieval_metrics(new_scores, gold)
    expected = selection["schema_held_out_validation"]["comparison"][selection["winner"]]
    if abs(new_metrics["recall_at_8"] - expected["recall_at_8"]) > 1e-12:
        raise ValueError("fresh-process Recall@8 does not match selection")
    cached_new_scores = retriever.project_queries(
        cached_query_vectors
    ) @ retriever.project_candidates(cached_naturalized_vectors).T
    cached_new_order = np.argsort(-cached_new_scores, axis=1, kind="stable")
    top32_rank_changes = int(np.any(new_order[:, :32] != cached_new_order[:, :32], axis=1).sum())
    top8_set_changes = int(
        sum(
            set(fresh[:8]) != set(cached[:8])
            for fresh, cached in zip(new_order, cached_new_order)
        )
    )
    max_score_delta = float(np.max(np.abs(new_scores - cached_new_scores)))
    if top8_set_changes:
        raise ValueError("fresh-process top-8 candidate sets changed")

    permutation_trials = 0
    permutation_changes = 0
    rng = random.Random(SEED)
    for row_index in range(64):
        baseline = {candidates[index]["candidate_id"]: float(new_scores[row_index, index]) for index in range(len(candidates))}
        for _ in range(4):
            order = list(range(len(candidates)))
            rng.shuffle(order)
            shuffled_vectors = naturalized_vectors[order]
            scores = retriever.project_queries(query_vectors[row_index : row_index + 1])[0] @ retriever.project_candidates(shuffled_vectors).T
            restored = {candidates[source]["candidate_id"]: float(score) for source, score in zip(order, scores)}
            permutation_trials += 1
            if any(abs(restored[key] - baseline[key]) > 1e-6 for key in baseline):
                permutation_changes += 1

    report = {
        "format": "veya-phase10-validation-v1",
        "status": "RESEARCH_VALIDATED",
        "release_manifest_sha256": file_sha256(MANIFEST_PATH),
        "release": manifest["release"],
        "artifact_sha256": file_sha256(ARTIFACT_PATH),
        "fresh_process": {
            "base_embedding_max_abs_delta": embedding_delta,
            "base_embedding_p99_abs_delta": embedding_p99_abs_delta,
            "base_embedding_min_cosine": embedding_min_cosine,
            "base_embedding_direction_tolerance": 0.99999,
            "retrieval_score_max_abs_delta": max_score_delta,
            "top8_candidate_set_changes": top8_set_changes,
            "top32_rows_changed_from_training_cache": top32_rank_changes,
            "candidate_projection_sha256": hashlib.sha256(
                retriever.project_candidates(naturalized_vectors).tobytes()
            ).hexdigest(),
            "top32_ranks_sha256": hashlib.sha256(new_order[:, :32].tobytes()).hexdigest(),
            "baseline_metrics": base_metrics,
            "phase10_metrics": new_metrics,
            "matches_selection": True,
        },
        "permutation": {
            "trials": permutation_trials,
            "score_map_changes": permutation_changes,
        },
        "end_to_end": evaluate_e2e(bundle, validation, candidates, base_order, new_order),
        "robustness": robustness(
            retriever,
            embed,
            validation,
            candidates,
            query_vectors,
            naturalized_vectors,
            gold,
        ),
        "performance": performance(bundle, ARTIFACT_PATH),
        "unchanged": {
            "production_manifest": True,
            "production_retriever": True,
            "scorer": "base_decision_head",
            "calibration": True,
            "automation_policy": True,
            "serving_default": "current",
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"fresh_process": report["fresh_process"], "end_to_end": report["end_to_end"], "performance": report["performance"]}, indent=2))


if __name__ == "__main__":
    main()
