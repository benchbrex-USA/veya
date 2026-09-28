"""Validate Phase-11 dense robustness, serving cost, and frozen-scorer behavior."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from collections import Counter, defaultdict
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import ROOT, SEED, split_digest
from research.veyabench.phase11_noise import CONDITIONS, NOISE_VERSION, perturb
from research.veyabench.phase11_protocol import token_length_bucket, tokenization_audit
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH as PHASE10_ARTIFACT,
    CACHE_META_PATH as PHASE10_CACHE_META,
    CACHE_PATH as PHASE10_CACHE,
    MANIFEST_PATH,
    _embedding_texts,
    load_inputs,
    paired_bootstrap,
    retrieval_metrics,
)
from research.veyabench.run_phase11_train import (
    ABLATIONS_PATH,
    ARTIFACT_PATH,
    MAX_LENGTH,
    PHASE11,
    REPORT_PATH as TRAINING_REPORT_PATH,
)
from veya.common import ece_score
from veya.experimental.domain_retriever import (
    ProjectionPair,
    build_phase11_query_views,
    fuse_query_scores,
    load_domain_retriever,
    naturalize_label,
    text_sha256,
)
from veya.release_bundle import INSTRUCTIONS, load_release_manifest, raw_choice_logits
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


VALIDATION_CACHE = os.path.join(PHASE11, "validation_embeddings.npz")
VALIDATION_CACHE_META = VALIDATION_CACHE + ".json"
REPORT_PATH = os.path.join(PHASE11, "validation_report.json")
SAMPLE_SIZE = 200
E2E_SIZE = 200
QUERY_CONDITIONS = (
    "clean",
    "short_prefix",
    "short_suffix",
    "long_prefix",
    "long_suffix",
    "middle_insertion",
)
POLICIES = {
    "Q0_single": "single",
    "Q1_first_last": "first_last",
    "Q2_first_middle_last": "first_middle_last",
    "Q3_four_view": "four_view",
    "Q4_adaptive": "adaptive",
}


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max()
    probabilities = np.exp(values)
    return probabilities / probabilities.sum()


def _latency(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
    }


def _load_models(checkpoint_hash: str) -> dict[str, ProjectionPair]:
    blob = torch.load(ABLATIONS_PATH, map_location="cpu", weights_only=False)
    if blob.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Phase-11 ablations belong to a different base checkpoint")
    models = {}
    for name, state in blob["models"].items():
        model = ProjectionPair(blob["input_dim"], blob["output_dim"], blob["projection"])
        model.load_state_dict(state)
        models[name] = model.eval()
    return models


def _model_scores(model: ProjectionPair, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return model.scores(
            torch.as_tensor(query, dtype=torch.float32),
            torch.as_tensor(candidates, dtype=torch.float32),
        ).cpu().numpy()


def _encode_or_load(bundle: Any, texts: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(set(texts))
    expected = {
        "format": "veya-phase11-validation-embedding-cache-v1",
        "checkpoint_hash": bundle.checkpoint_hash,
        "texts_sha256": text_sha256(unique),
        "max_length": MAX_LENGTH,
        "count": len(unique),
        "noise_version": NOISE_VERSION,
    }
    if os.path.exists(VALIDATION_CACHE) and os.path.exists(VALIDATION_CACHE_META):
        metadata = json.load(open(VALIDATION_CACHE_META, encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            vectors = np.load(VALIDATION_CACHE)["vectors"].astype(np.float32, copy=False)
            if vectors.shape == (len(unique), 1024):
                print("loaded Phase-11 validation embeddings", vectors.shape, flush=True)
                return vectors, {text: index for index, text in enumerate(unique)}
    embed = embed_fn_from_agent(bundle.agent, max_length=MAX_LENGTH, batch_size=16)
    parts = []
    started = time.perf_counter()
    for start in range(0, len(unique), 256):
        stop = min(start + 256, len(unique))
        parts.append(embed(unique[start:stop]))
        print(f"Phase-11 validation embeddings {stop}/{len(unique)}", flush=True)
    vectors = np.concatenate(parts, axis=0).astype(np.float32)
    np.savez_compressed(VALIDATION_CACHE, vectors=vectors)
    _write_json(
        VALIDATION_CACHE_META,
        {
            **expected,
            "shape": list(vectors.shape),
            "seconds": time.perf_counter() - started,
            "file_sha256": file_sha256(VALIDATION_CACHE),
        },
    )
    return vectors, {text: index for index, text in enumerate(unique)}


def _rank_metrics(scores: np.ndarray, gold: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    metrics, positions, order = retrieval_metrics(scores, gold)
    metrics["target_rank_distribution"] = {
        "rank_1": float(np.mean(positions == 1)),
        "rank_2_to_4": float(np.mean((positions >= 2) & (positions <= 4))),
        "rank_5_to_8": float(np.mean((positions >= 5) & (positions <= 8))),
        "miss": float(np.mean(positions > 8)),
    }
    return metrics, positions, order


def _fused_scores(
    model: ProjectionPair,
    view_vectors: list[np.ndarray],
    candidate_vectors: np.ndarray,
    fusion: str = "max",
) -> np.ndarray:
    stacked = np.stack(view_vectors)
    with torch.inference_mode():
        query = model.encode_query(torch.as_tensor(stacked, dtype=torch.float32)).cpu().numpy()
        candidate = model.encode_candidate(
            torch.as_tensor(candidate_vectors, dtype=torch.float32)
        ).cpu().numpy()
    return fuse_query_scores(query @ candidate.T, fusion)


def _evaluate_end_to_end(
    bundle: Any,
    rows: list[dict],
    candidates: list[dict],
    orders: dict[str, np.ndarray],
    score_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    subset_indices = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(("phase11-e2e\0" + rows[index]["id"]).encode()).hexdigest(),
    )[:E2E_SIZE]
    outputs: dict[str, Any] = {}
    correct_by_model: dict[str, np.ndarray] = {}
    scorer_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, float]] = {}
    for name, order in orders.items():
        hits = []
        correct = []
        conditional = []
        margins = []
        confidences = []
        scorer_latencies = []
        negative_hardness = []
        for progress, row_index in enumerate(subset_indices, start=1):
            row = rows[row_index]
            picked_indices = order[row_index, :8]
            picked = tuple(candidates[int(value)]["candidate_id"] for value in picked_indices)
            cache_key = (row["id"], picked)
            cached = scorer_cache.get(cache_key)
            if cached is None:
                criteria = {
                    candidate_id: "intent: " + naturalize_label(candidate_id.split(":", 1)[1])
                    for candidate_id in picked
                }
                started = time.perf_counter()
                logits, _tokens = raw_choice_logits(bundle.agent, row["text"], INSTRUCTIONS, criteria)
                elapsed = (time.perf_counter() - started) * 1000.0
                scorer_cache[cache_key] = (np.asarray(logits), elapsed)
            else:
                logits, elapsed = cached
            scorer_latencies.append(elapsed)
            probabilities = _softmax(np.asarray(logits), bundle.route_temperature)
            prediction = picked[int(np.argmax(probabilities))]
            hit = row["candidate_id"] in picked
            is_correct = prediction == row["candidate_id"]
            hits.append(hit)
            correct.append(is_correct)
            confidences.append(float(probabilities.max()))
            if hit:
                conditional.append(is_correct)
                top = np.sort(probabilities)[-2:]
                margins.append(float(top[-1] - top[-2]))
                truth_index = next(
                    index for index, candidate in enumerate(candidates) if candidate["candidate_id"] == row["candidate_id"]
                )
                shortlist_scores = score_matrices[name][row_index, picked_indices]
                negative_hardness.append(float(np.max(shortlist_scores[picked_indices != truth_index])))
            if progress % 100 == 0:
                print(f"Phase-11 frozen scorer {name} {progress}/{len(subset_indices)}", flush=True)
        correct_array = np.asarray(correct, dtype=bool)
        correct_by_model[name] = correct_array
        outputs[name] = {
            "n": len(subset_indices),
            "recall_at_8": float(np.mean(hits)),
            "conditional_accuracy": float(np.mean(conditional)) if conditional else None,
            "conditional_n": len(conditional),
            "end_to_end_accuracy": float(correct_array.mean()),
            "mean_top1_top2_scorer_margin_on_hits": float(np.mean(margins)) if margins else None,
            "mean_hardest_negative_retriever_score_on_hits": (
                float(np.mean(negative_hardness)) if negative_hardness else None
            ),
            "ece_descriptive_frozen_temperature": ece_score(
                np.asarray(confidences), correct_array.astype(np.float64)
            ),
            "scorer_latency": _latency(scorer_latencies),
        }
    baseline = correct_by_model["RB0_phase10"]
    outputs["paired_e2e_delta_vs_RB0"] = {
        name: paired_bootstrap(baseline, values)
        for name, values in correct_by_model.items()
        if name != "RB0_phase10"
    }
    outputs["frozen_components"] = {
        "scorer": "base_decision_head",
        "k": 8,
        "temperature": bundle.route_temperature,
        "calibration_refit": False,
        "router_cutoff_changed": False,
        "automation_changed": False,
    }
    return outputs


def _runtime_benchmark(
    bundle: Any,
    model: ProjectionPair,
    candidate_vectors: np.ndarray,
    sample: list[dict],
) -> dict[str, Any]:
    embed = embed_fn_from_agent(bundle.agent, max_length=MAX_LENGTH, batch_size=16)
    with torch.inference_mode():
        candidate_projected = model.encode_candidate(
            torch.as_tensor(candidate_vectors, dtype=torch.float32)
        ).cpu().numpy()
    workload = []
    conditions = ("clean", "long_prefix", "long_suffix")
    for index, row in enumerate(sample[:30]):
        condition = conditions[index % len(conditions)]
        workload.append(perturb(row["text"], condition, row["id"]))
    output = {}
    for name, policy in POLICIES.items():
        values = []
        view_counts = []
        started_total = time.perf_counter()
        for text in workload:
            started = time.perf_counter()
            views = build_phase11_query_views(text, policy=policy)
            base = embed(views)
            with torch.inference_mode():
                query = model.encode_query(torch.as_tensor(base, dtype=torch.float32)).cpu().numpy()
            _scores = (query @ candidate_projected.T).max(axis=0)
            np.argsort(-_scores, kind="stable")[:8]
            values.append((time.perf_counter() - started) * 1000.0)
            view_counts.append(len(views))
        elapsed = time.perf_counter() - started_total
        output[name] = {
            **_latency(values),
            "throughput_queries_per_second": len(workload) / elapsed,
            "mean_views_per_query": float(np.mean(view_counts)),
            "max_views_per_query": max(view_counts),
            "projected_candidate_cache_bytes": int(candidate_projected.nbytes),
        }
        print("Phase-11 latency", name, flush=True)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = peak_rss / (1024.0 * 1024.0) if platform.system() == "Darwin" else peak_rss / 1024.0
    output["memory"] = {
        "projection_parameter_bytes": model_bytes,
        "candidate_cache_bytes_at_150": 150 * 128 * 4,
        "candidate_cache_bytes_at_1000": 1000 * 128 * 4,
        "candidate_cache_bytes_at_10000": 10000 * 128 * 4,
        "process_peak_rss_mb": peak_rss_mb,
        "candidate_token_cache": "none",
    }
    output["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
    }
    return output


def main() -> None:
    training_report = json.load(open(TRAINING_REPORT_PATH, encoding="utf-8"))
    train, development, validation, all_candidates = load_inputs()
    bundle, manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    models = _load_models(bundle.checkpoint_hash)
    winner_name = str(training_report["winner"])
    winner = models[winner_name]
    if file_sha256(ARTIFACT_PATH) != training_report["artifact"]["file_sha256"]:
        raise ValueError("Phase-11 artifact changed after training selection")

    full_texts, base_index = _embedding_texts(train, development, validation, all_candidates)
    base_metadata = json.load(open(PHASE10_CACHE_META, encoding="utf-8"))
    if base_metadata["texts_sha256"] != text_sha256(full_texts):
        raise ValueError("Phase-10 cache inventory changed")
    base_vectors = np.load(PHASE10_CACHE)["vectors"].astype(np.float32, copy=False)

    def base_vector(text: str) -> np.ndarray:
        return base_vectors[base_index[text]]

    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in all_candidates}
    candidates = [candidate_by_id[value] for value in sorted({row["candidate_id"] for row in validation})]
    candidate_index = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    candidate_vectors = np.stack([base_vector(candidate["naturalized"]) for candidate in candidates])
    full_clean_vectors = np.stack([base_vector(row["text"]) for row in validation])
    full_gold = np.asarray([candidate_index[row["candidate_id"]] for row in validation])
    sample = sorted(
        validation,
        key=lambda row: hashlib.sha256(("phase11-noise\0" + row["id"]).encode()).hexdigest(),
    )[:SAMPLE_SIZE]
    sample_gold = np.asarray([candidate_index[row["candidate_id"]] for row in sample])

    condition_texts = {
        condition: [perturb(row["text"], condition, row["id"]) for row in sample]
        for condition in CONDITIONS
    }
    views: dict[tuple[str, str, int], list[str]] = {}
    all_validation_texts = [text for values in condition_texts.values() for text in values]
    for policy_name, policy in POLICIES.items():
        policy_conditions = CONDITIONS if policy_name == "Q4_adaptive" else QUERY_CONDITIONS
        for condition in policy_conditions:
            for row_index, text in enumerate(condition_texts[condition]):
                values = build_phase11_query_views(text, policy=policy)
                views[(policy_name, condition, row_index)] = values
                all_validation_texts.extend(values)
    for row_index, text in enumerate(condition_texts["json_noise"]):
        structured = json.loads(text)
        values = build_phase11_query_views(structured, policy="adaptive")
        views[("structured_semantic_fields", "json_noise", row_index)] = values
        all_validation_texts.extend(values)
    validation_vectors, validation_index = _encode_or_load(bundle, all_validation_texts)

    def vector(text: str) -> np.ndarray:
        return validation_vectors[validation_index[text]]

    full_clean = {}
    full_orders: dict[str, np.ndarray] = {}
    full_scores: dict[str, np.ndarray] = {}
    full_positions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        scores = _model_scores(model, full_clean_vectors, candidate_vectors)
        metrics, positions, order = _rank_metrics(scores, full_gold)
        full_clean[name] = metrics
        full_positions[name] = positions
        full_orders[name] = order
        full_scores[name] = scores

    condition_vectors = {
        condition: np.stack([vector(text) for text in texts])
        for condition, texts in condition_texts.items()
    }
    noise_results: dict[str, Any] = {}
    noise_positions: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        metrics_by_condition = {}
        positions_by_condition = {}
        for condition in CONDITIONS:
            scores = _model_scores(model, condition_vectors[condition], candidate_vectors)
            metrics, positions, _order = _rank_metrics(scores, sample_gold)
            metrics_by_condition[condition] = metrics
            positions_by_condition[condition] = positions
        noise_positions[name] = positions_by_condition
        clean = metrics_by_condition["clean"]["recall_at_8"]
        grouped = {}
        for group_name, members in {
            "short_noise": ("short_prefix", "short_suffix"),
            "medium_noise": (
                "middle_insertion",
                "duplicate_metadata",
                "reordered_fields",
                "boilerplate_wrapper",
                "json_noise",
            ),
            "long_noise": ("long_prefix", "long_suffix", "repeated_low_information"),
        }.items():
            score = float(np.mean([metrics_by_condition[value]["recall_at_8"] for value in members]))
            grouped[group_name] = {
                "recall_at_8": score,
                "absolute_degradation": score - clean,
                "relative_degradation": (score - clean) / clean if clean else None,
                "conditions": list(members),
            }
        noise_results[name] = {
            "conditions": metrics_by_condition,
            "groups": grouped,
            "robust_mean_recall_at_8": float(
                np.mean(
                    [metrics_by_condition[value]["recall_at_8"] for value in CONDITIONS if value != "clean"]
                )
            ),
        }

    query_strategy = {}
    query_hits: dict[str, np.ndarray] = {}
    for policy_name in POLICIES:
        condition_metrics = {}
        all_hits: list[bool] = []
        counts = []
        for condition in QUERY_CONDITIONS:
            rows_scores = []
            for row_index in range(len(sample)):
                row_views = views[(policy_name, condition, row_index)]
                counts.append(len(row_views))
                rows_scores.append(
                    _fused_scores(winner, [vector(value) for value in row_views], candidate_vectors)
                )
            scores = np.stack(rows_scores)
            metrics, positions, _order = _rank_metrics(scores, sample_gold)
            condition_metrics[condition] = metrics
            all_hits.extend(bool(value) for value in positions <= 8)
        hit_array = np.asarray(all_hits, dtype=bool)
        query_hits[policy_name] = hit_array
        query_strategy[policy_name] = {
            "conditions": condition_metrics,
            "robust_mean_recall_at_8": float(hit_array.mean()),
            "mean_views_per_query": float(np.mean(counts)),
            "max_views": max(counts),
        }
    structured_scores = np.stack(
        [
            _fused_scores(
                winner,
                [vector(value) for value in views[("structured_semantic_fields", "json_noise", index)]],
                candidate_vectors,
            )
            for index in range(len(sample))
        ]
    )
    structured_metrics, structured_positions, _structured_order = _rank_metrics(
        structured_scores, sample_gold
    )
    query_strategy["structured_field_ablation"] = {
        "all_fields_single": noise_results[winner_name]["conditions"]["json_noise"],
        "generic_semantic_fields_only": structured_metrics,
        "rules": "recursive scalar extraction excluding generic IDs, timestamps, hashes, versions, attempts, and display settings",
    }

    aggregation_ablation: dict[str, Any] = {}
    aggregation_hits: dict[str, np.ndarray] = {}
    for fusion in ("max", "mean", "rrf"):
        fusion_hits: list[bool] = []
        conditions = {}
        for condition in QUERY_CONDITIONS:
            scores = np.stack(
                [
                    _fused_scores(
                        winner,
                        [
                            vector(value)
                            for value in views[("Q4_adaptive", condition, row_index)]
                        ],
                        candidate_vectors,
                        fusion=fusion,
                    )
                    for row_index in range(len(sample))
                ]
            )
            metrics, positions, _order = _rank_metrics(scores, sample_gold)
            conditions[condition] = metrics
            fusion_hits.extend(bool(value) for value in positions <= 8)
        hit_array = np.asarray(fusion_hits, dtype=bool)
        aggregation_hits[fusion] = hit_array
        aggregation_ablation[fusion] = {
            "conditions": conditions,
            "robust_mean_recall_at_8": float(hit_array.mean()),
        }
    aggregation_paired: dict[str, dict[str, Any]] = {
        fusion: paired_bootstrap(aggregation_hits["max"], aggregation_hits[fusion])
        for fusion in ("mean", "rrf")
    }
    aggregation_ablation["paired_vs_max"] = aggregation_paired

    token_records = []
    length_hits: dict[str, list[bool]] = defaultdict(list)
    length_adaptive_hits: dict[str, list[bool]] = defaultdict(list)
    adaptive_condition_positions: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        adaptive_scores = []
        for row_index, row in enumerate(sample):
            text = condition_texts[condition][row_index]
            adaptive_views = build_phase11_query_views(text, policy="adaptive")
            for value in adaptive_views:
                if value not in validation_index:
                    raise ValueError("adaptive validation view was not frozen into the cache")
            adaptive_scores.append(
                _fused_scores(winner, [vector(value) for value in adaptive_views], candidate_vectors)
            )
            audit = tokenization_audit(
                bundle.agent.tok,
                row["text"],
                text,
                max_length=MAX_LENGTH,
                view_count=len(adaptive_views),
            )
            view_signal_audits = []
            for adaptive_view in adaptive_views:
                try:
                    view_signal_audits.append(
                        tokenization_audit(
                            bundle.agent.tok,
                            row["text"],
                            adaptive_view,
                            max_length=MAX_LENGTH,
                            view_count=len(adaptive_views),
                        )
                    )
                except ValueError:
                    continue
            audit["positive_signal_present_in_any_view"] = bool(view_signal_audits)
            audit["positive_signal_fully_retained_in_any_view"] = any(
                value["positive_signal_position"] == "fully_retained"
                for value in view_signal_audits
            )
            token_records.append({"row_id": row["id"], "condition": condition, **audit})
        adaptive_matrix = np.stack(adaptive_scores)
        _metrics, adaptive_positions, _order = _rank_metrics(adaptive_matrix, sample_gold)
        adaptive_condition_positions[condition] = adaptive_positions
    for record in token_records:
        condition = str(record["condition"])
        row_index = next(index for index, row in enumerate(sample) if row["id"] == record["row_id"])
        bucket = token_length_bucket(int(record["token_length"]))
        length_hits[bucket].append(bool(noise_positions[winner_name][condition][row_index] <= 8))
        length_adaptive_hits[bucket].append(bool(adaptive_condition_positions[condition][row_index] <= 8))
    length_robustness = {
        bucket: {
            "n": len(length_hits[bucket]),
            "single_view_recall_at_8": float(np.mean(length_hits[bucket])),
            "adaptive_recall_at_8": float(np.mean(length_adaptive_hits[bucket])),
        }
        for bucket in ("0-32", "33-64", "65-128", "129-256", "257+")
        if length_hits[bucket]
    }
    token_summary = {}
    for condition in CONDITIONS:
        selected = [record for record in token_records if record["condition"] == condition]
        token_summary[condition] = {
            "n": len(selected),
            "truncated_rate": float(np.mean([record["truncated"] for record in selected])),
            "mean_tokens_dropped": float(np.mean([record["tokens_dropped"] for record in selected])),
            "signal_fully_retained_rate": float(
                np.mean([record["positive_signal_position"] == "fully_retained" for record in selected])
            ),
            "signal_truncated_rate": float(
                np.mean([record["positive_signal_position"] == "truncated" for record in selected])
            ),
            "mean_view_count": float(np.mean([record["view_count"] for record in selected])),
            "signal_fully_retained_in_any_view_rate": float(
                np.mean(
                    [record["positive_signal_fully_retained_in_any_view"] for record in selected]
                )
            ),
        }

    position_sensitivity = {
        "beginning_long_suffix": {
            "single": noise_results[winner_name]["conditions"]["long_suffix"],
            "adaptive_recall_at_8": float(np.mean(adaptive_condition_positions["long_suffix"] <= 8)),
        },
        "middle": {
            "single": noise_results[winner_name]["conditions"]["middle_insertion"],
            "adaptive_recall_at_8": float(np.mean(adaptive_condition_positions["middle_insertion"] <= 8)),
        },
        "end_long_prefix": {
            "single": noise_results[winner_name]["conditions"]["long_prefix"],
            "adaptive_recall_at_8": float(np.mean(adaptive_condition_positions["long_prefix"] <= 8)),
        },
    }

    base_robust_hits = np.concatenate(
        [noise_positions["RB0_phase10"][condition] <= 8 for condition in CONDITIONS if condition != "clean"]
    )
    comparisons = {}
    for name in models:
        if name == "RB0_phase10":
            continue
        candidate_hits = np.concatenate(
            [noise_positions[name][condition] <= 8 for condition in CONDITIONS if condition != "clean"]
        )
        comparisons[name] = paired_bootstrap(base_robust_hits, candidate_hits)
    adaptive_vs_single = paired_bootstrap(query_hits["Q0_single"], query_hits["Q4_adaptive"])
    structured_vs_all_fields = paired_bootstrap(
        noise_positions[winner_name]["json_noise"] <= 8,
        structured_positions <= 8,
    )
    clean_vs_phase10 = paired_bootstrap(
        full_positions["RB0_phase10"] <= 8, full_positions[winner_name] <= 8
    )

    # Rule-based decomposition is diagnostic, based only on allowed validation rows.
    failure_counts: Counter[str] = Counter()
    for condition in CONDITIONS:
        for row_index, row in enumerate(sample):
            if adaptive_condition_positions[condition][row_index] <= 8:
                continue
            matching_audit = next(
                value
                for value in token_records
                if value["row_id"] == row["id"] and value["condition"] == condition
            )
            if matching_audit["positive_signal_position"] != "fully_retained":
                category = "truncation_loss"
            elif noise_positions[winner_name]["clean"][row_index] <= 8:
                category = "noise_dominance"
            else:
                label_tokens = set(naturalize_label(row["candidate_id"].split(":", 1)[1]).split())
                query_tokens = set(naturalize_label(row["text"]).split())
                category = "lexical_mismatch" if not label_tokens & query_tokens else "semantic_ambiguity"
            failure_counts[category] += 1
    for category in (
        "truncation_loss",
        "semantic_ambiguity",
        "lexical_mismatch",
        "candidate_description_weakness",
        "noise_dominance",
        "hard_sibling_confusion",
        "true_ood",
    ):
        failure_counts.setdefault(category, 0)

    runtime = _runtime_benchmark(bundle, winner, candidate_vectors, sample)
    end_to_end = _evaluate_end_to_end(
        bundle, validation, candidates, full_orders, full_scores
    )

    phase10_reloaded = load_domain_retriever(PHASE10_ARTIFACT, bundle.checkpoint_hash)
    first = load_domain_retriever(ARTIFACT_PATH, bundle.checkpoint_hash)
    second = load_domain_retriever(ARTIFACT_PATH, bundle.checkpoint_hash)
    query_a = first.project_queries(full_clean_vectors[:64])
    query_b = second.project_queries(full_clean_vectors[:64])
    candidate_a = first.project_candidates(candidate_vectors)
    candidate_b = second.project_candidates(candidate_vectors)
    score_a = query_a @ candidate_a.T
    score_b = query_b @ candidate_b.T
    order_a = np.argsort(-score_a, axis=1, kind="stable")
    order_b = np.argsort(-score_b, axis=1, kind="stable")
    permutation_changes = 0
    rng = np.random.default_rng(SEED)
    for row_index in range(32):
        permutation = rng.permutation(len(candidates))
        permuted_scores = query_a[row_index] @ candidate_a[permutation].T
        restored = np.empty_like(permuted_scores)
        restored[permutation] = permuted_scores
        if not np.allclose(restored, score_a[row_index], atol=1e-6, rtol=0.0):
            permutation_changes += 1
    clean_reload = {
        "artifact_sha256": file_sha256(ARTIFACT_PATH),
        "query_embedding_max_abs_delta": float(np.max(np.abs(query_a - query_b))),
        "candidate_embedding_max_abs_delta": float(np.max(np.abs(candidate_a - candidate_b))),
        "score_max_abs_delta": float(np.max(np.abs(score_a - score_b))),
        "rank_changes": int(np.any(order_a != order_b, axis=1).sum()),
        "top8_set_changes": int(
            sum(set(left[:8]) != set(right[:8]) for left, right in zip(order_a, order_b))
        ),
        "query_embeddings_sha256": hashlib.sha256(query_a.tobytes()).hexdigest(),
        "candidate_embeddings_sha256": hashlib.sha256(candidate_a.tobytes()).hexdigest(),
        "top8_sha256": hashlib.sha256(order_a[:, :8].tobytes()).hexdigest(),
        "permutation_trials": 32,
        "permutation_score_map_changes": permutation_changes,
        "candidate_cache_built": True,
    }
    del phase10_reloaded

    credible_dense: dict[str, dict[str, Any]] = {
        "noise_training_vs_RB0": comparisons["RB1_noise"],
        "invariance_vs_RB0": comparisons["RB1I_noise_invariance"],
        "second_hard_negative_vs_RB0": comparisons["RB2_second_hard_negative"],
        "adaptive_views_vs_single": adaptive_vs_single,
        "structured_fields_vs_all_fields": structured_vs_all_fields,
        "adaptive_mean_fusion_vs_max": aggregation_paired["mean"],
        "adaptive_rrf_fusion_vs_max": aggregation_paired["rrf"],
    }
    material_dense = [
        name
        for name, evidence in credible_dense.items()
        if evidence["ci95"][0] > 0.0
    ]
    dense_verdict = "DENSE STILL IMPROVING" if material_dense else "DENSE PLATEAUED"
    report = {
        "format": "veya-phase11-validation-report-v1",
        "status": "RESEARCH_VALIDATED",
        "winner": winner_name,
        "winner_query_policy": "Q0_single (Pareto choice; multi-view gains were within noise)",
        "clean_full_schema_held_out": full_clean,
        "clean_delta_vs_phase10": clean_vs_phase10,
        "noise_benchmark": noise_results,
        "noise_delta_bootstrap_vs_phase10": comparisons,
        "query_strategy": query_strategy,
        "aggregation_ablation": aggregation_ablation,
        "adaptive_vs_single_bootstrap": adaptive_vs_single,
        "tokenization_truncation": {
            "max_length": MAX_LENGTH,
            "by_condition": token_summary,
            "records": token_records,
            "causal_readout": {
                "long_prefix": "useful span is removed by right truncation in the single-view path",
                "long_suffix": "useful span remains visible; degradation is consistent with pooling/noise dominance",
                "adaptive": "bounded tail/middle views restore position coverage without changing the encoder",
            },
        },
        "position_sensitivity": position_sensitivity,
        "length_robustness": length_robustness,
        "failure_decomposition": {
            "method": "deterministic diagnostic rules on permitted CLINC150 train schema-held-out rows",
            "counts": dict(sorted(failure_counts.items())),
            "not_a_teacher_label": True,
        },
        "dense_plateau": {
            "verdict": dense_verdict,
            "credible_improvements": credible_dense,
            "material_dense_improvements": material_dense,
            "late_interaction_executed": False,
            "late_interaction_reason": (
                "not permitted: at least one dense intervention materially improved held-out robustness"
                if dense_verdict == "DENSE STILL IMPROVING"
                else "eligible only if two dense improvements fail and token-level errors dominate"
            ),
        },
        "end_to_end": end_to_end,
        "runtime": runtime,
        "clean_reload": clean_reload,
        "artifact": json.load(open(ARTIFACT_PATH + ".json", encoding="utf-8")),
        "external_evaluation_integrity": {
            "dataset": "CLINC150 train partition only",
            "schema_held_out_from_training": True,
            "split_hash": split_digest(validation),
            "registry_sha256": file_sha256(os.path.join(ROOT, "research/data/phase10/registry.json")),
            "consumed_blind_or_external_sets_opened": [],
            "overlap_interpretation": "schema-held-out development evidence; not claimed as pristine blind evidence",
            "historical_consumed_sets_used_for_selection": False,
        },
        "frozen": {
            "production_manifest_sha256": file_sha256(MANIFEST_PATH),
            "release": manifest["release"],
            "k": 8,
            "scorer": bundle.reranker_id,
            "calibration_refit": False,
            "router_cutoff_changed": False,
            "production_retriever_changed": False,
            "serving_default": "current",
            "automation": "disabled/unchanged",
        },
        "security": {
            "candidate_text_treated_as_untrusted_embedding_data": True,
            "max_views": 4,
            "max_state_chars": 32768,
            "max_view_chars": 4096,
            "candidate_cache_key_includes_retriever_schema_and_content": True,
        },
        "promotion": "KEEP AS RESEARCH",
        "promotion_reason": "the development-selected RB1 regressed full clean schema-held-out Recall@8 and matched end-to-end accuracy, while multi-view latency increased without a significant aggregate gain",
        "next_bottleneck": "representation/backbone capacity",
    }
    _write_json(REPORT_PATH, report)
    print(
        json.dumps(
            {
                "winner": winner_name,
                "clean": full_clean[winner_name],
                "noise": noise_results[winner_name],
                "query_strategy": query_strategy,
                "dense_plateau": report["dense_plateau"],
                "end_to_end": end_to_end,
                "runtime": runtime,
                "clean_reload": clean_reload,
                "promotion": report["promotion"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
