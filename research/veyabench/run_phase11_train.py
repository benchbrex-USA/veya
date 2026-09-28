"""Train Phase-11 dense robustness ablations without opening consumed blind data."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import ROOT, SEED, split_digest
from research.veyabench.phase11_noise import (
    CONDITIONS,
    NOISE_VERSION,
    audit_record,
    training_conditions,
    validate_augmentation,
)
from research.veyabench.phase11_protocol import (
    HARD_NEGATIVE_VERSION,
    mine_hard_negatives_with_lineage,
    tokenization_audit,
    validate_hard_negative_lineage,
)
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH as PHASE10_ARTIFACT,
    CACHE_META_PATH as PHASE10_CACHE_META,
    CACHE_PATH as PHASE10_CACHE,
    EPISODE_CARDINALITIES,
    LEARNING_RATE,
    MANIFEST_PATH,
    _embedding_texts,
    git_commit,
    load_inputs,
    model_scores,
    retrieval_metrics,
)
from veya.experimental.domain_retriever import (
    ProjectionPair,
    load_domain_retriever,
    naturalize_label,
    save_domain_retriever,
    text_sha256,
)
from veya.release_bundle import load_release_manifest
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


PHASE11 = os.path.join(ROOT, "artifacts", "experiments", "phase11")
ARTIFACT_PATH = os.path.join(PHASE11, "phase11_noise_robust_k8.pt")
ABLATIONS_PATH = os.path.join(PHASE11, "ablation_models.pt")
AUGMENTED_CACHE = os.path.join(PHASE11, "augmented_embeddings.npz")
AUGMENTED_CACHE_META = AUGMENTED_CACHE + ".json"
REPORT_PATH = os.path.join(PHASE11, "training_report.json")
LINEAGE_PATH = os.path.join(PHASE11, "hard_negatives_v2.json")
AUDIT_PATH = os.path.join(PHASE11, "tokenization_audit_train.json")
SEMANTIC_AUDIT_PATH = os.path.join(PHASE11, "augmentation_audit.json")
STEPS = 600
HARD_STEPS = 300
INVARIANCE_WEIGHT = 0.05
MAX_LENGTH = 96


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _base_vector_index(
    train: list[dict], development: list[dict], validation: list[dict], candidates: list[dict]
) -> tuple[np.ndarray, dict[str, int]]:
    texts, index = _embedding_texts(train, development, validation, candidates)
    metadata = json.load(open(PHASE10_CACHE_META, encoding="utf-8"))
    if metadata["texts_sha256"] != text_sha256(texts):
        raise ValueError("Phase-10 base embedding cache does not match its frozen text inventory")
    vectors = np.load(PHASE10_CACHE)["vectors"].astype(np.float32, copy=False)
    if vectors.shape != (len(texts), 1024):
        raise ValueError("unexpected Phase-10 embedding cache shape")
    return vectors, index


def _encode_or_load(bundle: Any, texts: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(set(texts))
    expected = {
        "format": "veya-phase11-augmented-embedding-cache-v1",
        "checkpoint_hash": bundle.checkpoint_hash,
        "texts_sha256": text_sha256(unique),
        "max_length": MAX_LENGTH,
        "count": len(unique),
        "noise_version": NOISE_VERSION,
    }
    if os.path.exists(AUGMENTED_CACHE) and os.path.exists(AUGMENTED_CACHE_META):
        metadata = json.load(open(AUGMENTED_CACHE_META, encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            vectors = np.load(AUGMENTED_CACHE)["vectors"].astype(np.float32, copy=False)
            if vectors.shape == (len(unique), 1024):
                print("loaded Phase-11 augmented embeddings", vectors.shape, flush=True)
                return vectors, {text: index for index, text in enumerate(unique)}
    embed = embed_fn_from_agent(bundle.agent, max_length=MAX_LENGTH, batch_size=16)
    parts: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(unique), 256):
        stop = min(start + 256, len(unique))
        parts.append(embed(unique[start:stop]))
        print(f"Phase-11 augmented embeddings {stop}/{len(unique)}", flush=True)
    vectors = np.concatenate(parts, axis=0).astype(np.float32)
    np.savez_compressed(AUGMENTED_CACHE, vectors=vectors)
    _write_json(
        AUGMENTED_CACHE_META,
        {
            **expected,
            "shape": list(vectors.shape),
            "seconds": time.perf_counter() - started,
            "file_sha256": file_sha256(AUGMENTED_CACHE),
        },
    )
    return vectors, {text: index for index, text in enumerate(unique)}


def _copy_projection(model: ProjectionPair) -> ProjectionPair:
    result = ProjectionPair(model.input_dim, model.output_dim, model.projection)
    result.load_state_dict(copy.deepcopy(model.state_dict()))
    return result


def train_robust_model(
    model: ProjectionPair,
    rows: list[dict],
    variant_vectors: np.ndarray,
    candidates: list[dict],
    candidate_vectors: np.ndarray,
    *,
    steps: int,
    invariance_weight: float,
    hard_negatives: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Run a deterministic clean-to-noisy curriculum over frozen vectors."""
    torch.manual_seed(SEED)
    rng = random.Random(SEED + (37 if hard_negatives else 11) + int(invariance_weight * 1000))
    by_schema: dict[str, list[int]] = defaultdict(list)
    schema_candidates: dict[str, list[int]] = defaultdict(list)
    candidate_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    for index, row in enumerate(rows):
        by_schema[row["schema"]].append(index)
    for index, candidate in enumerate(candidates):
        schema_candidates[candidate["schema"]].append(index)
    valid_schemas = sorted(schema for schema, values in schema_candidates.items() if len(values) >= 2)
    normalized = [naturalize_label(str(row["raw"])) for row in candidates]
    query_tensor = torch.as_tensor(variant_vectors, dtype=torch.float32)
    candidate_tensor = torch.as_tensor(candidate_vectors, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    losses: list[float] = []
    ce_losses: list[float] = []
    invariance_losses: list[float] = []
    variant_counts: Counter[int] = Counter()
    model.train()
    for step in range(steps):
        schema = valid_schemas[step % len(valid_schemas)]
        pool = by_schema[schema]
        batch_indices = [rng.choice(pool) for _ in range(min(32, max(8, len(pool))))]
        stage = min(2, (3 * step) // max(steps, 1))
        selected_variants = []
        for _ in batch_indices:
            draw = rng.random()
            if stage == 0:
                variant = 0 if draw < 0.75 else 1
            elif stage == 1:
                variant = 0 if draw < 0.50 else 1
            else:
                variant = 0 if draw < 0.34 else (1 if draw < 0.67 else 2)
            selected_variants.append(variant)
            variant_counts[variant] += 1
        local = list(schema_candidates[schema])
        blocked = {normalized[index] for index in local}
        if hard_negatives:
            counts: Counter[int] = Counter()
            for row_index in batch_indices:
                counts.update(hard_negatives.get(str(rows[row_index]["id"]), []))
            for candidate_value, _count in counts.most_common():
                if candidate_value not in local and normalized[candidate_value] not in blocked:
                    local.append(candidate_value)
                    blocked.add(normalized[candidate_value])
        cardinality = EPISODE_CARDINALITIES[step % len(EPISODE_CARDINALITIES)]
        extras = list(range(len(candidates)))
        rng.shuffle(extras)
        for candidate_value in extras:
            if len(local) >= cardinality:
                break
            if candidate_value in local or normalized[candidate_value] in blocked:
                continue
            local.append(candidate_value)
            blocked.add(normalized[candidate_value])
        lookup = {candidate_value: index for index, candidate_value in enumerate(local)}
        target = torch.tensor(
            [lookup[candidate_index[rows[index]["candidate_id"]]] for index in batch_indices],
            dtype=torch.long,
        )
        row_tensor = torch.tensor(batch_indices, dtype=torch.long)
        variant_tensor = torch.tensor(selected_variants, dtype=torch.long)
        noisy_base = query_tensor[row_tensor, variant_tensor]
        scores = model.scores(noisy_base, candidate_tensor[local]) / 0.07
        ce = torch.nn.functional.cross_entropy(scores, target)
        invariance = torch.zeros((), dtype=torch.float32)
        if invariance_weight:
            clean_projected = model.encode_query(query_tensor[row_tensor, 0])
            noisy_projected = model.encode_query(noisy_base)
            invariance = (1.0 - (clean_projected * noisy_projected).sum(dim=1)).mean()
        loss = ce + invariance_weight * invariance
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        ce_losses.append(float(ce.detach()))
        invariance_losses.append(float(invariance.detach()))
    model.eval()
    return {
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "ce_first": ce_losses[0],
        "ce_last": ce_losses[-1],
        "invariance_first": invariance_losses[0],
        "invariance_last": invariance_losses[-1],
        "invariance_weight": invariance_weight,
        "variant_counts": {str(key): value for key, value in sorted(variant_counts.items())},
        "curriculum": ["75% clean + 25% short", "50% clean + 50% short", "clean + short + long"],
    }


def _evaluate_conditions(
    model: ProjectionPair,
    rows: list[dict],
    condition_vectors: dict[str, np.ndarray],
    candidates: list[dict],
    candidate_vectors: np.ndarray,
) -> dict[str, Any]:
    lookup = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    gold = np.asarray([lookup[row["candidate_id"]] for row in rows])
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        metrics, _positions, _order = retrieval_metrics(
            model_scores(model, condition_vectors[condition], candidate_vectors), gold
        )
        conditions[condition] = metrics
    clean = conditions["clean"]["recall_at_8"]
    groups = {
        "clean": ("clean",),
        "short_noise": ("short_prefix", "short_suffix"),
        "medium_noise": (
            "middle_insertion",
            "duplicate_metadata",
            "reordered_fields",
            "boilerplate_wrapper",
            "json_noise",
        ),
        "long_noise": ("long_prefix", "long_suffix", "repeated_low_information"),
    }
    curves: dict[str, Any] = {}
    for name, members in groups.items():
        score = float(np.mean([conditions[member]["recall_at_8"] for member in members]))
        curves[name] = {
            "recall_at_8": score,
            "absolute_degradation_from_clean": score - clean,
            "relative_degradation_from_clean": (score - clean) / clean if clean else None,
            "conditions": list(members),
        }
    robust_mean = float(
        np.mean([conditions[condition]["recall_at_8"] for condition in CONDITIONS if condition != "clean"])
    )
    return {"conditions": conditions, "curves": curves, "robust_mean_recall_at_8": robust_mean}


def _audit_training_data(bundle: Any, train: list[dict]) -> None:
    semantic_records = []
    for row in train:
        for condition in training_conditions(str(row["id"])):
            validate_augmentation(row, condition)
            semantic_records.append(audit_record(row, condition))
    digest = hashlib.sha256(
        "\n".join(
            f"{row['row_id']}\0{row['candidate_id']}\0{row['condition']}\0{row['augmented_chars']}"
            for row in semantic_records
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        SEMANTIC_AUDIT_PATH,
        {
            "format": "veya-phase11-augmentation-audit-v1",
            "noise_version": NOISE_VERSION,
            "partition": "TRAIN only",
            "rows": len(train),
            "variants": len(semantic_records),
            "label_preserved": all(row["label_preserved"] for row in semantic_records),
            "candidate_or_label_used_by_noise_generator": False,
            "records_sha256": digest,
            "sample": semantic_records[:64],
        },
    )
    sample = sorted(train, key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())[:200]
    records = []
    for row in sample:
        for condition in CONDITIONS:
            value = validate_augmentation(row, condition)
            records.append(
                {
                    "row_id": row["id"],
                    "condition": condition,
                    **tokenization_audit(
                        bundle.agent.tok,
                        row["text"],
                        value,
                        max_length=MAX_LENGTH,
                        view_count=1,
                    ),
                }
            )
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in records if row["condition"] == condition]
        by_condition[condition] = {
            "n": len(selected),
            "truncated_rate": float(np.mean([row["truncated"] for row in selected])),
            "mean_tokens_dropped": float(np.mean([row["tokens_dropped"] for row in selected])),
            "signal_fully_retained_rate": float(
                np.mean([row["positive_signal_position"] == "fully_retained" for row in selected])
            ),
            "signal_truncated_rate": float(
                np.mean([row["positive_signal_position"] == "truncated" for row in selected])
            ),
        }
    _write_json(
        AUDIT_PATH,
        {
            "format": "veya-phase11-tokenization-audit-v1",
            "partition": "TRAIN deterministic audit sample",
            "max_length": MAX_LENGTH,
            "tokenizer": type(bundle.agent.tok).__name__,
            "by_condition": by_condition,
            "records": records,
        },
    )


def main() -> None:
    os.makedirs(PHASE11, exist_ok=True)
    train, development, validation, all_candidates = load_inputs()
    bundle, release_manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    if bundle.reranker_id != "base_decision_head" or bundle.hypothesis is not None:
        raise SystemExit("Phase 11 requires the frozen base decision head")
    phase10 = load_domain_retriever(PHASE10_ARTIFACT, bundle.checkpoint_hash)
    production_lock = {
        "release_manifest_sha256": file_sha256(MANIFEST_PATH),
        "production_projection_sha256": bundle.shortlist_sha256,
        "phase10_artifact_sha256": file_sha256(PHASE10_ARTIFACT),
        "scorer": bundle.reranker_id,
        "k": 8,
        "route_temperature": bundle.route_temperature,
        "serving_default_changed": False,
        "release": release_manifest["release"],
    }
    _write_json(os.path.join(PHASE11, "baseline_lock.json"), production_lock)
    _audit_training_data(bundle, train)

    base_vectors, base_index = _base_vector_index(train, development, validation, all_candidates)

    def base_vector(text: str) -> np.ndarray:
        return base_vectors[base_index[text]]

    train_texts = [
        validate_augmentation(row, condition)
        for row in train
        for condition in training_conditions(str(row["id"]))[1:]
    ]
    dev_texts = [
        validate_augmentation(row, condition)
        for row in development
        for condition in CONDITIONS
        if condition != "clean"
    ]
    augmented_vectors, augmented_index = _encode_or_load(bundle, train_texts + dev_texts)

    def vector(text: str) -> np.ndarray:
        index = base_index.get(text)
        return base_vectors[index] if index is not None else augmented_vectors[augmented_index[text]]

    train_variants = np.stack(
        [
            np.stack(
                [
                    base_vector(row["text"])
                    if condition == "clean"
                    else vector(validate_augmentation(row, condition))
                    for condition in training_conditions(str(row["id"]))
                ]
            )
            for row in train
        ]
    )
    dev_conditions = {
        condition: np.stack(
            [
                base_vector(row["text"])
                if condition == "clean"
                else vector(validate_augmentation(row, condition))
                for row in development
            ]
        )
        for condition in CONDITIONS
    }
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in all_candidates}
    train_candidates = [candidate_by_id[value] for value in sorted({row["candidate_id"] for row in train})]
    dev_candidates = [
        candidate_by_id[value] for value in sorted({row["candidate_id"] for row in development})
    ]
    train_candidate_vectors = np.stack([base_vector(row["naturalized"]) for row in train_candidates])
    dev_candidate_vectors = np.stack([base_vector(row["naturalized"]) for row in dev_candidates])

    rb0 = _copy_projection(phase10.projection)
    rb1 = _copy_projection(phase10.projection)
    rb1_training = train_robust_model(
        rb1,
        train,
        train_variants,
        train_candidates,
        train_candidate_vectors,
        steps=STEPS,
        invariance_weight=0.0,
    )
    rb1i = _copy_projection(phase10.projection)
    rb1i_training = train_robust_model(
        rb1i,
        train,
        train_variants,
        train_candidates,
        train_candidate_vectors,
        steps=STEPS,
        invariance_weight=INVARIANCE_WEIGHT,
    )
    development_metrics = {
        "RB0_phase10": _evaluate_conditions(
            rb0, development, dev_conditions, dev_candidates, dev_candidate_vectors
        ),
        "RB1_noise": _evaluate_conditions(
            rb1, development, dev_conditions, dev_candidates, dev_candidate_vectors
        ),
        "RB1I_noise_invariance": _evaluate_conditions(
            rb1i, development, dev_conditions, dev_candidates, dev_candidate_vectors
        ),
    }
    floor = development_metrics["RB0_phase10"]["conditions"]["clean"]["recall_at_8"] - 0.01
    robust_candidates = {"RB1_noise": rb1, "RB1I_noise_invariance": rb1i}
    eligible = [
        name
        for name in robust_candidates
        if development_metrics[name]["conditions"]["clean"]["recall_at_8"] >= floor
    ]
    if not eligible:
        eligible = list(robust_candidates)
    selected_rb1_name = max(
        eligible,
        key=lambda name: (
            development_metrics[name]["robust_mean_recall_at_8"],
            development_metrics[name]["conditions"]["clean"]["recall_at_8"],
            name == "RB1_noise",
        ),
    )
    selected_rb1 = robust_candidates[selected_rb1_name]
    noisy_train_scores = model_scores(
        selected_rb1, train_variants[:, 1, :], train_candidate_vectors
    )
    hard_negatives, lineage = mine_hard_negatives_with_lineage(
        noisy_train_scores,
        train,
        train_candidates,
        train_candidate_vectors,
        retriever_id="phase11-rb1-selected-v1",
    )
    validate_hard_negative_lineage(lineage)
    _write_json(LINEAGE_PATH, lineage)
    rb2 = _copy_projection(selected_rb1)
    rb2_training = train_robust_model(
        rb2,
        train,
        train_variants,
        train_candidates,
        train_candidate_vectors,
        steps=HARD_STEPS,
        invariance_weight=INVARIANCE_WEIGHT if selected_rb1_name.endswith("invariance") else 0.0,
        hard_negatives=hard_negatives,
    )
    development_metrics["RB2_second_hard_negative"] = _evaluate_conditions(
        rb2, development, dev_conditions, dev_candidates, dev_candidate_vectors
    )
    final_candidates = {selected_rb1_name: selected_rb1, "RB2_second_hard_negative": rb2}
    final_eligible = [
        name
        for name in final_candidates
        if development_metrics[name]["conditions"]["clean"]["recall_at_8"] >= floor
    ]
    if not final_eligible:
        final_eligible = list(final_candidates)
    winner_name = max(
        final_eligible,
        key=lambda name: (
            development_metrics[name]["robust_mean_recall_at_8"],
            development_metrics[name]["conditions"]["clean"]["recall_at_8"],
            name == selected_rb1_name,
        ),
    )
    winner = final_candidates[winner_name]
    validation_hash = split_digest(validation)
    metadata = {
        "retriever_id": "phase11-noise-robust-k8-v1",
        "backbone": "models/production frozen ModernBERT-large encoder",
        "base_backbone": "models/production frozen ModernBERT-large encoder",
        "checkpoint_hash": bundle.checkpoint_hash,
        "training_datasets": ["banking77:train", "schema_guided_dialogue:train"],
        "training_data": ["banking77:train", "schema_guided_dialogue:train"],
        "held_out_schemas": ["clinc150"],
        "candidate_representation": "naturalized_label",
        "query_representation": "state_only with bounded deterministic views",
        "objective": (
            "episodic InfoNCE with TRAIN-only generic noise curriculum and one second hard-negative refresh"
            if winner_name == "RB2_second_hard_negative"
            else "episodic InfoNCE with TRAIN-only generic noise curriculum"
        ),
        "noise_augmentation_version": NOISE_VERSION,
        "hard_negative_version": HARD_NEGATIVE_VERSION,
        "hard_negative_refresh_selected": winner_name == "RB2_second_hard_negative",
        "query_view_policy": "adaptive_1_to_4_max_fusion",
        "pooling": "frozen masked mean pooling followed by learned shared projection",
        "k": 8,
        "git_commit": git_commit(),
        "validation_split_hash": validation_hash,
        "split_freeze_sha256": file_sha256(os.path.join(ROOT, "artifacts/experiments/phase10/splits.json")),
        "selected_variant": winner_name,
        "selection_partition": "schema-held-out SGD development services",
        "selection_metrics": development_metrics[winner_name],
        "clean_noninferiority_floor_recall_at_8": floor,
        "max_length": MAX_LENGTH,
        "max_views": 4,
        "fusion": "max",
    }
    artifact = save_domain_retriever(ARTIFACT_PATH, winner, metadata)
    torch.save(
        {
            "format": "veya-phase11-ablation-models-v1",
            "checkpoint_hash": bundle.checkpoint_hash,
            "models": {
                "RB0_phase10": rb0.state_dict(),
                "RB1_noise": rb1.state_dict(),
                "RB1I_noise_invariance": rb1i.state_dict(),
                "RB2_second_hard_negative": rb2.state_dict(),
            },
            "projection": rb0.projection,
            "input_dim": rb0.input_dim,
            "output_dim": rb0.output_dim,
        },
        ABLATIONS_PATH,
    )
    report = {
        "format": "veya-phase11-training-report-v1",
        "status": "TRAINED_SCHEMA_HELD_OUT_NOT_YET_VALIDATED",
        "production_lock": production_lock,
        "partitions": {
            "training": {"rows": len(train), "schemas": len({row["schema"] for row in train})},
            "development": {
                "rows": len(development),
                "schemas": len({row["schema"] for row in development}),
            },
            "schema_held_out_validation": {
                "rows": len(validation),
                "schema": "clinc150",
                "opened_for_selection": False,
                "split_hash": validation_hash,
            },
        },
        "development": development_metrics,
        "clean_noninferiority_floor": floor,
        "selected_after_noise": selected_rb1_name,
        "winner": winner_name,
        "training": {
            "RB1_noise": rb1_training,
            "RB1I_noise_invariance": rb1i_training,
            "RB2_second_hard_negative": rb2_training,
        },
        "hard_negative_lineage": {
            key: value for key, value in lineage.items() if key not in ("records", "manual_audit_sample")
        },
        "artifact": artifact,
        "ablation_models_sha256": file_sha256(ABLATIONS_PATH),
        "consumed_blind_sets_opened": [],
    }
    _write_json(REPORT_PATH, report)
    print(
        json.dumps(
            {
                "winner": winner_name,
                "development": development_metrics,
                "artifact": artifact,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
