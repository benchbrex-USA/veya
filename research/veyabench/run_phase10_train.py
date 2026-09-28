"""Train and select the Phase-10 frozen-backbone domain retriever.

This harness never reads Banking77 test or CLINC150 validation/test rows.  It
fits small projection heads on precomputed frozen-encoder vectors, selects on a
whole held-out CLINC150 schema, and writes only aggregate validation metrics.
"""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from collections import Counter, defaultdict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import DATA_OUT, PHASE10, ROOT, SEED, build
from veya.experimental.domain_retriever import (
    ProjectionPair,
    naturalize_label,
    save_domain_retriever,
    text_sha256,
)
from veya.release_bundle import load_release_manifest
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


MODEL_PATH = os.path.join(ROOT, "models", "production")
MANIFEST_PATH = os.path.join(ROOT, "artifacts", "release", "manifest.json")
CACHE_PATH = os.path.join(PHASE10, "frozen_embeddings.npz")
CACHE_META_PATH = CACHE_PATH + ".json"
ARTIFACT_PATH = os.path.join(PHASE10, "domain_general_k8.pt")
REPORT_PATH = os.path.join(PHASE10, "retriever_comparison.json")
OUTPUT_DIM = 128
STEPS = 600
HARD_STEPS = 300
LEARNING_RATE = 1e-3
EPISODE_CARDINALITIES = (8, 16, 32, 50, 77, 100)


def load_jsonl(name: str) -> list[dict]:
    path = os.path.join(DATA_OUT, name + ".jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_inputs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    build()
    with open(os.path.join(DATA_OUT, "candidates.json"), encoding="utf-8") as handle:
        candidates = json.load(handle)
    return (
        load_jsonl("train"),
        load_jsonl("development"),
        load_jsonl("schema_held_out_validation"),
        candidates,
    )


def _embedding_texts(
    train: list[dict], development: list[dict], validation: list[dict], candidates: list[dict]
) -> tuple[list[str], dict[str, int]]:
    values = {row["text"] for row in train + development + validation}
    for candidate in candidates:
        values.update(
            str(candidate[key])
            for key in ("raw", "naturalized", "name_definition")
            if candidate.get(key)
        )
        values.add("intent: " + str(candidate["naturalized"]))
    texts = sorted(values)
    return texts, {text: index for index, text in enumerate(texts)}


def encode_or_load(bundle, texts: list[str]) -> np.ndarray:
    expected = {
        "format": "veya-phase10-embedding-cache-v1",
        "checkpoint_hash": bundle.checkpoint_hash,
        "texts_sha256": text_sha256(texts),
        "max_length": int(bundle.router["embed_max_length"]),
        "count": len(texts),
    }
    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if all(metadata.get(key) == value for key, value in expected.items()):
            cached = np.load(CACHE_PATH)["vectors"]
            if cached.shape == (len(texts), 1024):
                print("loaded frozen embedding cache", cached.shape, flush=True)
                return cached.astype(np.float32, copy=False)

    embed = embed_fn_from_agent(
        bundle.agent,
        max_length=int(bundle.router["embed_max_length"]),
        batch_size=16,
    )
    parts = []
    started = time.perf_counter()
    for start in range(0, len(texts), 256):
        stop = min(start + 256, len(texts))
        parts.append(embed(texts[start:stop]))
        print("base embeddings %d/%d" % (stop, len(texts)), flush=True)
    vectors = np.concatenate(parts, axis=0).astype(np.float32)
    np.savez_compressed(CACHE_PATH, vectors=vectors)
    metadata = {
        **expected,
        "shape": list(vectors.shape),
        "seconds": time.perf_counter() - started,
        "file_sha256": file_sha256(CACHE_PATH),
    }
    with open(CACHE_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return vectors


def current_projection(bundle, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        q = bundle.projection(torch.as_tensor(query, dtype=torch.float32))
        c = bundle.projection(torch.as_tensor(candidates, dtype=torch.float32))
        return (q @ c.T).cpu().numpy()


def model_scores(model: ProjectionPair, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return model.scores(
            torch.as_tensor(query, dtype=torch.float32),
            torch.as_tensor(candidates, dtype=torch.float32),
        ).cpu().numpy()


def retrieval_metrics(scores: np.ndarray, gold: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    positions = np.argmax(order == gold[:, None], axis=1) + 1
    result = {
        "n": int(len(gold)),
        "candidate_count": int(scores.shape[1]),
        "recall_at_1": float(np.mean(positions <= 1)),
        "recall_at_4": float(np.mean(positions <= 4)),
        "recall_at_8": float(np.mean(positions <= 8)),
        "recall_at_16": float(np.mean(positions <= 16)),
        "recall_at_32": float(np.mean(positions <= 32)),
        "mrr": float(np.mean(1.0 / positions)),
        "mean_rank": float(np.mean(positions)),
        "median_rank": float(np.median(positions)),
    }
    return result, positions, order


def paired_bootstrap(base_hits: np.ndarray, candidate_hits: np.ndarray, draws: int = 2000) -> dict:
    delta = candidate_hits.astype(np.float64) - base_hits.astype(np.float64)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "draws": draws,
    }


def initialize_from_current(bundle, mode: str) -> ProjectionPair:
    model = ProjectionPair(1024, OUTPUT_DIM, mode)
    weight = bundle.projection.weight.detach().cpu()
    if tuple(weight.shape) != (OUTPUT_DIM, 1024):
        raise ValueError("unexpected current projection shape %r" % (tuple(weight.shape),))
    with torch.no_grad():
        model.query.weight.copy_(weight)
        if mode == "dual":
            model.candidate.weight.copy_(weight)
    return model


def train_model(
    model: ProjectionPair,
    rows: list[dict],
    row_vectors: np.ndarray,
    candidates: list[dict],
    candidate_vectors: np.ndarray,
    steps: int,
    hard_negatives: dict[str, list[int]] | None = None,
) -> list[float]:
    torch.manual_seed(SEED)
    rng = random.Random(SEED + (1 if hard_negatives else 0))
    by_schema: dict[str, list[int]] = defaultdict(list)
    schema_candidates: dict[str, list[int]] = defaultdict(list)
    candidate_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    for index, row in enumerate(rows):
        by_schema[row["schema"]].append(index)
    for index, candidate in enumerate(candidates):
        schema_candidates[candidate["schema"]].append(index)
    valid_schemas = sorted(schema for schema, indices in schema_candidates.items() if len(indices) >= 2)
    normalized = [naturalize_label(row["raw"]) for row in candidates]
    query_tensor = torch.as_tensor(row_vectors, dtype=torch.float32)
    candidate_tensor = torch.as_tensor(candidate_vectors, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    losses = []
    model.train()
    for step in range(steps):
        schema = valid_schemas[step % len(valid_schemas)]
        pool = by_schema[schema]
        batch_indices = [rng.choice(pool) for _ in range(min(32, max(8, len(pool))))]
        local = list(schema_candidates[schema])
        blocked_text = {normalized[index] for index in local}
        if hard_negatives:
            counts: Counter[int] = Counter()
            for row_index in batch_indices:
                counts.update(hard_negatives.get(rows[row_index]["id"], []))
            for candidate_index_value, _count in counts.most_common():
                if candidate_index_value not in local and normalized[candidate_index_value] not in blocked_text:
                    local.append(candidate_index_value)
                    blocked_text.add(normalized[candidate_index_value])
        cardinality = EPISODE_CARDINALITIES[step % len(EPISODE_CARDINALITIES)]
        extras = list(range(len(candidates)))
        rng.shuffle(extras)
        for candidate_index_value in extras:
            if len(local) >= cardinality:
                break
            if candidate_index_value in local or normalized[candidate_index_value] in blocked_text:
                continue
            local.append(candidate_index_value)
            blocked_text.add(normalized[candidate_index_value])
        target_lookup = {candidate_id: index for index, candidate_id in enumerate(local)}
        target = torch.tensor(
            [target_lookup[candidate_index[rows[index]["candidate_id"]]] for index in batch_indices],
            dtype=torch.long,
        )
        scores = model.scores(query_tensor[batch_indices], candidate_tensor[local]) / 0.07
        loss = torch.nn.functional.cross_entropy(scores, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def mine_hard_negatives(
    model: ProjectionPair,
    rows: list[dict],
    row_vectors: np.ndarray,
    candidates: list[dict],
    candidate_vectors: np.ndarray,
) -> dict[str, list[int]]:
    scores = model_scores(model, row_vectors, candidate_vectors)
    order = np.argsort(-scores, axis=1, kind="stable")
    index = {candidate["candidate_id"]: i for i, candidate in enumerate(candidates)}
    normalized = [naturalize_label(candidate["raw"]) for candidate in candidates]
    out = {}
    for row_index, row in enumerate(rows):
        truth = index[row["candidate_id"]]
        selected = []
        for candidate_index in order[row_index]:
            candidate_index = int(candidate_index)
            if candidate_index == truth or normalized[candidate_index] == normalized[truth]:
                continue
            selected.append(candidate_index)
            if len(selected) == 8:
                break
        out[row["id"]] = selected
    return out


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    os.makedirs(PHASE10, exist_ok=True)
    train, development, validation, all_candidates = load_inputs()
    bundle, release_manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    if bundle.reranker_id != "base_decision_head" or bundle.hypothesis is not None:
        raise SystemExit("Phase 10 requires the frozen base decision head")
    texts, text_index = _embedding_texts(train, development, validation, all_candidates)
    vectors = encode_or_load(bundle, texts)

    def vector(text: str) -> np.ndarray:
        return vectors[text_index[text]]

    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in all_candidates}
    train_ids = {row["candidate_id"] for row in train}
    dev_ids = {row["candidate_id"] for row in development}
    validation_ids = {row["candidate_id"] for row in validation}
    train_candidates = [candidate_by_id[value] for value in sorted(train_ids)]
    dev_candidates = [candidate_by_id[value] for value in sorted(dev_ids)]
    validation_candidates = [candidate_by_id[value] for value in sorted(validation_ids)]

    train_q = np.stack([vector(row["text"]) for row in train])
    dev_q = np.stack([vector(row["text"]) for row in development])
    validation_q = np.stack([vector(row["text"]) for row in validation])
    dev_gold_ids = {candidate["candidate_id"]: i for i, candidate in enumerate(dev_candidates)}
    val_gold_ids = {candidate["candidate_id"]: i for i, candidate in enumerate(validation_candidates)}
    dev_gold = np.asarray([dev_gold_ids[row["candidate_id"]] for row in development])
    validation_gold = np.asarray([val_gold_ids[row["candidate_id"]] for row in validation])

    representations = {
        "raw": np.stack([vector(candidate["raw"]) for candidate in validation_candidates]),
        "naturalized": np.stack([vector(candidate["naturalized"]) for candidate in validation_candidates]),
        "production_prefixed": np.stack(
            [vector("intent: " + candidate["naturalized"]) for candidate in validation_candidates]
        ),
    }
    comparison: dict[str, dict] = {}
    positions: dict[str, np.ndarray] = {}
    orders: dict[str, np.ndarray] = {}
    for name, candidate_vectors in representations.items():
        started = time.perf_counter()
        scores = current_projection(bundle, validation_q, candidate_vectors)
        elapsed = time.perf_counter() - started
        metrics, rank, order = retrieval_metrics(scores, validation_gold)
        metrics["projection_and_exact_rank_ms_per_query"] = elapsed * 1000.0 / len(validation)
        comparison["B0_current_" + name] = metrics
        positions["B0_current_" + name] = rank
        orders["B0_current_" + name] = order

    train_candidate_vectors = {
        "naturalized": np.stack([vector(candidate["naturalized"]) for candidate in train_candidates]),
        "name_definition": np.stack([vector(candidate["name_definition"]) for candidate in train_candidates]),
    }
    dev_candidate_vectors = {
        "raw": np.stack([vector(candidate["raw"]) for candidate in dev_candidates]),
        "naturalized": np.stack([vector(candidate["naturalized"]) for candidate in dev_candidates]),
        "name_definition": np.stack([vector(candidate["name_definition"]) for candidate in dev_candidates]),
    }
    val_naturalized = representations["naturalized"]

    development_ablation = {}
    for representation, candidate_vectors in dev_candidate_vectors.items():
        metrics, _rank, _order = retrieval_metrics(
            current_projection(bundle, dev_q, candidate_vectors), dev_gold
        )
        development_ablation["current_" + representation] = metrics

    b2_shared = initialize_from_current(bundle, "shared")
    shared_losses = train_model(
        b2_shared,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["naturalized"],
        STEPS,
    )
    b2_dual = initialize_from_current(bundle, "dual")
    dual_losses = train_model(
        b2_dual,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["naturalized"],
        STEPS,
    )
    dev_models = {"shared": b2_shared, "dual": b2_dual}
    dev_choice = {}
    for mode, model in dev_models.items():
        metrics, _rank, _order = retrieval_metrics(
            model_scores(model, dev_q, dev_candidate_vectors["naturalized"]), dev_gold
        )
        dev_choice[mode] = metrics
    selected_projection = max(
        dev_choice,
        key=lambda mode: (
            dev_choice[mode]["recall_at_8"],
            dev_choice[mode]["mrr"],
            -dev_choice[mode]["mean_rank"],
            mode == "shared",
        ),
    )
    b2 = copy.deepcopy(dev_models[selected_projection])
    b2_losses = shared_losses if selected_projection == "shared" else dual_losses

    hard = mine_hard_negatives(
        b2,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["naturalized"],
    )
    b3 = copy.deepcopy(b2)
    b3_losses = train_model(
        b3,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["naturalized"],
        HARD_STEPS,
        hard_negatives=hard,
    )

    b4 = initialize_from_current(bundle, selected_projection)
    b4_losses = train_model(
        b4,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["name_definition"],
        STEPS,
    )
    b4_hard = mine_hard_negatives(
        b4,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["name_definition"],
    )
    b4_hard_losses = train_model(
        b4,
        train,
        train_q,
        train_candidates,
        train_candidate_vectors["name_definition"],
        HARD_STEPS,
        hard_negatives=b4_hard,
    )

    finalists = {"B2_multi_schema": b2, "B3_hard_negatives": b3, "B4_definitions": b4}
    for name, model in finalists.items():
        started = time.perf_counter()
        scores = model_scores(model, validation_q, val_naturalized)
        elapsed = time.perf_counter() - started
        metrics, rank, order = retrieval_metrics(scores, validation_gold)
        metrics["projection_and_exact_rank_ms_per_query"] = elapsed * 1000.0 / len(validation)
        comparison[name] = metrics
        positions[name] = rank
        orders[name] = order

    baseline_name = "B0_current_production_prefixed"
    base_hits = positions[baseline_name] <= 8
    uncertainty = {
        name: paired_bootstrap(base_hits, positions[name] <= 8)
        for name in finalists
    }
    winner_name = max(
        finalists,
        key=lambda name: (
            comparison[name]["recall_at_8"],
            comparison[name]["mrr"],
            -comparison[name]["projection_and_exact_rank_ms_per_query"],
        ),
    )
    winner = finalists[winner_name]
    metadata = {
        "retriever_id": "phase10-domain-general-k8-v1",
        "backbone": "models/production frozen ModernBERT-large encoder",
        "checkpoint_hash": bundle.checkpoint_hash,
        "training_datasets": ["banking77:train", "schema_guided_dialogue:train"],
        "held_out_schemas": ["clinc150"],
        "candidate_representation": "naturalized_label",
        "query_representation": "state_only",
        "objective": "episodic InfoNCE cross-entropy with one hard-negative refresh"
        if winner_name == "B3_hard_negatives"
        else "episodic InfoNCE cross-entropy",
        "k": 8,
        "git_commit": git_commit(),
        "dataset_registry_sha256": file_sha256(os.path.join(DATA_OUT, "registry.json")),
        "split_freeze_sha256": file_sha256(os.path.join(PHASE10, "splits.json")),
        "selected_variant": winner_name,
        "selection_metrics": comparison[winner_name],
        "training_steps": STEPS + (HARD_STEPS if winner_name in ("B3_hard_negatives", "B4_definitions") else 0),
        "learning_rate": LEARNING_RATE,
        "episode_cardinalities": list(EPISODE_CARDINALITIES),
    }
    artifact = save_domain_retriever(ARTIFACT_PATH, winner, metadata)

    report = {
        "format": "veya-phase10-retriever-comparison-v1",
        "status": "FINALIST_FROZEN_BLIND_NOT_OPENED",
        "release_manifest_sha256": file_sha256(MANIFEST_PATH),
        "release": release_manifest["release"],
        "current_retriever": {
            "architecture": "shared bias-free 1024-to-128 projection",
            "projection": "models/production/shortlist_k8.pt",
            "projection_sha256": bundle.shortlist_sha256,
            "checkpoint_sha256": bundle.checkpoint_hash,
            "training_dataset": "Banking77 train seed17 rows 0:800",
            "candidate_representation": "criterion value or label",
            "query_representation": "serialized state",
            "similarity": "unnormalized dot product",
            "normalization": "none",
            "k": 8,
            "candidate_encoding": "frozen encoder, mean pool, max length 96",
            "cache": "process-local by exact text",
        },
        "split_freeze_sha256": metadata["split_freeze_sha256"],
        "development_projection_choice": {
            "metrics": dev_choice,
            "selected": selected_projection,
        },
        "development_candidate_representation": development_ablation,
        "schema_held_out_validation": {
            "dataset": "CLINC150 train partition only",
            "schema": "clinc150",
            "examples": len(validation),
            "candidates": len(validation_candidates),
            "comparison": comparison,
            "recall_at_8_delta_bootstrap": uncertainty,
        },
        "hard_negative_result": {
            "B2_recall_at_8": comparison["B2_multi_schema"]["recall_at_8"],
            "B3_recall_at_8": comparison["B3_hard_negatives"]["recall_at_8"],
            "delta": comparison["B3_hard_negatives"]["recall_at_8"]
            - comparison["B2_multi_schema"]["recall_at_8"],
        },
        "training": {
            "steps": STEPS,
            "hard_steps": HARD_STEPS,
            "shared_loss": [shared_losses[0], shared_losses[-1]],
            "dual_loss": [dual_losses[0], dual_losses[-1]],
            "selected_B2_loss": [b2_losses[0], b2_losses[-1]],
            "B3_loss": [b3_losses[0], b3_losses[-1]],
            "B4_loss": [b4_losses[0], b4_losses[-1]],
            "B4_hard_loss": [b4_hard_losses[0], b4_hard_losses[-1]],
        },
        "winner": winner_name,
        "artifact": artifact,
        "future_blind": "not downloaded or inspected",
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"winner": winner_name, "comparison": comparison, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
