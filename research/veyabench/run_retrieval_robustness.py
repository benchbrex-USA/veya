"""Measure retrieval robustness and one train-only noise projection.

Does not read CLINC150 validation or test, MASSIVE, or any sealed blind file.
Does not write production weights, calibration, or the Phase-10 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import DATA_OUT, PHASE10, ROOT
from research.veyabench.retrieval_robustness import apply_perturbation, query_views
from research.veyabench.run_phase10_train import (
    CACHE_META_PATH,
    CACHE_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
    _embedding_texts,
    encode_or_load,
    initialize_from_current,
    mine_hard_negatives,
    model_scores,
    retrieval_metrics,
    train_model,
)
from veya.experimental.domain_retriever import load_domain_retriever
from veya.release_bundle import load_release_manifest
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256

OUT = os.path.join(ROOT, "artifacts", "experiments", "retrieval_robustness")
KINDS = ("clean", "short_suffix", "long_suffix", "prefix", "json_wrapper")
ARTIFACT = os.path.join(PHASE10, "domain_general_k8.pt")


def _read_existing_splits():
    """Read the already built Phase-10 splits. Do not regenerate them."""

    def rows(name: str) -> list[dict]:
        path = os.path.join(DATA_OUT, name + ".jsonl")
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with open(os.path.join(DATA_OUT, "candidates.json"), encoding="utf-8") as handle:
        candidates = json.load(handle)
    return rows("train"), rows("development"), rows("schema_held_out_validation"), candidates


def rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes. Linux reports kilobytes.
    if platform.system() == "Darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def encode_missing(bundle, texts: list[str], cache_path: str) -> dict[str, np.ndarray]:
    meta_path = cache_path + ".json"
    found: dict[str, np.ndarray] = {}
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("checkpoint_hash") == bundle.checkpoint_hash:
            blob = np.load(cache_path)
            stored = metadata.get("texts") or []
            vectors = blob["vectors"]
            if len(stored) == len(vectors):
                found = {text: vectors[index] for index, text in enumerate(stored)}
    missing = [text for text in texts if text not in found]
    if missing:
        embed = embed_fn_from_agent(
            bundle.agent,
            max_length=int(bundle.router["embed_max_length"]),
            batch_size=32,
        )
        started = time.perf_counter()
        parts = []
        for start in range(0, len(missing), 256):
            stop = min(start + 256, len(missing))
            parts.append(embed(missing[start:stop]))
            print("robust embeddings %d/%d" % (stop, len(missing)), flush=True)
        encoded = np.concatenate(parts, axis=0).astype(np.float32)
        for text, vector in zip(missing, encoded):
            found[text] = vector
        ordered = sorted(found)
        matrix = np.stack([found[text] for text in ordered])
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, vectors=matrix)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": "veya-retrieval-robustness-cache-v1",
                    "checkpoint_hash": bundle.checkpoint_hash,
                    "texts": ordered,
                    "seconds": time.perf_counter() - started,
                    "count": len(ordered),
                },
                handle,
            )
            handle.write("\n")
        found = {text: matrix[index] for index, text in enumerate(ordered)}
    return found


def stack(found: dict[str, np.ndarray], texts: list[str]) -> np.ndarray:
    return np.stack([found[text] for text in texts]).astype(np.float32, copy=False)


def fuse_max(parts: list[np.ndarray]) -> np.ndarray:
    return np.max(np.stack(parts, axis=0), axis=0)


def permutation_delta(scores: np.ndarray) -> float:
    perm = np.random.default_rng(0).permutation(scores.shape[1])
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    moved = scores[:, perm][:, inverse]
    return float(np.max(np.abs(moved - scores)))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    train, development, validation, all_candidates = _read_existing_splits()
    bundle, _manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    clean_texts, clean_index = _embedding_texts(train, development, validation, all_candidates)
    with open(CACHE_META_PATH, encoding="utf-8") as handle:
        clean_meta = json.load(handle)
    from veya.experimental.domain_retriever import text_sha256

    if clean_meta.get("texts_sha256") != text_sha256(clean_texts):
        raise SystemExit("frozen embedding cache does not match the rebuilt text list")
    clean_vectors = encode_or_load(bundle, clean_texts)
    phase10 = load_domain_retriever(ARTIFACT, bundle.checkpoint_hash).projection
    phase10.eval()

    validation_ids = {row["candidate_id"] for row in validation}
    candidates = [row for row in all_candidates if row["candidate_id"] in validation_ids]
    candidates.sort(key=lambda row: row["candidate_id"])
    gold_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    gold = np.asarray([gold_index[row["candidate_id"]] for row in validation], dtype=np.int64)
    candidate_vectors = np.stack(
        [clean_vectors[clean_index[row["naturalized"]]] for row in candidates]
    ).astype(np.float32)

    needed: list[str] = []
    observed: dict[str, dict[str, dict[str, str]]] = {}
    for kind in KINDS:
        if kind == "clean":
            continue
        for row in validation:
            perturbed = apply_perturbation(row["text"], kind)
            views = query_views(perturbed)
            observed.setdefault(kind, {})[row["id"]] = views
            needed.extend(views.values())
    for row in train:
        for kind in ("short_suffix", "long_suffix"):
            needed.append(apply_perturbation(row["text"], kind))
    unique = sorted(set(needed))
    print("unique strings to encode", len(unique), flush=True)
    encoded = encode_missing(
        bundle, unique, os.path.join(OUT, "perturbed_embeddings.npz")
    )

    def score_with(model_name: str, queries: np.ndarray) -> np.ndarray:
        if model_name == "production":
            with torch.inference_mode():
                q = bundle.projection(torch.as_tensor(queries, dtype=torch.float32))
                c = bundle.projection(torch.as_tensor(candidate_vectors, dtype=torch.float32))
                return (q @ c.T).cpu().numpy()
        return model_scores(phase10, queries, candidate_vectors)

    report_rows = []
    for kind in KINDS:
        if kind == "clean":
            full = np.stack([clean_vectors[clean_index[row["text"]]] for row in validation])
            view_bank = {"full": full}
        else:
            view_bank = {}
            for name in ("full", "head", "tail", "json_text"):
                texts = []
                present = True
                for row in validation:
                    views = observed[kind][row["id"]]
                    if name not in views:
                        present = False
                        break
                    texts.append(views[name])
                if present:
                    view_bank[name] = stack(encoded, texts)
        for model_name in ("production", "phase10"):
            started = time.perf_counter()
            full_scores = score_with(model_name, view_bank["full"])
            elapsed = time.perf_counter() - started
            metrics, _positions, _order = retrieval_metrics(full_scores, gold)
            fused = fuse_max(
                [score_with(model_name, vectors) for vectors in view_bank.values()]
            )
            fused_metrics, _positions, _order = retrieval_metrics(fused, gold)
            report_rows.append(
                {
                    "model": model_name,
                    "condition": kind,
                    "view": "full",
                    "metrics": metrics,
                    "score_seconds": elapsed,
                    "permutation_max_abs_delta": permutation_delta(full_scores),
                }
            )
            report_rows.append(
                {
                    "model": model_name,
                    "condition": kind,
                    "view": "max_fusion",
                    "views": sorted(view_bank),
                    "metrics": fused_metrics,
                }
            )
            print(model_name, kind, "full", round(metrics["recall_at_8"], 4), "fusion", round(fused_metrics["recall_at_8"], 4), flush=True)

    # Train-only noise projection. Validation strings are not in this step.
    train_aug_rows = []
    train_aug_vectors = []
    for row in train:
        train_aug_rows.append(row)
        train_aug_vectors.append(clean_vectors[clean_index[row["text"]]])
        for kind in ("short_suffix", "long_suffix"):
            clone = dict(row)
            clone["id"] = row["id"] + ":" + kind
            train_aug_rows.append(clone)
            train_aug_vectors.append(encoded[apply_perturbation(row["text"], kind)])
    train_matrix = np.stack(train_aug_vectors).astype(np.float32)
    candidate_by_id = {row["candidate_id"]: row for row in all_candidates}
    train_candidates = [candidate_by_id[value] for value in sorted({row["candidate_id"] for row in train})]
    train_candidate_vectors = np.stack(
        [clean_vectors[clean_index[row["naturalized"]]] for row in train_candidates]
    ).astype(np.float32)
    model = initialize_from_current(bundle, "shared")
    losses = train_model(model, train_aug_rows, train_matrix, train_candidates, train_candidate_vectors, steps=400)
    hard = mine_hard_negatives(model, train, np.stack([clean_vectors[clean_index[row["text"]]] for row in train]), train_candidates, train_candidate_vectors)
    model = initialize_from_current(bundle, "shared")
    losses_refresh = train_model(
        model, train_aug_rows, train_matrix, train_candidates, train_candidate_vectors, steps=400, hard_negatives=hard
    )
    torch.save(
        {"format": "retrieval-robustness-research-v1", "state_dict": model.state_dict(), "promoted": False},
        os.path.join(OUT, "noise_projection.pt"),
    )
    candidate_results = []
    for kind in ("clean", "long_suffix"):
        if kind == "clean":
            queries = np.stack([clean_vectors[clean_index[row["text"]]] for row in validation])
        else:
            queries = stack(encoded, [observed[kind][row["id"]]["full"] for row in validation])
        metrics, _positions, _order = retrieval_metrics(model_scores(model, queries, candidate_vectors), gold)
        candidate_results.append({"condition": kind, "view": "full", "metrics": metrics})
        print("candidate", kind, round(metrics["recall_at_8"], 4), flush=True)

    production_clean = next(row["metrics"]["recall_at_8"] for row in report_rows if row["model"] == "production" and row["condition"] == "clean" and row["view"] == "full")
    production_long = next(row["metrics"]["recall_at_8"] for row in report_rows if row["model"] == "production" and row["condition"] == "long_suffix" and row["view"] == "full")
    candidate_clean = candidate_results[0]["metrics"]["recall_at_8"]
    candidate_long = candidate_results[1]["metrics"]["recall_at_8"]
    gate = {
        "schema_held_out_recall_at_8_improved": candidate_clean > production_clean,
        "long_suffix_materially_improved": candidate_long >= production_long + 0.05,
        "clean_regression_within_0.02": candidate_clean >= production_clean - 0.02,
        "permutation_deterministic": all(row.get("permutation_max_abs_delta", 0.0) == 0.0 for row in report_rows if "permutation_max_abs_delta" in row),
        "blind_set_used": False,
        "production_artifacts_written": False,
    }
    gate["passed"] = all(
        gate[key]
        for key in (
            "schema_held_out_recall_at_8_improved",
            "long_suffix_materially_improved",
            "clean_regression_within_0.02",
            "permutation_deterministic",
        )
    )
    payload = {
        "format": "veya-retrieval-robustness-v1",
        "measurement": "measured",
        "status": "CANDIDATE" if gate["passed"] else "RESEARCH",
        "promoted": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": "cpu",
        "checkpoint_hash": bundle.checkpoint_hash,
        "validation_rows": len(validation),
        "validation_candidates": len(candidates),
        "train_rows": len(train),
        "rss_mb": rss_mb(),
        "comparisons": report_rows,
        "noise_projection": {
            "steps": 400,
            "hard_negative_refresh": 1,
            "loss_tail": losses[-5:],
            "refresh_loss_tail": losses_refresh[-5:],
            "artifact": "artifacts/experiments/retrieval_robustness/noise_projection.pt",
            "results": candidate_results,
        },
        "gate": gate,
        "data_hash": hashlib.sha256(open(os.path.join(DATA_OUT, "schema_held_out_validation.jsonl"), "rb").read()).hexdigest(),
        "model_dir": MODEL_PATH,
        "clean_cache_sha256": file_sha256(CACHE_PATH),
    }
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": payload["status"], "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
