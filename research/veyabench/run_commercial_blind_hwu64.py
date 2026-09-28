"""One-shot aggregate-only HWU64 commercial blind evaluation.

The protocol is frozen in ``docs/commercialization/blind_hwu64_freeze.json``.
Do not use this script for model or policy selection after the first run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import ROOT
from research.veyabench.run_phase10_train import (
    ARTIFACT_PATH,
    MANIFEST_PATH,
    load_inputs,
    paired_bootstrap,
    retrieval_metrics,
)
from veya.common import ece_score, serialize_state
from veya.experimental.domain_retriever import (
    build_query_views,
    fuse_query_scores,
    load_domain_retriever,
    naturalize_label,
    text_sha256,
)
from veya.release_bundle import INSTRUCTIONS, load_release_manifest, raw_choice_logits
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import file_sha256


DATA_PATH = os.path.join(
    ROOT,
    "research/data/hwu64/NLU-Data-Home-Domain-Annotated-All.csv",
)
LICENSE_PATH = os.path.join(ROOT, "research/data/hwu64/LICENSE")
FREEZE_PATH = os.path.join(ROOT, "docs/commercialization/blind_hwu64_freeze.json")
POLICY_PATH = os.path.join(ROOT, "artifacts/commercialization/query_policy_freeze.json")
REPORT_PATH = os.path.join(ROOT, "artifacts/commercialization/blind_hwu64_result.json")
REGISTRY_PATH = os.path.join(ROOT, "research/data/hwu64/registry.json")
MAX_ROWS = 3000
E2E_ROWS = 500


def normalized_component(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def load_rows() -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    with open(DATA_PATH, encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter=";"))
    required = {
        "userid",
        "answerid",
        "scenario",
        "intent",
        "status",
        "answer_normalised",
    }
    if not source or not required.issubset(source[0]):
        raise ValueError("HWU64 source schema does not match the frozen protocol")
    rows = []
    rejected_irregular = 0
    rejected_empty = 0
    for source_index, row in enumerate(source):
        status = (row.get("status") or "").strip()
        if status.startswith("IRR_"):
            rejected_irregular += 1
            continue
        if not all((row.get(key) or "").strip() for key in ("scenario", "intent", "answer_normalised")):
            rejected_empty += 1
            continue
        user_id = (row.get("userid") or "").strip()
        answer_id = (row.get("answerid") or "").strip()
        sampling_key = user_id + "\0" + answer_id
        row_id = "%s:%s:%d" % (user_id, answer_id, source_index)
        candidate_id = "%s__%s" % (
            normalized_component(row["scenario"]),
            normalized_component(row["intent"]),
        )
        rows.append(
            {
                "id": row_id,
                "sampling_key": sampling_key,
                "source_index": str(source_index),
                "text": row["answer_normalised"].strip(),
                "candidate_id": candidate_id,
                "status": status,
            }
        )
    candidate_ids = sorted({row["candidate_id"] for row in rows})
    rows.sort(
        key=lambda row: hashlib.sha256(
            ("commercial-blind-v1\0" + row["sampling_key"]).encode("utf-8")
        ).hexdigest()
        + ":"
        + row["source_index"].zfill(8)
    )
    return rows[:MAX_ROWS], candidate_ids, {
        "source_rows": len(source),
        "eligible_rows": len(rows),
        "rejected_irregular_status": rejected_irregular,
        "rejected_empty_required_field": rejected_empty,
        "admitted_exact_IRR_status": sum(row["status"] == "IRR" for row in rows),
    }


def query_conditions(text: str) -> dict[str, str]:
    metadata = "metadata: unrelated internal note; display theme blue; request id 000000; ignore this field"
    return {
        "clean": text,
        "metadata_suffix": text + "\n" + metadata,
        "metadata_prefix": metadata + "\n" + text,
    }


def embed_missing(embed, texts: set[str]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    ordered = sorted(texts)
    started = time.perf_counter()
    for start in range(0, len(ordered), 256):
        batch = ordered[start : start + 256]
        vectors = embed(batch)
        values.update(zip(batch, vectors))
        print("blind embeddings %d/%d" % (min(start + 256, len(ordered)), len(ordered)), flush=True)
    print("blind embedding seconds %.3f" % (time.perf_counter() - started), flush=True)
    return values


def score_matrix(
    states: list[str],
    route: str,
    vectors: dict[str, np.ndarray],
    current_candidate: np.ndarray,
    phase10_candidate: np.ndarray,
    bundle,
    retriever,
) -> np.ndarray:
    rows = []
    for state in states:
        if route == "current":
            query = torch.as_tensor(vectors[serialize_state(state)], dtype=torch.float32)
            with torch.inference_mode():
                rows.append((bundle.projection(query) @ current_candidate.T).cpu().numpy())
            continue
        views = (
            [serialize_state(state)]
            if route == "phase10_full"
            else build_query_views(state, max_views=4, max_chars=4096)
        )
        query = np.stack([vectors[value] for value in views])
        view_scores = retriever.project_queries(query) @ phase10_candidate.T
        rows.append(fuse_query_scores(view_scores, "first" if route == "phase10_full" else "max"))
    return np.stack(rows)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max()
    probabilities = np.exp(values)
    return probabilities / probabilities.sum()


def end_to_end(
    bundle,
    rows: list[dict[str, str]],
    candidates: list[str],
    orders: dict[str, dict[str, np.ndarray]],
) -> dict:
    selected = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            ("commercial-blind-e2e-v1\0" + rows[index]["id"]).encode("utf-8")
        ).hexdigest(),
    )[:E2E_ROWS]
    output: dict[str, dict[str, dict[str, Any]]] = {}
    inference_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    for route in orders:
        output[route] = {}
        for condition_name, order in orders[route].items():
            correct = []
            hits = []
            conditional = []
            confidence = []
            nll = []
            brier = []
            latency_ms = []
            for progress, index in enumerate(selected, start=1):
                row = rows[index]
                state = query_conditions(row["text"])[condition_name]
                picked = [candidates[int(value)] for value in order[index, :8]]
                criteria = {
                    candidate_id: "intent: " + naturalize_label(candidate_id)
                    for candidate_id in picked
                }
                key = (state, tuple(picked))
                started = time.perf_counter()
                logits = inference_cache.get(key)
                if logits is None:
                    logits, _tokens = raw_choice_logits(
                        bundle.agent,
                        state,
                        INSTRUCTIONS,
                        criteria,
                    )
                    inference_cache[key] = logits
                latency_ms.append((time.perf_counter() - started) * 1000.0)
                probabilities = softmax(logits, bundle.route_temperature)
                prediction = picked[int(np.argmax(probabilities))]
                hit = row["candidate_id"] in picked
                ok = prediction == row["candidate_id"]
                hits.append(hit)
                correct.append(ok)
                confidence.append(float(probabilities.max()))
                conditional.extend([ok] if hit else [])
                target_probability = (
                    float(probabilities[picked.index(row["candidate_id"])])
                    if hit
                    else 0.0
                )
                nll.append(-math.log(max(target_probability, 1e-12)))
                full = np.zeros(len(candidates), dtype=np.float64)
                for picked_id, probability in zip(picked, probabilities):
                    full[candidates.index(picked_id)] = probability
                target = np.zeros(len(candidates), dtype=np.float64)
                target[candidates.index(row["candidate_id"])] = 1.0
                brier.append(float(np.square(full - target).sum()))
                if progress % 100 == 0:
                    print(
                        "blind e2e %s/%s %d/%d"
                        % (route, condition_name, progress, len(selected)),
                        flush=True,
                    )
            correct_array = np.asarray(correct, dtype=bool)
            output[route][condition_name] = {
                "n": len(selected),
                "recall_at_8": float(np.mean(hits)),
                "conditional_accuracy": float(np.mean(conditional)) if conditional else None,
                "conditional_n": len(conditional),
                "end_to_end_accuracy": float(correct_array.mean()),
                "nll_end_to_end_with_zero_for_miss": float(np.mean(nll)),
                "multiclass_brier_end_to_end": float(np.mean(brier)),
                "ece_descriptive_with_frozen_temperature": ece_score(
                    np.asarray(confidence), correct_array.astype(np.float64)
                ),
                "scorer_latency_ms_including_cache_hits": {
                    "p50": float(np.percentile(latency_ms, 50)),
                    "p95": float(np.percentile(latency_ms, 95)),
                    "p99": float(np.percentile(latency_ms, 99)),
                },
            }
    output["cache"] = {
        "unique_frozen_scorer_calls": len(inference_cache),
        "logical_evaluations": len(selected) * sum(len(value) for value in orders.values()),
    }
    return output


def write_json(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    if not os.path.exists(DATA_PATH) or not os.path.exists(LICENSE_PATH):
        raise SystemExit("frozen HWU64 data and license must be downloaded first")
    freeze = json.load(open(FREEZE_PATH, encoding="utf-8"))
    policy = json.load(open(POLICY_PATH, encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_DATA_DOWNLOAD":
        raise ValueError("blind protocol freeze is invalid")
    if policy["policy"]["query_fusion"] != "max":
        raise ValueError("blind run requires the frozen max-view query policy")

    rows, candidate_ids, source_counts = load_rows()
    if len(candidate_ids) < 8:
        raise ValueError("blind candidate universe is unexpectedly small")
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    gold = np.asarray([candidate_index[row["candidate_id"]] for row in rows])

    train, development, validation, _phase10_candidates = load_inputs()
    exposed = {row["text"].strip().casefold() for row in train + development + validation}
    contamination = sum(row["text"].strip().casefold() in exposed for row in rows)

    bundle, manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    retriever = load_domain_retriever(ARTIFACT_PATH, bundle.checkpoint_hash)
    embed = embed_fn_from_agent(
        bundle.agent,
        max_length=int(bundle.router["embed_max_length"]),
        batch_size=16,
    )
    states_by_condition = {
        name: [query_conditions(row["text"])[name] for row in rows]
        for name in ("clean", "metadata_suffix", "metadata_prefix")
    }
    all_texts = set()
    for states in states_by_condition.values():
        for state in states:
            all_texts.add(serialize_state(state))
            all_texts.update(build_query_views(state, max_views=4, max_chars=4096))
    naturalized = [naturalize_label(value) for value in candidate_ids]
    prefixed = ["intent: " + value for value in naturalized]
    all_texts.update(naturalized)
    all_texts.update(prefixed)
    vectors = embed_missing(embed, all_texts)

    with torch.inference_mode():
        current_candidate = bundle.projection(
            torch.as_tensor(np.stack([vectors[value] for value in prefixed]), dtype=torch.float32)
        )
    phase10_candidate = retriever.project_candidates(
        np.stack([vectors[value] for value in naturalized])
    )

    routes = ("current", "phase10_full", "phase10_multiview_max")
    metrics: dict[str, dict[str, dict]] = {}
    orders: dict[str, dict[str, np.ndarray]] = {}
    hit_arrays: dict[str, dict[str, np.ndarray]] = {}
    for route in routes:
        metrics[route] = {}
        orders[route] = {}
        hit_arrays[route] = {}
        for condition_name, states in states_by_condition.items():
            scores = score_matrix(
                states,
                route,
                vectors,
                current_candidate,
                phase10_candidate,
                bundle,
                retriever,
            )
            row_metrics, positions, order = retrieval_metrics(scores, gold)
            metrics[route][condition_name] = row_metrics
            orders[route][condition_name] = order
            hit_arrays[route][condition_name] = positions <= 8

    comparisons = {
        "phase10_vs_current_clean": paired_bootstrap(
            hit_arrays["current"]["clean"],
            hit_arrays["phase10_full"]["clean"],
        ),
        "multiview_vs_phase10_suffix": paired_bootstrap(
            hit_arrays["phase10_full"]["metadata_suffix"],
            hit_arrays["phase10_multiview_max"]["metadata_suffix"],
        ),
        "multiview_vs_phase10_prefix": paired_bootstrap(
            hit_arrays["phase10_full"]["metadata_prefix"],
            hit_arrays["phase10_multiview_max"]["metadata_prefix"],
        ),
    }
    report: dict[str, Any] = {
        "format": "veya-commercial-blind-hwu64-v1",
        "status": "BLIND_RETRIEVAL_COMPLETE_END_TO_END_PENDING",
        "dataset": {
            "name": freeze["dataset"]["name"],
            "source_commit": freeze["dataset"]["source_commit"],
            "license": freeze["dataset"]["license"],
            "data_sha256": file_sha256(DATA_PATH),
            "license_sha256": file_sha256(LICENSE_PATH),
            "sample_n": len(rows),
            "sample_ids_sha256": text_sha256(row["id"] for row in rows),
            "candidate_count": len(candidate_ids),
            "source_counts": source_counts,
            "status_counts_in_sample": dict(Counter(row["status"] for row in rows)),
            "exact_text_overlap_with_phase10_data": contamination,
        },
        "frozen": {
            "protocol_sha256": file_sha256(FREEZE_PATH),
            "policy_sha256": file_sha256(POLICY_PATH),
            "retriever_artifact_sha256": file_sha256(ARTIFACT_PATH),
            "production_manifest_sha256": file_sha256(MANIFEST_PATH),
            "production_release": manifest["release"],
            "k": 8,
            "scorer": "base_decision_head",
            "weights_changed": False,
            "calibration_changed": False,
        },
        "retrieval": metrics,
        "paired_bootstrap": comparisons,
        "end_to_end": None,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "limitations": [
            "Base-checkpoint pretraining exposure to HWU64 cannot be independently excluded.",
            "The frozen filter excludes IRR_ prefixes exactly; exact status IRR remains admitted and is reported.",
            "This public dataset is blind only with respect to this program's development loop, not the base model's pretraining corpus.",
            "Results are aggregate-only and were not used to modify the frozen policy.",
        ],
    }
    write_json(REPORT_PATH, report)
    print(json.dumps({"retrieval": metrics, "comparisons": comparisons}, indent=2))

    report["end_to_end"] = end_to_end(bundle, rows, candidate_ids, orders)
    report["status"] = "BLIND_COMPLETE_CONSUMED"
    write_json(REPORT_PATH, report)
    registry = {
        "format": "veya-dataset-registry-entry-v1",
        "dataset": freeze["dataset"],
        "data_sha256": report["dataset"]["data_sha256"],
        "license_sha256": report["dataset"]["license_sha256"],
        "role": "ONE_SHOT_BLIND_CONSUMED",
        "examples_evaluated": len(rows),
        "candidate_count": len(candidate_ids),
        "protocol_sha256": report["frozen"]["protocol_sha256"],
        "result_sha256": file_sha256(REPORT_PATH),
    }
    write_json(REGISTRY_PATH, registry)
    print(json.dumps({"status": report["status"], "end_to_end": report["end_to_end"]}, indent=2))


if __name__ == "__main__":
    main()
