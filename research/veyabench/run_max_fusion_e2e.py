"""End-to-end check of Phase-10 max fusion with a frozen decision head.

No training. No calibration refit. No production writes. No sealed blind data.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase10_data import DATA_OUT, PHASE10, ROOT
from research.veyabench.retrieval_robustness import (
    apply_perturbation,
    adversarial_suffix,
    choose_wrong_index,
    classify_outcome,
    fusion_recommendation,
    query_views,
)
from research.veyabench.run_phase10_train import (
    CACHE_META_PATH,
    MANIFEST_PATH,
    _embedding_texts,
    encode_or_load,
    model_scores,
    paired_bootstrap,
    retrieval_metrics,
)
from research.veyabench.run_retrieval_robustness import encode_missing, permutation_delta, stack
from veya.common import QTYPES, build_sequence, collate_items, ece_score
from veya.experimental.domain_retriever import load_domain_retriever, naturalize_label, text_sha256
from veya.release_bundle import INSTRUCTIONS, load_release_manifest

OUT = os.path.join(ROOT, "artifacts", "experiments", "phase10_max_fusion_e2e")
ARTIFACT = os.path.join(PHASE10, "domain_general_k8.pt")
CONDITIONS = ("clean", "short_suffix", "long_suffix", "prefix", "adversarial_wrong_intent_suffix")
METHODS = ("production", "phase10_full", "phase10_max_fusion")
BATCH = 16


def _rows(name: str) -> list[dict]:
    with open(os.path.join(DATA_OUT, name + ".jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def batch_logits(agent, states: list[str], criteria_list: list[dict]) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    for start in range(0, len(states), BATCH):
        chunk_states = states[start : start + BATCH]
        chunk_criteria = criteria_list[start : start + BATCH]
        items = []
        widths = []
        for state, criteria in zip(chunk_states, chunk_criteria):
            question = {"t": "choice", "ins": INSTRUCTIONS, "crit": criteria}
            seq, markers = build_sequence(agent.tok, state, question, max_len, head_max_len, option_token_cap=48)
            if len(markers) != len(criteria):
                raise ValueError("packed head dropped an option")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
            widths.append(len(criteria))
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.inference_mode():
            logits, _act = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )
        logits = logits.float().cpu().numpy()
        for row_index, width in enumerate(widths):
            outputs.append(logits[row_index, :width])
    return outputs


def score_queries(name: str, bundle, phase10, queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if name == "production":
        with torch.inference_mode():
            q = bundle.projection(torch.as_tensor(queries, dtype=torch.float32))
            c = bundle.projection(torch.as_tensor(candidates, dtype=torch.float32))
            return (q @ c.T).cpu().numpy()
    return model_scores(phase10, queries, candidates)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    train = _rows("train")
    development = _rows("development")
    validation = _rows("schema_held_out_validation")
    with open(os.path.join(DATA_OUT, "candidates.json"), encoding="utf-8") as handle:
        all_candidates = json.load(handle)
    dataset_hash = hashlib.sha256(
        open(os.path.join(DATA_OUT, "schema_held_out_validation.jsonl"), "rb").read()
    ).hexdigest()
    bundle, _manifest = load_release_manifest(MANIFEST_PATH, device="cpu", root=ROOT)
    clean_texts, clean_index = _embedding_texts(train, development, validation, all_candidates)
    with open(CACHE_META_PATH, encoding="utf-8") as handle:
        clean_meta = json.load(handle)
    if clean_meta.get("texts_sha256") != text_sha256(clean_texts):
        raise SystemExit("frozen embedding cache does not match the rebuilt text list")
    clean_vectors = encode_or_load(bundle, clean_texts)
    phase10 = load_domain_retriever(ARTIFACT, bundle.checkpoint_hash).projection.eval()

    validation_ids = {row["candidate_id"] for row in validation}
    candidates = sorted(
        (row for row in all_candidates if row["candidate_id"] in validation_ids),
        key=lambda row: row["candidate_id"],
    )
    if len(candidates) != 150 or len(validation) != 3000:
        raise SystemExit("unexpected held-out shape")
    gold_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    gold = np.asarray([gold_index[row["candidate_id"]] for row in validation], dtype=np.int64)
    naturalized = [row["naturalized"] for row in candidates]
    candidate_vectors = np.stack(
        [clean_vectors[clean_index[row["naturalized"]]] for row in candidates]
    ).astype(np.float32)
    clean_queries = np.stack([clean_vectors[clean_index[row["text"]]] for row in validation])

    production_clean = score_queries("production", bundle, phase10, clean_queries, candidate_vectors)
    clean_order = np.argsort(-production_clean, axis=1, kind="stable")
    wrong_index = np.asarray(
        [
            choose_wrong_index(clean_order[row_index].tolist(), int(gold[row_index]), naturalized)
            for row_index in range(len(validation))
        ],
        dtype=np.int64,
    )
    if np.any(wrong_index == gold):
        raise SystemExit("adversarial intent collided with gold")

    observed: dict[str, list[str]] = {}
    view_texts: dict[str, dict[str, list[str]]] = {}
    for kind in CONDITIONS:
        texts = []
        views_for_kind: dict[str, list[str]] = {}
        for row_index, row in enumerate(validation):
            if kind == "clean":
                perturbed = row["text"]
            elif kind == "adversarial_wrong_intent_suffix":
                perturbed = row["text"] + adversarial_suffix(naturalized[int(wrong_index[row_index])])
            else:
                perturbed = apply_perturbation(row["text"], kind)
            texts.append(perturbed)
            views = query_views(perturbed)
            for name, value in views.items():
                views_for_kind.setdefault(name, []).append(value)
        observed[kind] = texts
        view_texts[kind] = views_for_kind
        if any(len(values) != len(validation) for values in views_for_kind.values()):
            raise SystemExit("view lists diverged")

    needed = []
    for kind, views in view_texts.items():
        if kind == "clean":
            continue
        for values in views.values():
            needed.extend(values)
    encoded = encode_missing(bundle, sorted(set(needed)), os.path.join(OUT, "view_embeddings.npz"))

    def queries_for(kind: str, view: str) -> np.ndarray:
        if kind == "clean" and view == "full":
            return clean_queries
        return stack(encoded, view_texts[kind][view])

    methods_scores: dict[str, dict[str, np.ndarray]] = {name: {} for name in METHODS}
    retrieval_seconds = {}
    for kind in CONDITIONS:
        banks = {view: queries_for(kind, view) for view in view_texts[kind]}
        for method in ("production", "phase10_full"):
            started = time.perf_counter()
            methods_scores[method][kind] = score_queries(
                "production" if method == "production" else "phase10",
                bundle,
                phase10,
                banks["full"],
                candidate_vectors,
            )
            retrieval_seconds[(method, kind)] = time.perf_counter() - started
        started = time.perf_counter()
        parts = [
            score_queries("phase10", bundle, phase10, vectors, candidate_vectors)
            for vectors in banks.values()
        ]
        methods_scores["phase10_max_fusion"][kind] = np.max(np.stack(parts, axis=0), axis=0)
        retrieval_seconds[("phase10_max_fusion", kind)] = time.perf_counter() - started
        if permutation_delta(methods_scores["phase10_max_fusion"][kind]) != 0.0:
            raise SystemExit("fusion scores changed under candidate permutation")

    temperature = float(bundle.route_temperature)
    summary = {}
    predictions = {}
    changes = []
    category_counts = {}
    for method in METHODS:
        summary[method] = {}
        predictions[method] = {}
        for kind in CONDITIONS:
            scores = methods_scores[method][kind]
            metrics, positions, order = retrieval_metrics(scores, gold)
            states = []
            criteria_list = []
            picked_ids = []
            for row_index, row in enumerate(validation):
                picked = [candidates[int(index)]["candidate_id"] for index in order[row_index, :8]]
                picked_ids.append(picked)
                criteria_list.append(
                    {
                        candidate_id: "intent: " + naturalize_label(candidate_id.split(":", 1)[1])
                        for candidate_id in picked
                    }
                )
                states.append(observed[kind][row_index])
            started = time.perf_counter()
            logit_rows = batch_logits(bundle.agent, states, criteria_list)
            decision_seconds = time.perf_counter() - started
            correct = []
            hits = []
            confidences = []
            conditional_nll = []
            conditional_brier = []
            wrong_top8 = []
            wrong_top1 = []
            gold_evicted = []
            flips = []
            harm = []
            clean_correct = predictions[method].get("clean") if kind != "clean" else None
            for row_index, row in enumerate(validation):
                probs = _softmax(logit_rows[row_index], temperature)
                labels = list(criteria_list[row_index])
                prediction = labels[int(np.argmax(probs))]
                hit = row["candidate_id"] in labels
                ok = prediction == row["candidate_id"]
                hits.append(hit)
                correct.append(ok)
                confidences.append(float(probs.max()))
                if hit:
                    gold_pos = labels.index(row["candidate_id"])
                    conditional_nll.append(float(-np.log(max(probs[gold_pos], 1e-12))))
                    target = np.zeros(len(labels))
                    target[gold_pos] = 1.0
                    conditional_brier.append(float(np.sum((probs - target) ** 2)))
                injected = candidates[int(wrong_index[row_index])]["candidate_id"]
                wrong_top8.append(injected in labels)
                wrong_top1.append(order[row_index, 0] == wrong_index[row_index])
                gold_evicted.append(not hit)
                if clean_correct is not None:
                    flips.append(bool(clean_correct[row_index]) != ok)
                    harm.append(bool(clean_correct[row_index]) and not ok)
                if method == "phase10_max_fusion":
                    base_labels = [
                        candidates[int(index)]["candidate_id"]
                        for index in np.argsort(-methods_scores["production"][kind][row_index], kind="stable")[:8]
                    ]
                    base_hit = row["candidate_id"] in base_labels
                    outcome = classify_outcome(
                        baseline_correct=bool(summary["production"][kind]["correct"][row_index]),
                        fusion_correct=ok,
                        baseline_hit=base_hit,
                        fusion_hit=hit,
                        adversarial=kind == "adversarial_wrong_intent_suffix",
                        prediction=prediction,
                        injected=injected,
                    )
                    category_counts.setdefault(kind, {})
                    category_counts[kind][outcome["primary"]] = category_counts[kind].get(outcome["primary"], 0) + 1
                    if outcome["retrieval_recovery"]:
                        category_counts[kind]["RETRIEVAL_RECOVERY"] = category_counts[kind].get("RETRIEVAL_RECOVERY", 0) + 1
                    if outcome["retrieval_loss"]:
                        category_counts[kind]["RETRIEVAL_LOSS"] = category_counts[kind].get("RETRIEVAL_LOSS", 0) + 1
                    if outcome["adversarial_capture"]:
                        category_counts[kind]["ADVERSARIAL_CAPTURE"] = category_counts[kind].get("ADVERSARIAL_CAPTURE", 0) + 1
                    if outcome["primary"] != "UNCHANGED_CORRECT" or outcome["adversarial_capture"]:
                        changes.append(
                            {
                                "id_sha256": hashlib.sha256(row["id"].encode("utf-8")).hexdigest(),
                                "condition": kind,
                                "gold": row["candidate_id"],
                                "injected": injected if kind == "adversarial_wrong_intent_suffix" else None,
                                "baseline_prediction": summary["production"][kind]["prediction"][row_index],
                                "fusion_prediction": prediction,
                                "baseline_hit": base_hit,
                                "fusion_hit": hit,
                                "gold_rank_baseline": int(retrieval_metrics(methods_scores["production"][kind][row_index : row_index + 1], gold[row_index : row_index + 1])[1][0]),
                                "gold_rank_fusion": int(positions[row_index]),
                                "category": outcome["primary"],
                                "adversarial_capture": outcome["adversarial_capture"],
                            }
                        )
            correct_array = np.asarray(correct, dtype=bool)
            summary[method][kind] = {
                "retrieval": metrics,
                "recall_at_8_from_shortlist": float(np.mean(hits)),
                "conditional_accuracy": float(np.mean(correct_array[np.asarray(hits)])) if any(hits) else None,
                "conditional_n": int(sum(hits)),
                "conditional_nll": float(np.mean(conditional_nll)) if conditional_nll else None,
                "conditional_brier": float(np.mean(conditional_brier)) if conditional_brier else None,
                "end_to_end_accuracy": float(correct_array.mean()),
                "retrieval_misses": int((~np.asarray(hits)).sum()),
                "scorer_errors_when_retrieved": int((np.asarray(hits) & ~correct_array).sum()),
                "ece": float(ece_score(np.asarray(confidences), correct_array.astype(np.float64))),
                "wrong_intent_top8_rate": float(np.mean(wrong_top8)) if kind == "adversarial_wrong_intent_suffix" else None,
                "wrong_intent_top1_rate": float(np.mean(wrong_top1)) if kind == "adversarial_wrong_intent_suffix" else None,
                "gold_eviction_rate": float(np.mean(gold_evicted)) if kind == "adversarial_wrong_intent_suffix" else None,
                "decision_flip_rate": float(np.mean(flips)) if flips else None,
                "clean_correct_to_adversarial_wrong_rate": float(np.mean(harm)) if harm else None,
                "retrieval_seconds": retrieval_seconds[(method, kind)],
                "decision_seconds": decision_seconds,
                "decision_mean_ms": decision_seconds * 1000.0 / len(validation),
                "correct": correct,
                "prediction": None,
            }
            # Store predictions only long enough for the fusion comparison, then drop.
            summary[method][kind]["prediction"] = [
                list(criteria_list[row_index])[int(np.argmax(_softmax(logit_rows[row_index], temperature)))]
                for row_index in range(len(validation))
            ]
            predictions[method][kind] = correct_array
            if method == "production":
                summary[method][kind]["correct"] = correct
            print(method, kind, "e2e", round(float(correct_array.mean()), 4), "R8", round(metrics["recall_at_8"], 4), flush=True)

    def delta(method: str, kind: str, field: str) -> dict:
        if field == "recall_at_8":
            fusion_hits = retrieval_metrics(methods_scores["phase10_max_fusion"][kind], gold)[1] <= 8
            base_hits = retrieval_metrics(methods_scores[method][kind], gold)[1] <= 8
            return paired_bootstrap(base_hits, fusion_hits)
        return paired_bootstrap(predictions[method][kind], predictions["phase10_max_fusion"][kind])

    paired = {}
    for other in ("production", "phase10_full"):
        paired[other] = {}
        for kind in ("clean", "short_suffix", "long_suffix", "adversarial_wrong_intent_suffix"):
            paired[other][kind] = {
                "end_to_end_accuracy": delta(other, kind, "e2e"),
                "recall_at_8": delta(other, kind, "recall_at_8"),
            }

    fusion = summary["phase10_max_fusion"]
    production = summary["production"]
    phase10_full = summary["phase10_full"]
    recommendation = fusion_recommendation(
        {
            "short_e2e_delta": fusion["short_suffix"]["end_to_end_accuracy"] - production["short_suffix"]["end_to_end_accuracy"],
            "long_e2e_delta": fusion["long_suffix"]["end_to_end_accuracy"] - production["long_suffix"]["end_to_end_accuracy"],
            "clean_e2e_delta": fusion["clean"]["end_to_end_accuracy"] - production["clean"]["end_to_end_accuracy"],
            "long_recall8_delta_vs_phase10_full": fusion["long_suffix"]["retrieval"]["recall_at_8"] - phase10_full["long_suffix"]["retrieval"]["recall_at_8"],
            "fusion_harm_rate": fusion["adversarial_wrong_intent_suffix"]["clean_correct_to_adversarial_wrong_rate"],
            "production_harm_rate": production["adversarial_wrong_intent_suffix"]["clean_correct_to_adversarial_wrong_rate"],
            "fusion_wrong_top1": fusion["adversarial_wrong_intent_suffix"]["wrong_intent_top1_rate"],
            "production_wrong_top1": production["adversarial_wrong_intent_suffix"]["wrong_intent_top1_rate"],
        }
    )
    def public(block: dict) -> dict:
        return {key: value for key, value in block.items() if key not in {"correct", "prediction"}}

    payload = {
        "experiment": "phase10-max-fusion-e2e-v1",
        "status": recommendation,
        "promoted": False,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "name": "phase10 schema_held_out_validation",
            "split": "clinc150 train schema holdout",
            "examples": len(validation),
            "candidates": len(candidates),
            "sha256": dataset_hash,
            "blind_set_used": False,
        },
        "artifacts": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "checkpoint_hash": bundle.checkpoint_hash,
            "phase10_retriever_sha256": hashlib.sha256(open(ARTIFACT, "rb").read()).hexdigest(),
            "temperature": temperature,
            "temperature_refit": False,
            "k": 8,
            "fusion": "max",
            "views": ["full", "head", "tail", "json_text"],
            "instructions": INSTRUCTIONS,
            "seed": 0,
            "device": "cpu",
            "platform": platform.platform(),
        },
        "methods": {method: {kind: public(summary[method][kind]) for kind in CONDITIONS} for method in METHODS},
        "paired_deltas": paired,
        "failure_analysis": category_counts,
        "changed_outcomes": len(changes),
        "recommendation": recommendation,
    }
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(OUT, "changes.jsonl"), "w", encoding="utf-8") as handle:
        for row in changes:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"recommendation": recommendation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
