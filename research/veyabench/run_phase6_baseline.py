"""Freeze the Phase-5 reranker and measure why K=8 decisions fail.

Validation and the confusion graph come from the Banking77 train file.
Release-test predictions are saved and are not mined for negatives.
"""
import json
import os
import random
import subprocess

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase6_protocol import (
    CALIBRATION, LOCKED_TEST, RELEASE_TEST, RERANKER_TRAIN, RERANKER_VAL,
    confusion_graph, example_id, pair_relation, split_hash, top_confused_partners, training_slice_for,
)
from research.veyabench.phase6_score import score_sets
from research.veyabench.run_phase3_eval import ROOT, SEED, load_banking
from research.veyabench.run_release_eval import load_from_manifest
from veya.common import build_sequence, ece_score
from veya.release_bundle import INSTRUCTIONS, raw_choice_logits

ART = os.path.join(ROOT, "artifacts", "experiments", "phase6")
TRAIN_GRAPH_ROWS = 1600


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def rank_queries(bundle, texts, k=8):
    queries = bundle.project(bundle.embed(list(texts)))
    labels = bundle.project(bundle.embed(list(bundle.label_texts)))
    scores = queries @ labels.T
    index = torch.tensor(np.asarray(scores)).topk(k, dim=1).indices.tolist()
    return index, labels


def attach(rows, picks, scored, temperature_name):
    out = []
    for (text, truth), picked, record in zip(rows, picks, scored):
        hit = truth in record["candidates"]
        correct = record["prediction"] == truth
        gold = record["candidates"].index(truth) if hit else None
        nll = None if gold is None else float(-np.log(record["probabilities"][gold] + 1e-12))
        brier = None
        if gold is not None:
            truth_vec = np.zeros(len(record["probabilities"]))
            truth_vec[gold] = 1.0
            brier = float(np.sum((np.array(record["probabilities"]) - truth_vec) ** 2))
        out.append({
            "target": truth,
            "full_candidate_count": 77,
            "retrieved_top8": list(picked),
            "target_in_top8": hit,
            "reranker_logits": record["logits"],
            "reranker_probabilities": record["probabilities"],
            "predicted_label": record["prediction"],
            "confidence": record["confidence"],
            "margin": record["margin"],
            "correct": correct,
            "nll_if_retrieved": nll,
            "brier_if_retrieved": brier,
            "input_tokens": record["input_tokens"],
            "temperature": temperature_name,
        })
    return out


def summarize(records):
    correct = np.array([1.0 if row["correct"] else 0.0 for row in records])
    conf = np.array([row["confidence"] for row in records], dtype=np.float64)
    margin = np.array([row["margin"] for row in records], dtype=np.float64)
    hits = [row for row in records if row["target_in_top8"]]
    hit_correct = np.array([1.0 if row["correct"] else 0.0 for row in hits])
    nlls = [row["nll_if_retrieved"] for row in hits if row["nll_if_retrieved"] is not None]
    briers = [row["brier_if_retrieved"] for row in hits if row["brier_if_retrieved"] is not None]
    wrong_margin = margin[(correct == 0) & np.array([row["target_in_top8"] for row in records])]
    right_margin = margin[(correct == 1)]
    return {
        "n": len(records),
        "retrieval_recall_at_8": float(np.mean([row["target_in_top8"] for row in records])),
        "conditional_accuracy": float(hit_correct.mean()) if len(hit_correct) else None,
        "end_to_end_accuracy": float(correct.mean()) if len(records) else None,
        "nll_on_hits": float(np.mean(nlls)) if nlls else None,
        "brier_on_hits": float(np.mean(briers)) if briers else None,
        "ece_honest": float(ece_score(conf, correct)) if len(records) else None,
        "mean_confidence": float(conf.mean()) if len(records) else None,
        "high_conf_wrong_0.70": int(np.sum((correct == 0) & (conf >= 0.70))),
        "high_conf_wrong_0.80": int(np.sum((correct == 0) & (conf >= 0.80))),
        "high_conf_wrong_0.90": int(np.sum((correct == 0) & (conf >= 0.90))),
        "median_margin_correct": float(np.median(right_margin)) if len(right_margin) else None,
        "median_margin_incorrect_retrieved": float(np.median(wrong_margin)) if len(wrong_margin) else None,
        "mean_input_tokens": float(np.mean([row["input_tokens"] for row in records])),
    }


def taxonomy(records, vectors):
    mistakes = [row for row in records if row["target_in_top8"] and not row["correct"]]
    relations = {}
    pairs = []
    for row in mistakes:
        rel = pair_relation(row["target"], row["predicted_label"], vectors)
        relations[rel["relation"]] = relations.get(rel["relation"], 0) + 1
        pairs.append((row["target"], row["predicted_label"]))
    edges = confusion_graph(pairs)
    return {
        "reranker_mistakes": len(mistakes),
        "relations": relations,
        "top_edges": edges[:30],
    }


def shuffled(labels, rng):
    copied = list(labels)
    rng.shuffle(copied)
    return copied


def random_set(labels, truth, rng):
    others = [label for label in labels if label != truth]
    rng.shuffle(others)
    return shuffled([truth] + others[:7], rng)


def nearest_set(label_names, truth, projected, rng):
    index = label_names.index(truth)
    sims = projected @ projected[index]
    order = np.argsort(-np.asarray(sims))
    picked = [truth]
    for position in order:
        name = label_names[int(position)]
        if name != truth:
            picked.append(name)
        if len(picked) == 8:
            break
    return shuffled(picked, rng)


def confusion_set(truth, partners, shortlist, label_names, rng):
    picked = [truth]
    for partner in partners:
        if partner not in picked and partner in label_names:
            picked.append(partner)
    for label in shortlist:
        if label not in picked:
            picked.append(label)
        if len(picked) == 8:
            break
    for label in label_names:
        if len(picked) == 8:
            break
        if label not in picked:
            picked.append(label)
    return shuffled(picked[:8], rng)


def guaranteed(rows, picks, scored):
    """Analysis-only sets where a missing target replaces the last candidate."""
    forced = []
    need = []
    need_rows = []
    for i, ((text, truth), picked) in enumerate(zip(rows, picks)):
        if truth in picked:
            forced.append(picked)
        else:
            replaced = list(picked[:-1]) + [truth]
            forced.append(replaced)
            need.append(i)
            need_rows.append((text, replaced))
    return forced, need, need_rows


def token_audit(agent, rows, picks, crit):
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    identical = 0
    state_full, state_kept = [], []
    option_full, option_kept = [], []
    lengths = []
    shrunk = 0
    for (text, _truth), picked in zip(rows, picks):
        criteria = {label: crit[label] for label in picked}
        question = {"t": "choice", "ins": INSTRUCTIONS, "crit": criteria}
        standard, _markers = build_sequence(agent.tok, text, question, max_len, head_max_len, option_token_cap=48)
        wider, _markers_b = build_sequence(agent.tok, text, question, max_len, 512, option_token_cap=128)
        if standard == wider:
            identical += 1
        lengths.append(len(standard))
        full_state = agent.tok(text, add_special_tokens=False)["input_ids"]
        # State sits between the last two separators. Compare the budgeted sequence
        # with a sequence that asks for the whole state by using a huge max_len
        # only as a length probe, capped at the tokenizer call itself.
        state_full.append(len(full_state))
        # Tokens kept are whatever remained after the head. Reconstruct from room.
        head_ids = agent.tok("choice question: %s" % INSTRUCTIONS, add_special_tokens=False)["input_ids"]
        option_lengths = []
        for label in picked:
            rendered = "%s: %s" % (label, crit[label])
            raw = agent.tok(" " + rendered, add_special_tokens=False)["input_ids"]
            option_full.append(len(raw) + 1)
            option_lengths.append(min(48, len(raw)) + 1)
        if sum(option_lengths) + max(8, len(head_ids)) > head_max_len:
            shrunk += 1
            per = max(4, (head_max_len - 16) // max(1, len(picked)))
            option_kept.extend([min(per, length) for length in option_lengths])
        else:
            option_kept.extend(option_lengths)
        room = max(0, max_len - (1 + min(len(head_ids), head_max_len) + 1 + sum(option_lengths) + 1 + 1))
        state_kept.append(min(len(full_state), room))
    return {
        "n": len(rows),
        "wider_budget_identical_sequences": identical,
        "wider_budget_changed_sequences": len(rows) - identical,
        "mean_sequence_tokens": float(np.mean(lengths)) if lengths else None,
        "mean_state_tokens_available": float(np.mean(state_full)) if state_full else None,
        "mean_state_tokens_kept_estimate": float(np.mean(state_kept)) if state_kept else None,
        "mean_option_tokens_unbounded": float(np.mean(option_full)) if option_full else None,
        "mean_option_tokens_after_cap": float(np.mean(option_kept)) if option_kept else None,
        "sequences_whose_options_hit_the_head_budget": shrunk,
    }


def permutation_report(agent, rows, picks, crit, temperature):
    sample = [(row, picked) for row, picked in zip(rows, picks) if row[1] in picked][:24]
    flips = 0
    trials = 0
    abs_deltas = []
    max_deltas = []
    for (text, truth), picked in sample:
        base = score_sets(agent, [text], [picked], crit, temperature, batch_size=1)[0]
        base_map = dict(zip(base["candidates"], base["probabilities"]))
        for shuffle_id in range(6):
            local = random.Random(SEED + 1000 * shuffle_id + trials)
            order = shuffled(picked, local)
            alt = score_sets(agent, [text], [order], crit, temperature, batch_size=1)[0]
            alt_map = dict(zip(alt["candidates"], alt["probabilities"]))
            if alt["prediction"] != base["prediction"]:
                flips += 1
            deltas = [abs(base_map[label] - alt_map[label]) for label in picked]
            abs_deltas.append(float(np.mean(deltas)))
            max_deltas.append(float(np.max(deltas)))
            trials += 1
    return {
        "examples": len(sample),
        "shuffles_per_example": 6,
        "trials": trials,
        "top1_flip_rate": (flips / trials) if trials else None,
        "mean_abs_probability_delta": float(np.mean(abs_deltas)) if abs_deltas else None,
        "mean_max_probability_delta": float(np.mean(max_deltas)) if max_deltas else None,
    }


def main():
    os.makedirs(ART, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.load(open(os.path.join(ROOT, "artifacts", "release", "manifest.json"), encoding="utf-8"))
    timings = {}
    print("loading", device, flush=True)
    bundle = load_from_manifest(manifest, device, timings)
    train = load_banking("train")
    test = load_banking("test")
    if len(train) < RERANKER_VAL[1]:
        raise SystemExit("train file has %d rows, phase-6 val expects %d" % (len(train), RERANKER_VAL[1]))
    if len(test) < RELEASE_TEST[1]:
        raise SystemExit("test file has %d rows, release test expects %d" % (len(test), RELEASE_TEST[1]))
    for index in range(RERANKER_TRAIN[0], RERANKER_VAL[1]):
        training_slice_for(index)
    labels = list(bundle.labels)
    crit = {label: text for label, text in zip(labels, bundle.label_texts)}
    temperature = bundle.route_temperature
    val_rows = train[RERANKER_VAL[0]:RERANKER_VAL[1]]
    graph_rows = train[RERANKER_TRAIN[0]:RERANKER_TRAIN[0] + TRAIN_GRAPH_ROWS]
    print("ranking validation and train graph", flush=True)
    val_index, projected = rank_queries(bundle, [row[0] for row in val_rows], 8)
    graph_index, _projected = rank_queries(bundle, [row[0] for row in graph_rows], 8)
    val_picks = [[labels[i] for i in row] for row in val_index]
    graph_picks = [[labels[i] for i in row] for row in graph_index]
    vectors = {label: projected[i] for i, label in enumerate(labels)}

    print("score validation shortlist", flush=True)
    val_scored = score_sets(agent=bundle.agent, texts=[row[0] for row in val_rows], picked_lists=val_picks,
                            crit=crit, temperature=temperature, batch_size=4)
    probe = raw_choice_logits(bundle.agent, val_rows[0][0], INSTRUCTIONS, {label: crit[label] for label in val_picks[0]})
    if not np.allclose(probe[0], val_scored[0]["logits"], atol=1e-4):
        raise SystemExit("batched logits do not match raw_choice_logits")
    val_records = attach(val_rows, val_picks, val_scored, "route_temperature")
    for offset, record in enumerate(val_records):
        record["example_id"] = example_id("train", RERANKER_VAL[0] + offset)
    val_summary = summarize(val_records)
    val_taxonomy = taxonomy(val_records, vectors)
    print("val conditional", val_summary["conditional_accuracy"], "e2e", val_summary["end_to_end_accuracy"], flush=True)

    print("score train graph sample", flush=True)
    graph_scored = score_sets(bundle.agent, [row[0] for row in graph_rows], graph_picks, crit, temperature, batch_size=4)
    graph_records = attach(graph_rows, graph_picks, graph_scored, "route_temperature")
    for offset, record in enumerate(graph_records):
        record["example_id"] = example_id("train", RERANKER_TRAIN[0] + offset)
    graph_summary = summarize(graph_records)
    graph_taxonomy = taxonomy(graph_records, vectors)
    edges = graph_taxonomy["top_edges"]

    def run_construction(name, picks):
        print("construction", name, flush=True)
        scored = score_sets(bundle.agent, [row[0] for row in val_rows], picks, crit, temperature, batch_size=4)
        records = attach(val_rows, picks, scored, "route_temperature")
        summary = summarize(records)
        summary["construction"] = name
        return summary

    rng_sets = []
    nearest_sets = []
    confusion_sets = []
    for i, ((text, truth), picked) in enumerate(zip(val_rows, val_picks)):
        rng_sets.append(random_set(labels, truth, random.Random(SEED + 17 + i)))
        nearest_sets.append(nearest_set(labels, truth, projected, random.Random(SEED + 29 + i)))
        partners = top_confused_partners(edges, truth, 2)
        confusion_sets.append(confusion_set(truth, partners, picked, labels, random.Random(SEED + 41 + i)))
    constructions = {
        "shortlist": val_summary,
        "random_7_plus_target": run_construction("random_7_plus_target", rng_sets),
        "embedding_nearest_7_plus_target": run_construction("embedding_nearest_7_plus_target", nearest_sets),
        "train_confusion_2_plus_shortlist": run_construction("train_confusion_2_plus_shortlist", confusion_sets),
    }
    forced, need, _need_rows = guaranteed(val_rows, val_picks, val_scored)
    if need:
        print("oracle replacements", len(need), flush=True)
        oracle_scored = score_sets(
            bundle.agent,
            [val_rows[i][0] for i in need],
            [forced[i] for i in need],
            crit,
            temperature,
            batch_size=4,
        )
        oracle_by_index = {i: record for i, record in zip(need, oracle_scored)}
        oracle_records = []
        oracle_picks = []
        for i, (row, picked, record) in enumerate(zip(val_rows, val_picks, val_scored)):
            if i in oracle_by_index:
                oracle_picks.append(forced[i])
                oracle_records.append(oracle_by_index[i])
            else:
                oracle_picks.append(picked)
                oracle_records.append(record)
        constructions["oracle_target_forced_into_top8"] = summarize(attach(val_rows, oracle_picks, oracle_records, "route_temperature"))
    else:
        constructions["oracle_target_forced_into_top8"] = val_summary

    audit_n = min(200, len(val_rows))
    audit = token_audit(bundle.agent, val_rows[:audit_n], val_picks[:audit_n], crit)
    print("token audit identical", audit["wider_budget_identical_sequences"], "/", audit["n"], flush=True)
    if audit["wider_budget_changed_sequences"]:
        print("rescore changed-budget validation", flush=True)
        wider = score_sets(
            bundle.agent, [row[0] for row in val_rows], val_picks, crit, temperature,
            batch_size=4, head_max_len=512, option_token_cap=128,
        )
        constructions["larger_token_budget_same_weights"] = summarize(attach(val_rows, val_picks, wider, "route_temperature"))
    else:
        constructions["larger_token_budget_same_weights"] = dict(val_summary)
        constructions["larger_token_budget_same_weights"]["note"] = "every audited sequence was identical at the larger budget, so the full rescore was skipped"

    print("permutation", flush=True)
    perm = permutation_report(bundle.agent, val_rows[:40], val_picks[:40], crit, temperature)

    print("freeze release-test predictions", flush=True)
    test_rows = test[RELEASE_TEST[0]:RELEASE_TEST[1]]
    test_index, _ = rank_queries(bundle, [row[0] for row in test_rows], 8)
    test_picks = [[labels[i] for i in row] for row in test_index]
    test_scored = score_sets(bundle.agent, [row[0] for row in test_rows], test_picks, crit, temperature, batch_size=4)
    test_records = attach(test_rows, test_picks, test_scored, "route_temperature")
    for offset, record in enumerate(test_records):
        record["example_id"] = "test:%d" % (RELEASE_TEST[0] + offset)
    test_summary = summarize(test_records)
    print("frozen test e2e", test_summary["end_to_end_accuracy"], "conditional", test_summary["conditional_accuracy"], flush=True)

    splits = {
        "shortlist_train": [0, 800],
        "reranker_train": list(RERANKER_TRAIN),
        "calibration": list(CALIBRATION),
        "reranker_val": list(RERANKER_VAL),
        "locked_test_do_not_train": list(LOCKED_TEST),
        "release_test_do_not_train": list(RELEASE_TEST),
        "hashes": {
            "reranker_train": split_hash(train[RERANKER_TRAIN[0]:RERANKER_TRAIN[1]]),
            "calibration": split_hash(train[CALIBRATION[0]:CALIBRATION[1]]),
            "reranker_val": split_hash(train[RERANKER_VAL[0]:RERANKER_VAL[1]]),
        },
        "contamination_note": "The decision to study the reranker used the Phase-5 test error counts. Pair-level negatives in this phase come from train rows 800:2400 only.",
    }
    freeze = {
        "git_commit": commit,
        "working_tree_lines": len(status.splitlines()),
        "manifest_release": manifest.get("release"),
        "checkpoint_sha256": bundle.checkpoint_hash,
        "shortlist_sha256": bundle.shortlist_sha256,
        "route_temperature": temperature,
        "choice_router_hash": manifest["routing"]["choice_router_hash"],
        "device": device,
    }
    dump(os.path.join(ART, "splits.json"), splits)
    dump(os.path.join(ART, "freeze.json"), freeze)
    dump(os.path.join(ART, "val_predictions.json"), val_records)
    dump(os.path.join(ART, "val_summary.json"), val_summary)
    dump(os.path.join(ART, "val_taxonomy.json"), val_taxonomy)
    dump(os.path.join(ART, "train_graph_summary.json"), graph_summary)
    dump(os.path.join(ART, "train_confusion.json"), graph_taxonomy)
    dump(os.path.join(ART, "constructions.json"), constructions)
    dump(os.path.join(ART, "token_audit.json"), audit)
    dump(os.path.join(ART, "permutation_before.json"), perm)
    dump(os.path.join(ART, "frozen_test", "predictions.json"), test_records)
    dump(os.path.join(ART, "frozen_test", "summary.json"), test_summary)
    print(json.dumps({
        "val": val_summary,
        "constructions": {key: {"conditional_accuracy": value.get("conditional_accuracy"), "end_to_end_accuracy": value.get("end_to_end_accuracy"), "nll_on_hits": value.get("nll_on_hits"), "ece_honest": value.get("ece_honest")} for key, value in constructions.items()},
        "token_audit": audit,
        "permutation": perm,
        "train_relations": graph_taxonomy["relations"],
        "val_relations": val_taxonomy["relations"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
