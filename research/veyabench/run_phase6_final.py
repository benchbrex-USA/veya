"""One release-test pass, validation cardinality, AG News, and HTTP.

Reads the training decisions. A reranker is loaded only when its own
validation rule promoted it. Cardinality uses the validation split.
"""
import json
import os
import random

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.phase6_protocol import CALIBRATION, RERANKER_VAL
from research.veyabench.phase6_score import score_sets, softmax_row
from research.veyabench.run_phase3_eval import ROOT, SEED, choice_logits, load_ag_news, load_banking, metrics, subset_criteria
from research.veyabench.run_phase6_baseline import summarize
from research.veyabench.run_release_eval import http_suite, load_from_manifest
from veya.calibration import fit_temperature
from veya.release_bundle import INSTRUCTIONS, raw_choice_logits
from veya.reranker import load_reranker

ART = os.path.join(ROOT, "artifacts", "experiments", "phase6")
CARD_N = 300
CARD_GRID = (20, 24, 28, 32)


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def convert(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(type(value).__name__)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert)
        handle.write("\n")


def load_decision(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def cardinality(bundle, rows, labels, crit, reranker):
    table = []
    for n in CARD_GRID:
        print("cardinality", n, flush=True)
        packed_ok, hier_ok = [], []
        for i, (text, truth) in enumerate(rows):
            rng = random.Random(SEED + n * 10007 + i)
            if n >= len(labels):
                subset = crit
            else:
                subset, _gold = subset_criteria(labels, truth, n, rng)
            logits, _tokens = raw_choice_logits(bundle.agent, text, INSTRUCTIONS, subset)
            temperature = bundle.temperature_for("packed", len(subset))
            pred = list(subset.keys())[int(np.argmax(softmax_row(logits, temperature)))]
            packed_ok.append(pred == truth)
            names = list(subset.keys())
            if len(names) > 8:
                picked, _stage = bundle.shortlist(text, list(subset.items()), 8)
            else:
                picked = names
            if reranker is None:
                h_logits, _h = raw_choice_logits(bundle.agent, text, INSTRUCTIONS, {label: subset[label] for label in picked})
            else:
                h_logits, _h = reranker.choice_logits(bundle.agent, text, INSTRUCTIONS, {label: subset[label] for label in picked})
            h_pred = picked[int(np.argmax(softmax_row(h_logits, bundle.route_temperature if reranker is None else 1.0)))]
            hier_ok.append(h_pred == truth)
        table.append({
            "n_candidates": n,
            "examples": len(rows),
            "packed_accuracy": float(np.mean(packed_ok)),
            "hierarchical_accuracy": float(np.mean(hier_ok)),
        })
    gaps = {row["n_candidates"]: row["hierarchical_accuracy"] - row["packed_accuracy"] for row in table}
    cutoff = 20
    reason = "32-candidate gap is below 0.05; keep 20"
    if gaps.get(32, 0.0) >= 0.05:
        winning = [n for n in CARD_GRID if gaps[n] >= 0.05 and all(gaps[m] >= 0.05 for m in CARD_GRID if m >= n)]
        if winning:
            first = min(winning)
            below = [n for n in CARD_GRID if n < first]
            cutoff = below[-1] if below else 16
            reason = "hierarchical leads by >= 0.05 from %d candidates" % first
        else:
            reason = "no grid point stays above 0.05 through 32"
    return {"rows": [RERANKER_VAL[0], RERANKER_VAL[0] + len(rows)], "table": table, "recommended_packed_max": cutoff, "reason": reason}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.load(open(os.path.join(ROOT, "artifacts", "release", "manifest.json"), encoding="utf-8"))
    print("loading", device, flush=True)
    bundle = load_from_manifest(manifest, device, {})
    decisions = {
        "listwise": load_decision("train_decision.json"),
        "pairwise": load_decision("train_decision_pairwise.json"),
        "interaction": load_decision("interaction_decision.json"),
    }
    promoted = [name for name, decision in decisions.items() if decision and decision.get("promoted")]
    reranker = None
    if len(promoted) == 1 and promoted[0] != "interaction":
        path = os.path.join(ROOT, decisions[promoted[0]]["artifact"])
        reranker = load_reranker(path, bundle.agent, bundle.checkpoint_hash, bundle.shortlist_sha256)
        reranker.to(bundle.agent.device)
        bundle.reranker = reranker
        bundle.reranker_id = promoted[0]
        bundle.reranker_sha256 = reranker.meta.get("file_sha256")
    elif len(promoted) > 1:
        raise SystemExit("more than one reranker was promoted: %s" % promoted)
    print("promoted", promoted or ["base_decision_head"], flush=True)

    train = load_banking("train")
    labels = list(bundle.labels)
    crit = {label: text for label, text in zip(labels, bundle.label_texts)}
    val_rows = train[RERANKER_VAL[0]:RERANKER_VAL[0] + CARD_N]
    report = {"promoted": promoted, "decisions": {name: decision for name, decision in decisions.items() if decision}}
    report["cardinality_validation"] = cardinality(bundle, val_rows, labels, crit, reranker)

    print("ag news", flush=True)
    ag = load_ag_news()[400:800]
    names = ["World", "Sports", "Business", "Sci/Tech"]
    ag_crit = {name: "news about %s" % name for name in names}
    parts = []
    for start in range(0, len(ag), 8):
        parts.append(choice_logits(bundle.agent, [row[0] for row in ag[start:start + 8]], ag_crit))
    ag_t = float(bundle.agent.temperature_by_options["choice:3-5"])
    report["ag_news_400_800"] = metrics(np.concatenate(parts), np.array([row[1] for row in ag]), ag_t)

    if reranker is not None:
        print("calibrate on train", CALIBRATION, flush=True)
        cal_rows = train[CALIBRATION[0]:CALIBRATION[1]]
        from research.veyabench.run_phase6_train import rank_all
        ranked = rank_all(bundle, [row[0] for row in cal_rows])
        picks = []
        keep_rows = []
        for (text, truth), top in zip(cal_rows, ranked):
            picked = [labels[i] for i in top]
            if truth in picked:
                picks.append(picked)
                keep_rows.append((text, truth))
        scored = score_sets(bundle.agent, [row[0] for row in keep_rows], picks, crit, temperature=1.0, reranker=reranker, batch_size=4)
        logits = np.stack([row["logits"] for row in scored])
        gold = np.array([row["candidates"].index(truth) for row, (_text, truth) in zip(scored, keep_rows)])
        temperature = fit_temperature(logits, gold)
        before = summarize_from_scores(scored, keep_rows, 1.0)
        after = summarize_from_scores(scored, keep_rows, temperature)
        report["calibration_val_not_used_for_selection"] = {
            "fit_split": "train 8600:9300 hits",
            "fit_rows": len(keep_rows),
            "temperature": temperature,
            "before_T1": before,
            "after": after,
        }
        print("release test once", flush=True)
        test = load_banking("test")[1300:]
        test_rank = rank_all(bundle, [row[0] for row in test])
        test_picks = [[labels[i] for i in top] for top in test_rank]
        test_scored = score_sets(bundle.agent, [row[0] for row in test], test_picks, crit, temperature=temperature, reranker=reranker, batch_size=4)
        from research.veyabench.run_phase6_baseline import attach
        report["release_test"] = summarize(attach(test, test_picks, test_scored, "fitted"))
    else:
        report["release_test"] = json.load(open(os.path.join(ART, "frozen_test", "summary.json"), encoding="utf-8"))
        report["release_test_source"] = "phase6 frozen baseline; no promoted reranker, so the test was not rescored"

    print("http", flush=True)
    report["http_failed"] = http_suite(bundle, labels, crit)["failed"]
    dump(os.path.join(ART, "final_report.json"), report)
    print(json.dumps({"promoted": promoted, "cardinality": report["cardinality_validation"], "ag": report["ag_news_400_800"], "http_failed": report["http_failed"]}, indent=2), flush=True)


def summarize_from_scores(scored, rows, temperature):
    correct, conf, nlls = [], [], []
    for record, (_text, truth) in zip(scored, rows):
        probs = softmax_row(record["logits"], temperature)
        pred = record["candidates"][int(np.argmax(probs))]
        gold = record["candidates"].index(truth)
        correct.append(pred == truth)
        conf.append(float(probs.max()))
        nlls.append(float(-np.log(probs[gold] + 1e-12)))
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "nll": float(np.mean(nlls)),
        "mean_confidence": float(np.mean(conf)),
    }


if __name__ == "__main__":
    main()
