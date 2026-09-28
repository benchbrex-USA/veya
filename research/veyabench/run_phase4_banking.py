"""Identical-set Banking77 comparison: packed 77-way Choice vs learned shortlist rerank.

Eval rows are Banking77 test.csv after the seed-17 shuffle, indices 200:296.
They are disjoint from the temperature fit (0:160) and from shortlist training (train.csv).
The true label is never inserted into a shortlist that missed it.
"""
import json
import math
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.run_phase3_eval import (
    MODEL, ROOT, SEED, choice_logits, load_banking,
)
from veya.common import ece_score, safe_automation_coverage

ART = os.path.join(ROOT, "artifacts", "experiments", "phase4")
EVAL_START, EVAL_END = 200, 296


def wilson(incorrect, accepted, z=1.96):
    if accepted <= 0:
        return None
    p = incorrect / accepted
    den = 1.0 + z * z / accepted
    center = (p + z * z / (2 * accepted)) / den
    margin = z * math.sqrt(p * (1 - p) / accepted + z * z / (4 * accepted * accepted)) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(correct, confidence):
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    curve = []
    wrong = 0.0
    for i, idx in enumerate(order):
        wrong += 1.0 - float(correct[idx])
        curve.append({"coverage": (i + 1) / len(correct), "risk": wrong / (i + 1)})
    sac = {}
    for eps in (0.005, 0.01, 0.02, 0.05):
        coverage = safe_automation_coverage(correct, confidence, eps)
        accepted = int(round(coverage * len(correct)))
        kept = correct[order[:accepted]] if accepted else np.array([])
        incorrect = int(accepted - kept.sum()) if accepted else 0
        sac["sac@%.3f" % eps] = {
            "coverage": coverage,
            "accepted": accepted,
            "incorrect": incorrect,
            "observed_risk": (incorrect / accepted) if accepted else None,
            "wilson95_risk": wilson(incorrect, accepted),
            "reliable_for_1pct": len(correct) >= 500,
        }
    return {"n": int(len(correct)), "accuracy": float(correct.mean()), "curve_tail": curve[:: max(1, len(curve)//8)], "sac": sac}


def probs_from_logits(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def main():
    import veya
    from veya.shortlist import embed_fn_from_agent
    os.makedirs(ART, exist_ok=True)
    agent = veya.load(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    print("device", agent.device, "calibration", agent.calibration_version, flush=True)
    test = load_banking("test")
    labels = sorted({row[1] for row in test})
    index = {label: i for i, label in enumerate(labels)}
    eval_rows = test[EVAL_START:EVAL_END]
    crit77 = {label: "banking intent: %s" % label.replace("_", " ") for label in labels}
    t11 = float(agent.temperature_by_options.get("choice:11+", agent.temperature[0]))

    print("packed 77-way", len(eval_rows), flush=True)
    started = time.perf_counter()
    packed_logits = []
    packed_fail = 0
    for text, _truth in eval_rows:
        try:
            packed_logits.append(choice_logits(agent, [text], crit77)[0])
        except RuntimeError:
            packed_fail += 1
            packed_logits.append(None)
    packed_seconds = time.perf_counter() - started

    embed = embed_fn_from_agent(agent, max_length=96, batch_size=16)
    label_text = [crit77[label] for label in labels]
    lv = torch.tensor(embed(label_text), dtype=torch.float32)
    train = load_banking("train")[:800]
    tv = torch.tensor(embed([row[0] for row in train]), dtype=torch.float32)
    y = torch.tensor([index[row[1]] for row in train])
    proj = torch.nn.Linear(lv.shape[1], 128, bias=False)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _epoch in range(6):
        order = torch.randperm(len(y))
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            scores = proj(tv[batch]) @ proj(lv).T
            loss = loss_fn(scores, y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()

    eval_text = [row[0] for row in eval_rows]
    with torch.no_grad():
        ranking = (proj(torch.tensor(embed(eval_text))) @ proj(lv).T).topk(32, dim=1).indices.tolist()

    # candidate-set head on frozen embeddings, true label among all 77
    head = torch.nn.Sequential(torch.nn.Linear(lv.shape[1], 128), torch.nn.Tanh(), torch.nn.Linear(128, 128))
    opt_h = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _epoch in range(6):
        order = torch.randperm(len(y))
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            q = head(tv[batch])
            c = head(lv)
            loss = loss_fn(q @ c.T, y[batch])
            opt_h.zero_grad()
            loss.backward()
            opt_h.step()
    with torch.no_grad():
        cand_scores = head(torch.tensor(embed(eval_text))) @ head(lv).T
        cand_pred = cand_scores.argmax(dim=1).tolist()
    cand_correct = [labels[pred] == truth for pred, (_text, truth) in zip(cand_pred, eval_rows)]

    paths = {}
    per_row = []
    for k in (8, 16, 32):
        correct = []
        conf = []
        retrieved = 0
        rerank_ok = 0
        t0 = time.perf_counter()
        for (text, truth), top in zip(eval_rows, ranking):
            picked = [labels[i] for i in top[:k]]
            hit = truth in picked
            retrieved += int(hit)
            if not hit:
                correct.append(0.0)
                conf.append(0.0)
                continue
            crit = {label: crit77[label] for label in picked}
            raw = choice_logits(agent, [text], crit)[0]
            prob = probs_of(raw, t11)
            pred = picked[int(np.argmax(prob))]
            ok = pred == truth
            rerank_ok += int(ok)
            correct.append(1.0 if ok else 0.0)
            conf.append(float(np.max(prob)))
        elapsed = time.perf_counter() - t0
        paths["shortlist_%d" % k] = {
            "recall": retrieved / len(eval_rows),
            "retrieval_failures": len(eval_rows) - retrieved,
            "rerank_accuracy_given_hit": (rerank_ok / retrieved) if retrieved else None,
            "end_to_end_accuracy": float(np.mean(correct)),
            "seconds": round(elapsed, 2),
            "sac": summarize(correct, conf),
        }
        print("K", k, paths["shortlist_%d" % k]["recall"], paths["shortlist_%d" % k]["end_to_end_accuracy"], flush=True)
        per_row.append(correct)

    packed_correct = []
    packed_conf = []
    for logit, (_text, truth) in zip(packed_logits, eval_rows):
        if logit is None:
            packed_correct.append(0.0)
            packed_conf.append(0.0)
            continue
        prob = probs_of(logit, t11)
        pred = labels[int(np.argmax(prob))]
        packed_correct.append(1.0 if pred == truth else 0.0)
        packed_conf.append(float(np.max(prob)))
    paths["packed_77"] = {
        "pack_failures": packed_fail,
        "end_to_end_accuracy": float(np.mean(packed_correct)),
        "seconds": round(packed_seconds, 2),
        "temperature": t11,
        "nll": nll_if_complete(packed_logits, eval_rows, labels, t11),
        "ece": ece_score(np.array(packed_conf), np.array(packed_correct)),
        "sac": summarize(packed_correct, packed_conf),
    }
    paths["candidate_set_frozen_encoder"] = {
        "accuracy": float(np.mean(cand_correct)),
        "n": len(cand_correct),
        "note": "Scores frozen encoder embeddings. Not the production decision head.",
    }
    report = {
        "seed": SEED,
        "eval_rows": [EVAL_START, EVAL_END],
        "n": len(eval_rows),
        "device": str(agent.device),
        "calibration_version": agent.calibration_version,
        "temperature_choice_11plus": t11,
        "paths": paths,
    }
    path = os.path.join(ART, "banking77_routes.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sac"} for k, v in paths.items()}, indent=2))
    print("wrote", path)


def probs_of(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - np.max(z)
    p = np.exp(z)
    return p / p.sum()


def nll_if_complete(logits, rows, labels, temperature):
    total = 0.0
    count = 0
    for logit, (_text, truth) in zip(logits, rows):
        if logit is None:
            continue
        prob = probs_of(logit, temperature)
        total += -math.log(float(prob[labels.index(truth)]) + 1e-12)
        count += 1
    return total / count if count else None


if __name__ == "__main__":
    main()
