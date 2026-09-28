"""Train, save, and reload the K=8 shortlist, then fit its own temperature.

Calibration rows are Banking77 test indices 1100:1300.
Release rows are 1300:1500.
Rows 200:296, which selected K=8, are not used for fitting or for the release score.
"""
import json
import math
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from research.veyabench.run_phase3_eval import MODEL, ROOT, SEED, choice_logits, load_ag_news, load_banking
from veya.calibration import fit_temperature
from veya.common import ece_score, safe_automation_coverage
from veya.shortlist import embed_fn_from_agent
from veya.shortlist_artifact import (
    file_sha256, load_route_calibration, load_shortlist, save_route_calibration, save_shortlist,
)

ART = os.path.join(ROOT, "artifacts", "experiments", "phase5")
PROJ = os.path.join(MODEL, "shortlist_k8.pt")
ROUTE_CAL = os.path.join(MODEL, "calibration_shortlist_k8.json")
CAL_SLICE = (1100, 1300)
TEST_SLICE = (1300, 1500)


def wilson(incorrect, accepted, z=1.96):
    if not accepted:
        return None
    p = incorrect / accepted
    den = 1.0 + z * z / accepted
    center = (p + z * z / (2 * accepted)) / den
    margin = z * math.sqrt(max(0.0, p * (1 - p) / accepted + z * z / (4 * accepted * accepted))) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def sac_report(correct, confidence):
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    out = {"n": int(len(correct)), "accuracy": float(correct.mean()) if len(correct) else None}
    for eps in (0.02, 0.05, 0.10):
        coverage = safe_automation_coverage(correct, confidence, eps)
        accepted = int(round(coverage * len(correct)))
        kept = correct[order[:accepted]] if accepted else np.array([])
        incorrect = int(accepted - kept.sum()) if accepted else 0
        out["sac@%.2f" % eps] = {
            "coverage": coverage,
            "accepted": accepted,
            "incorrect": incorrect,
            "observed_risk": (incorrect / accepted) if accepted else None,
            "wilson95": wilson(incorrect, accepted),
            "supports_1pct_claim": False,
        }
    return out


def softmax_row(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def train_projection(embed, labels, label_texts):
    torch.manual_seed(SEED)
    lv = torch.tensor(embed(label_texts), dtype=torch.float32)
    train = load_banking("train")[:800]
    tv = torch.tensor(embed([row[0] for row in train]), dtype=torch.float32)
    index = {label: i for i, label in enumerate(labels)}
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
    proj.eval()
    return proj, lv


def rank_texts(proj, embed, texts, lv, k):
    with torch.no_grad():
        queries = proj(torch.tensor(embed(texts), dtype=torch.float32))
        return (queries @ proj(lv).T).topk(k, dim=1).indices.tolist()


def rerank(agent, rows, labels, crit, ranking, temperature):
    correct, confidence, retrieved, rerank_ok = [], [], 0, 0
    nlls, briers = [], []
    started = time.perf_counter()
    for (text, truth), top in zip(rows, ranking):
        picked = [labels[i] for i in top]
        if truth not in picked:
            correct.append(0.0)
            confidence.append(0.0)
            continue
        retrieved += 1
        raw = choice_logits_one(agent, text, {label: crit[label] for label in picked})
        prob = softmax_row(raw, temperature)
        pred = picked[int(np.argmax(prob))]
        ok = pred == truth
        rerank_ok += int(ok)
        correct.append(1.0 if ok else 0.0)
        confidence.append(float(prob.max()))
        gold = picked.index(truth)
        nlls.append(-math.log(float(prob[gold]) + 1e-12))
        truth_vec = np.zeros_like(prob)
        truth_vec[gold] = 1.0
        briers.append(float(np.sum((prob - truth_vec) ** 2)))
    elapsed = time.perf_counter() - started
    correct = np.array(correct)
    return {
        "n": len(rows),
        "recall": retrieved / len(rows),
        "retrieval_failures": len(rows) - retrieved,
        "rerank_accuracy_given_hit": (rerank_ok / retrieved) if retrieved else None,
        "end_to_end_accuracy": float(correct.mean()),
        "nll_on_hits": float(np.mean(nlls)) if nlls else None,
        "brier_on_hits": float(np.mean(briers)) if briers else None,
        "ece_e2e": ece_score(np.array(confidence), correct),
        "seconds": round(elapsed, 3),
        "sac": sac_report(correct, confidence),
    }


def choice_logits_one(agent, text, criteria):
    return choice_logits(agent, [text], criteria)[0]


def collect_hit_logits(agent, rows, labels, crit, ranking):
    logits, gold = [], []
    for (text, truth), top in zip(rows, ranking):
        picked = [labels[i] for i in top]
        if truth not in picked:
            continue
        logits.append(choice_logits_one(agent, text, {label: crit[label] for label in picked}))
        gold.append(picked.index(truth))
    return np.stack(logits), np.array(gold)


def percentiles(samples):
    arr = np.array(samples, dtype=np.float64)
    return {name: float(np.percentile(arr, q)) for name, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))}


def main():
    import veya
    os.makedirs(ART, exist_ok=True)
    weights = os.path.join(MODEL, "model.safetensors")
    digest = file_sha256(weights)
    agent = veya.load(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    embed = embed_fn_from_agent(agent, max_length=96, batch_size=16)
    test = load_banking("test")
    labels = sorted({row[1] for row in test})
    crit = {label: "banking intent: %s" % label.replace("_", " ") for label in labels}
    texts = [crit[label] for label in labels]
    proj, lv = train_projection(embed, labels, texts)
    meta = save_shortlist(PROJ, proj, labels, texts, digest, k=8, seed=SEED)
    print("saved", PROJ, meta["file_sha256"][:12], flush=True)

    fresh_layer, _blob = load_shortlist(PROJ, digest, k=8)
    with torch.no_grad():
        probe = torch.tensor(embed([test[CAL_SLICE[0]][0]]), dtype=torch.float32)
        a = proj(probe) @ proj(lv).T
        b = fresh_layer(probe) @ fresh_layer(torch.tensor(embed(texts), dtype=torch.float32)).T
    if not torch.allclose(a, b, atol=1e-5):
        raise SystemExit("reloaded projection scores do not match")
    print("reload match", float((a - b).abs().max()), flush=True)

    cal_rows = test[CAL_SLICE[0]:CAL_SLICE[1]]
    te_rows = test[TEST_SLICE[0]:TEST_SLICE[1]]
    cal_rank = rank_texts(fresh_layer, embed, [row[0] for row in cal_rows], lv, 8)
    te_rank = rank_texts(fresh_layer, embed, [row[0] for row in te_rows], lv, 8)
    fit_x, fit_y = collect_hit_logits(agent, cal_rows, labels, crit, cal_rank)
    temperature = fit_temperature(fit_x, fit_y)
    print("fitted K8 temperature", temperature, "on", len(fit_y), "hits", flush=True)
    base_t = float(agent.temperature_by_options.get("choice:6-10", 1.0))
    before = rerank(agent, te_rows, labels, crit, te_rank, base_t)
    after = rerank(agent, te_rows, labels, crit, te_rank, temperature)
    print("before", before["end_to_end_accuracy"], before["ece_e2e"], "after", after["end_to_end_accuracy"], after["ece_e2e"], flush=True)
    improved = after["ece_e2e"] < before["ece_e2e"] and after["nll_on_hits"] < before["nll_on_hits"]
    if improved:
        save_route_calibration(
            ROUTE_CAL, temperature, len(fit_y), meta["file_sha256"], digest,
            "banking77_test_1100_1300",
            fit_split_hash="089e594c6bd1d708c9e866a1d8e7db08c13750ab72ad43b50332cd7f1b5f4e22",
        )
        loaded = load_route_calibration(ROUTE_CAL, digest, meta["file_sha256"])
        print("shipped route T", loaded["temperature"], flush=True)
    else:
        print("did not ship route calibration", flush=True)

    # packed comparison on the same release rows, capped so the run stays bounded
    packed_n = min(64, len(te_rows))
    packed_correct = []
    t0 = time.perf_counter()
    for text, truth in te_rows[:packed_n]:
        raw = choice_logits_one(agent, text, crit)
        pred = labels[int(np.argmax(softmax_row(raw, float(agent.temperature_by_options["choice:11+"]))))]
        packed_correct.append(pred == truth)
    packed_seconds = time.perf_counter() - t0

    names = ["World", "Sports", "Business", "Sci/Tech"]
    ag = load_ag_news()[400:520]
    ag_crit = {name: "news about %s" % name for name in names}
    from research.veyabench.run_phase3_eval import batched
    ag_logits = batched(agent, [row[0] for row in ag], ag_crit, batch_size=8)
    ag_y = np.array([row[1] for row in ag])
    ag_t = float(agent.temperature_by_options["choice:3-5"])
    from research.veyabench.run_phase3_eval import metrics as ag_metrics
    ag_report = ag_metrics(ag_logits, ag_y, ag_t)

    warm = te_rows[:20]
    samples = []
    for text, truth in warm:
        t1 = time.perf_counter()
        top = rank_texts(fresh_layer, embed, [text], lv, 8)[0]
        picked = [labels[i] for i in top]
        if truth in picked:
            choice_logits_one(agent, text, {label: crit[label] for label in picked})
        samples.append((time.perf_counter() - t1) * 1000)
    report = {
        "checkpoint_sha256": digest,
        "shortlist_sha256": meta["file_sha256"],
        "k": 8,
        "calibration_rows": [1100, 1300],
        "release_rows": [1300, 1500],
        "fit_hits": int(len(fit_y)),
        "temperature_choice_6_10": base_t,
        "temperature_k8": temperature,
        "shipped": improved,
        "k8_before": before,
        "k8_after": after,
        "packed_77_on_release_prefix": {"n": packed_n, "accuracy": float(np.mean(packed_correct)), "seconds": round(packed_seconds, 3)},
        "ag_news_rows_400_520": ag_report,
        "k8_latency_ms": {"n": len(samples), **percentiles(samples), "mean": float(np.mean(samples))},
    }
    path = os.path.join(ROOT, "artifacts", "experiments", "phase5", "release_eval.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"k8_after_acc": after["end_to_end_accuracy"], "packed": report["packed_77_on_release_prefix"], "ag": ag_report, "latency": report["k8_latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
