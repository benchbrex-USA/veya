"""Fit temperatures on real held-out rows and measure packed Choice versus a learned shortlist.

AG News public test and Banking77 are disjoint from each other. Calibration rows are
not reused as evaluation rows. Weights of the encoder are not updated. The shortlist
projection is a new head trained on Banking77 train.csv only.
"""
import csv
import json
import os
import random

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from veya.calibration import (
    brier,
    build_artifact,
    dump_calibration,
    nll,
)
from veya.common import QTYPES, build_sequence, collate_items, ece_score, temp_bucket
from veya.shortlist import embed_fn_from_agent

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "production")
ART = os.path.join(ROOT, "artifacts", "experiments", "phase3")
SEED = 17


def load_ag_news():
    import pyarrow.parquet as pq
    table = pq.read_table(os.path.join(ROOT, "research/data/ag_news/data/test-00000-of-00001.parquet"))
    names = ["World", "Sports", "Business", "Sci/Tech"]
    rows = []
    for text, label in zip(table.column("text").to_pylist(), table.column("label").to_pylist()):
        rows.append((text, int(label), names[int(label)]))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def load_banking(split):
    path = os.path.join(ROOT, "research/data/banking77", split + ".csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [(row["text"], row["category"]) for row in csv.DictReader(handle)]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def choice_logits(agent, texts, criteria):
    q = {"t": "choice", "ins": "Which label best fits the text?", "crit": criteria}
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    items = []
    for text in texts:
        seq, markers = build_sequence(agent.tok, text, q, max_len, head_max_len)
        if len(markers) != len(criteria):
            raise RuntimeError("packed head kept %d of %d options" % (len(markers), len(criteria)))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
    batch = collate_items([items], agent.tok.pad_token_id)
    device = agent.device
    with torch.inference_mode():
        logits, _act = agent.model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    return logits[:, : len(criteria)].float().cpu().numpy()


def batched(agent, texts, criteria, batch_size=8):
    parts = []
    for start in range(0, len(texts), batch_size):
        parts.append(choice_logits(agent, texts[start:start + batch_size], criteria))
        print("  logits %d/%d" % (min(start + batch_size, len(texts)), len(texts)), flush=True)
    return np.concatenate(parts, axis=0)


def metrics(logits, labels, temperature):
    from veya.calibration import _log_probs
    logp = _log_probs(logits, temperature)
    probs = np.exp(logp)
    labels = np.asarray(labels)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    return {
        "n": int(len(labels)),
        "temperature": float(temperature),
        "accuracy": float(correct.mean()),
        "nll": nll(logits, labels, temperature),
        "brier": brier(logits, labels, temperature),
        "ece": ece_score(probs.max(axis=1), correct),
    }


def subset_criteria(labels, truth, n, rng):
    others = [label for label in labels if label != truth]
    rng.shuffle(others)
    chosen = [truth] + others[: n - 1]
    rng.shuffle(chosen)
    return {label: "banking intent: %s" % label.replace("_", " ") for label in chosen}, chosen.index(truth)


def main():
    import veya
    os.makedirs(ART, exist_ok=True)
    print("loading", flush=True)
    agent = veya.load(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    print("device", agent.device, flush=True)
    shipped = float(agent.temperature_by_options.get("choice:3-5", agent.temperature[0]))
    shipped_wide = float(agent.temperature_by_options.get("choice:11+", agent.temperature[0]))

    ag = load_ag_news()
    names = ["World", "Sports", "Business", "Sci/Tech"]
    criteria = {name: "news about %s" % name for name in names}
    cal_rows, test_rows = ag[:400], ag[400:800]
    print("AG News calibration forward", flush=True)
    cal_logits = batched(agent, [row[0] for row in cal_rows], criteria)
    cal_y = np.array([row[1] for row in cal_rows])
    print("AG News test forward", flush=True)
    test_logits = batched(agent, [row[0] for row in test_rows], criteria)
    test_y = np.array([row[1] for row in test_rows])
    before = metrics(test_logits, test_y, shipped)
    artifact = build_artifact(
        {"choice:3-5": (cal_logits, cal_y)},
        checkpoint_id="models/production",
        dataset_hash="ag_news_test_seed17_0_400",
        min_observations=32,
    )
    fitted = artifact["cardinality_temperatures"]["choice:3-5"]
    after = metrics(test_logits, test_y, fitted)
    print("AG News shipped", before, "fitted", after, "T", fitted, flush=True)

    banking_test = load_banking("test")
    banking_labels = sorted({row[1] for row in banking_test})
    card_rows = []
    sample = banking_test[800:880]
    for n in (10, 20, 32):
        texts = []
        local_rng = random.Random(SEED + n)
        for text, truth in sample:
            crit, index = subset_criteria(banking_labels, truth, n, local_rng)
            # one shared criteria order cannot vary per row inside batched(); score per row
            texts.append((text, crit, index))
        logits = []
        gold = []
        for text, crit, index in texts:
            logits.append(choice_logits(agent, [text], crit)[0, :n])
            gold.append(index)
        # temperatures differ by bucket
        bucket = temp_bucket(QTYPES["choice"], n)
        temperature = float(agent.temperature_by_options.get(bucket, agent.temperature[0]))
        # pad logits to rectangle already n
        stacked = np.stack(logits, axis=0)
        card_rows.append({"options": n, "bucket": bucket, **metrics(stacked, np.array(gold), temperature)})
        print("cardinality", card_rows[-1], flush=True)

    wide_error = None
    try:
        wide_crit = {label: "banking intent: %s" % label.replace("_", " ") for label in banking_labels}
        choice_logits(agent, [sample[0][0]], wide_crit)
        wide_ok = True
    except RuntimeError as exc:
        wide_ok = False
        wide_error = str(exc)

    # shortlist head on frozen encoder embeddings
    embed = embed_fn_from_agent(agent, max_length=128, batch_size=16)
    label_text = ["banking intent: %s" % label.replace("_", " ") for label in banking_labels]
    label_vec = embed(label_text)
    train_rows = load_banking("train")[:800]
    eval_rows = banking_test[400:700]
    train_vec = embed([row[0] for row in train_rows])
    eval_vec = embed([row[0] for row in eval_rows])
    label_index = {label: i for i, label in enumerate(banking_labels)}
    proj = torch.nn.Linear(label_vec.shape[1], 128, bias=False)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_y = torch.tensor([label_index[row[1]] for row in train_rows])
    lv = torch.tensor(label_vec, dtype=torch.float32)
    tv = torch.tensor(train_vec, dtype=torch.float32)
    for epoch in range(8):
        order = torch.randperm(len(train_y))
        total = 0.0
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            scores = proj(tv[batch]) @ proj(lv).T
            loss = loss_fn(scores, train_y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
        print("shortlist epoch", epoch, "loss", round(total / len(train_y), 4), flush=True)
    with torch.inference_mode():
        learned = proj(torch.tensor(eval_vec, dtype=torch.float32)) @ proj(lv).T
        cosine = torch.tensor(eval_vec, dtype=torch.float32) @ lv.T
    gold = torch.tensor([label_index[row[1]] for row in eval_rows])

    def recalls(scores):
        out = {}
        for k in (1, 4, 8, 16, 32):
            top = scores.topk(k, dim=1).indices
            out["R@%d" % k] = float((top == gold[:, None]).any(dim=1).float().mean())
        return out

    shortlist_result = {"frozen_cosine": recalls(cosine), "learned_projection": recalls(learned)}
    print("shortlist", shortlist_result, flush=True)

    # end-to-end: shortlist top-8 then packed choice, on the eval rows, versus 20-way packed
    e2e_n = 40
    e2e_hits = {"shortlist8": 0, "packed20": 0}
    eval_slice = eval_rows[:e2e_n]
    with torch.inference_mode():
        qv = proj(torch.tensor(embed([row[0] for row in eval_slice]), dtype=torch.float32))
        rank = (qv @ proj(lv).T).topk(8, dim=1).indices.tolist()
    for row, top in zip(eval_slice, rank):
        text, truth = row
        labels = [banking_labels[i] for i in top]
        crit = {label: "banking intent: %s" % label.replace("_", " ") for label in labels}
        pred = choice_logits(agent, [text], crit)[0].argmax()
        if labels[int(pred)] == truth:
            e2e_hits["shortlist8"] += 1
        crit20, index20 = subset_criteria(banking_labels, truth, 20, random.Random(hash(text) % 100000))
        pred20 = choice_logits(agent, [text], crit20)[0, :20].argmax()
        if int(pred20) == index20:
            e2e_hits["packed20"] += 1
    e2e = {key: hits / e2e_n for key, hits in e2e_hits.items()}
    print("e2e", e2e, flush=True)

    ship = after["nll"] + 0.01 < before["nll"] and after["accuracy"] + 0.01 >= before["accuracy"]
    report = {
        "seed": SEED,
        "device": str(agent.device),
        "checkpoint": MODEL,
        "calibration_version_before": agent.calibration_version,
        "ag_news": {
            "shipped_temperature": shipped,
            "fitted_temperature": fitted,
            "calibration_split": "shuffled public test rows 0:400",
            "evaluation_split": "shuffled public test rows 400:800",
            "before": before,
            "after": after,
            "ship": ship,
        },
        "banking77_cardinality": card_rows,
        "banking77_77way_packs": wide_ok,
        "banking77_77way_error": wide_error,
        "shipped_choice_11plus": shipped_wide,
        "shortlist": shortlist_result,
        "choice_e2e_n40": e2e,
    }
    path = os.path.join(ART, "phase3_results.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if ship:
        dump_calibration(artifact, os.path.join(MODEL, "calibration.json"))
        print("shipped calibration", fitted, flush=True)
    else:
        print("did not ship calibration", flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
