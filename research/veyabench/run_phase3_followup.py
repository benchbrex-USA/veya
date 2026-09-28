"""Ship the AG News temperature only if a fresh runtime load sees it, and measure choice:11+ plus an honest shortlist.

The shortlist end-to-end score does not insert the true label when retrieval missed it.
"""
import json
import os
import random
import threading
from urllib.request import Request, urlopen

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from veya.calibration import build_artifact, dump_calibration
from veya.server import serve
from research.veyabench.run_phase3_eval import (
    MODEL, ROOT, SEED, batched, choice_logits, load_ag_news, load_banking, metrics, subset_criteria,
)

ART = os.path.join(ROOT, "artifacts", "experiments", "phase3")


def main():
    import veya
    ag = load_ag_news()
    names = ["World", "Sports", "Business", "Sci/Tech"]
    criteria = {name: "news about %s" % name for name in names}
    cal_rows, test_rows = ag[:400], ag[400:800]
    agent = veya.load(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    print("device", agent.device, flush=True)
    print("refit choice 3-5", flush=True)
    cal_logits = batched(agent, [row[0] for row in cal_rows], criteria, batch_size=8)
    cal_y = np.array([row[1] for row in cal_rows])
    artifact = build_artifact(
        {"choice:3-5": (cal_logits, cal_y)},
        checkpoint_id="models/production english",
        dataset_hash="ag_news_public_test_seed17_rows_0_400",
        min_observations=32,
    )
    fitted = artifact["cardinality_temperatures"]["choice:3-5"]
    test_logits = batched(agent, [row[0] for row in test_rows], criteria, batch_size=8)
    test_y = np.array([row[1] for row in test_rows])
    shipped = float(agent.temperature_by_options["choice:3-5"])
    before = metrics(test_logits, test_y, shipped)
    after = metrics(test_logits, test_y, fitted)
    improved = after["ece"] < before["ece"] and after["nll"] < before["nll"] and after["accuracy"] + 1e-9 >= before["accuracy"] - 0.005
    print("ag before", before, "after", after, "improved", improved, flush=True)
    if improved:
        dump_calibration(artifact, os.path.join(MODEL, "calibration.json"))
    fresh = veya.load(MODEL, device="cpu")
    loaded_t = float(fresh.temperature_by_options.get("choice:3-5"))
    print("reloaded choice:3-5", loaded_t, "version", fresh.calibration_version, flush=True)

    banking = load_banking("test")
    labels = sorted({row[1] for row in banking})
    def collect(rows):
        logits, gold = [], []
        for text, truth in rows:
            crit, index = subset_criteria(labels, truth, 20, random.Random(SEED))
            logits.append(choice_logits(fresh, [text], crit)[0])
            gold.append(index)
        return np.stack(logits), np.array(gold)
    print("banking 20-way calibration", flush=True)
    b_cal_x, b_cal_y = collect(banking[:80])
    print("banking 20-way test", flush=True)
    b_te_x, b_te_y = collect(banking[80:160])
    wide_artifact = build_artifact(
        {"choice:11+": (b_cal_x, b_cal_y)},
        checkpoint_id="models/production english",
        dataset_hash="banking77_test_seed17_20way_rows_0_80",
        min_observations=32,
    )
    wide_t = wide_artifact["cardinality_temperatures"]["choice:11+"]
    wide_before = metrics(b_te_x, b_te_y, 0.5)
    wide_after = metrics(b_te_x, b_te_y, wide_t)
    print("choice11+", wide_before, wide_after, "T", wide_t, flush=True)
    # Do not ship choice:11+ from an 80-row fit. Record it only.

    # honest shortlist, no gold injection
    from veya.shortlist import embed_fn_from_agent
    embed = embed_fn_from_agent(fresh, max_length=96, batch_size=16)
    label_text = ["banking intent: %s" % label.replace("_", " ") for label in labels]
    lv = torch.tensor(embed(label_text), dtype=torch.float32)
    train = load_banking("train")[:600]
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
    eval_rows = banking[400:440]
    with torch.no_grad():
        top = (proj(torch.tensor(embed([r[0] for r in eval_rows]))) @ proj(lv).T).topk(8, dim=1).indices.tolist()
    hits = 0
    missed = 0
    for row, ids in zip(eval_rows, top):
        picked = [labels[i] for i in ids]
        if row[1] not in picked:
            missed += 1
            continue
        crit = {label: "banking intent: %s" % label.replace("_", " ") for label in picked}
        pred = int(choice_logits(fresh, [row[0]], crit)[0].argmax())
        hits += picked[pred] == row[1]
    honest = {"n": len(eval_rows), "retrieval_misses": missed, "correct": hits, "accuracy": hits / len(eval_rows)}
    print("honest shortlist8", honest, flush=True)

    server = serve(fresh, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    request = Request(
        "http://127.0.0.1:%d/v1/choice" % port,
        data=json.dumps({
            "state": {"message": "Please refund invoice 4411."},
            "instructions": "Which team handles `message`?",
            "criteria": {"billing": "invoices and refunds", "technical": "bugs"},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    http_choice = json.loads(urlopen(request, timeout=60).read().decode())
    http_health = json.loads(urlopen("http://127.0.0.1:%d/health" % port, timeout=10).read().decode())
    server.shutdown()
    print("http", http_health, http_choice.get("answer"), http_choice.get("decision"), flush=True)

    out = {
        "ag_news_reloaded_temperature": loaded_t,
        "ag_news_before": before,
        "ag_news_after": after,
        "shipped_choice_3_5": improved,
        "choice_11plus_n80": {"before_clamped_0.5": wide_before, "fitted": wide_after, "temperature": wide_t, "shipped": False,
                              "reason": "80 calibration rows is enough to fit but too small to replace the production bucket"},
        "honest_shortlist8": honest,
        "http_health": http_health,
        "http_choice_answer": http_choice.get("answer"),
        "http_choice_decision": http_choice.get("decision"),
    }
    path = os.path.join(ART, "phase3_followup.json")
    os.makedirs(ART, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
