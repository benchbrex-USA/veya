"""Train the small top-8 interaction scorer on the same hard sets.

The packed head is not updated. Candidate order does not change the chosen
label: the block has no positional encoding, so attention is equivariant.
The release test is not read.
"""
import json
import os
import random

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

from research.veyabench.phase6_protocol import (
    RERANKER_TRAIN, RERANKER_VAL, assert_training_id, example_id, paired_bootstrap, promote,
)
from research.veyabench.run_phase3_eval import ROOT, SEED, load_banking
from research.veyabench.run_release_eval import load_from_manifest
from veya.experimental.top8 import Top8Interaction

ART = os.path.join(ROOT, "artifacts", "experiments", "phase6")


def rank_and_cache(bundle, texts):
    """Return 1024-d encoder means, plus shortlist indices from the saved projection."""
    vectors = np.asarray(bundle.embed(list(texts)), dtype=np.float32)
    label_vectors = np.asarray(bundle.embed(list(bundle.label_texts)), dtype=np.float32)
    projected_q = bundle.project(vectors)
    projected_l = bundle.project(label_vectors)
    index = torch.tensor(np.asarray(projected_q @ projected_l.T)).topk(8, dim=1).indices.tolist()
    return vectors, label_vectors, index


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.load(open(os.path.join(ROOT, "artifacts", "release", "manifest.json"), encoding="utf-8"))
    print("loading", device, flush=True)
    bundle = load_from_manifest(manifest, device, {})
    labels = list(bundle.labels)
    train = load_banking("train")
    rows = train[RERANKER_TRAIN[0]:RERANKER_TRAIN[1]]
    print("embedding train", len(rows), flush=True)
    query_vectors, label_vectors, ranked = rank_and_cache(bundle, [row[0] for row in rows])
    pool = []
    absent = 0
    for offset, ((text, truth), top) in enumerate(zip(rows, ranked)):
        index = RERANKER_TRAIN[0] + offset
        assert_training_id(example_id("train", index))
        picked = [labels[i] for i in top]
        if truth not in picked:
            absent += 1
            continue
        pool.append((index, offset, truth, picked))
    print("trainable", len(pool), "absent", absent, flush=True)

    val_records = json.load(open(os.path.join(ART, "val_predictions.json"), encoding="utf-8"))
    val_rows = train[RERANKER_VAL[0]:RERANKER_VAL[1]]
    print("embedding val", len(val_rows), flush=True)
    val_vectors, _, _ = rank_and_cache(bundle, [row[0] for row in val_rows])
    val_picks = [record["retrieved_top8"] for record in val_records]
    base_hits = [1.0 if record["correct"] else 0.0 for record in val_records if record["target_in_top8"]]
    from research.veyabench.phase6_score import softmax_row
    base_high = sum(
        1 for record in val_records
        if (not record["correct"]) and float(softmax_row(record["reranker_logits"], 1.0).max()) >= 0.70
    ) / len(val_records)
    label_index = {label: i for i, label in enumerate(labels)}

    model = Top8Interaction(dim=query_vectors.shape[1], width=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    label_tensor = torch.tensor(label_vectors, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device=device)
    val_tensor = torch.tensor(val_vectors, dtype=torch.float32, device=device)

    def evaluate():
        model.eval()
        hit_new = []
        high = 0
        with torch.no_grad():
            for row_i, ((text, truth), picked) in enumerate(zip(val_rows, val_picks)):
                columns = [label_index[label] for label in picked]
                candidates = label_tensor[columns].unsqueeze(0)
                logits = model(val_tensor[row_i].unsqueeze(0), candidates)[0]
                probs = torch.softmax(logits, dim=0)
                pred = picked[int(torch.argmax(probs).item())]
                confidence = float(probs.max().item())
                correct = pred == truth
                if truth in picked:
                    hit_new.append(1.0 if correct else 0.0)
                if (not correct) and confidence >= 0.70:
                    high += 1
        return hit_new, high / len(val_rows)

    best = {"accuracy": -1.0, "state": None, "epoch": None}
    history = []
    for epoch in range(8):
        model.train()
        order = list(range(len(pool)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            queries = []
            candidates = []
            golds = []
            for pool_index in batch:
                index, offset, truth, shortlist = pool[pool_index]
                rng = random.Random(SEED + 10007 * epoch + index)
                if index % 5 == 0:
                    others = [label for label in labels if label != truth]
                    rng.shuffle(others)
                    picked = [truth] + others[:7]
                    rng.shuffle(picked)
                else:
                    picked = list(shortlist)
                    rng.shuffle(picked)
                queries.append(query_tensor[offset])
                candidates.append(label_tensor[[label_index[label] for label in picked]])
                golds.append(picked.index(truth))
            logits = model(torch.stack(queries), torch.stack(candidates))
            loss = F.cross_entropy(logits, torch.tensor(golds, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        hit_new, high_rate = evaluate()
        boot = paired_bootstrap(base_hits, hit_new)
        accuracy = float(np.mean(hit_new))
        print("epoch", epoch, "loss", float(np.mean(losses)), "conditional", accuracy, "delta", boot["delta"], "ci", boot["ci95_low"], boot["ci95_high"], flush=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "conditional_accuracy": accuracy, "bootstrap": boot, "high_conf_rate": high_rate})
        if accuracy > best["accuracy"]:
            best = {
                "accuracy": accuracy,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "bootstrap": boot,
                "high_conf_rate": high_rate,
            }
        elif epoch >= 2:
            break
    promoted, reason = promote(best["bootstrap"]["ci95_low"], best["high_conf_rate"], base_high)
    path = os.path.join(ART, "candidates", "top8_interaction.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"format": "veya-top8-interaction-v1", "state_dict": best["state"], "width": 256}, path)
    decision = {
        "promoted": promoted,
        "reason": reason,
        "best": {key: value for key, value in best.items() if key != "state"},
        "baseline_high_conf_rate": base_high,
        "history": history,
        "artifact": "artifacts/experiments/phase6/candidates/top8_interaction.pt",
    }
    with open(os.path.join(ART, "interaction_decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"promoted": promoted, "reason": reason, "best_epoch": best["epoch"], "accuracy": best["accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
