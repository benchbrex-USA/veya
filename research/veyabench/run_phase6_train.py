"""Fine-tune a route-specific K=8 head on hard shortlist sets.

The encoder and the base decision head stay frozen. Gradients use train
indices 800:8600 only. Validation is train indices 9300:10016. The release
test is not read.
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
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, RERANKER_TRAIN, RERANKER_VAL,
    assert_training_id, example_id, paired_bootstrap, promote,
)
from research.veyabench.phase6_score import score_sets, softmax_row
from research.veyabench.run_phase3_eval import ROOT, SEED, load_banking
from research.veyabench.run_release_eval import load_from_manifest
from veya.common import QTYPES, build_sequence, collate_items
from veya.release_bundle import INSTRUCTIONS
from veya.reranker import K8Reranker, save_reranker

ART = os.path.join(ROOT, "artifacts", "experiments", "phase6")


def rank_all(bundle, texts):
    queries = bundle.project(bundle.embed(list(texts)))
    labels = bundle.project(bundle.embed(list(bundle.label_texts)))
    scores = queries @ labels.T
    return torch.tensor(np.asarray(scores)).topk(8, dim=1).indices.tolist()


def random_set(labels, truth, rng):
    others = [label for label in labels if label != truth]
    rng.shuffle(others)
    picked = [truth] + others[:7]
    rng.shuffle(picked)
    return picked


def loss_on_batch(agent, reranker, texts, picks, golds, crit, pairwise_weight=0.0):
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    items = []
    for text, picked in zip(texts, picks):
        criteria = {label: crit[label] for label in picked}
        question = {"t": "choice", "ins": INSTRUCTIONS, "crit": criteria}
        seq, markers = build_sequence(agent.tok, text, question, max_len, head_max_len)
        if len(markers) != 8:
            raise RuntimeError("training row did not keep 8 candidates")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
    batch = collate_items([items], agent.tok.pad_token_id)
    device = next(reranker.parameters()).device
    with torch.no_grad():
        hidden = agent.model.encoder(
            input_ids=batch["input_ids"].to(agent.device),
            attention_mask=batch["attention_mask"].to(agent.device),
        ).last_hidden_state.detach()
    logits = reranker.score_hidden(
        hidden.to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
    )
    golds = golds.to(device)
    logits = logits[:, :8]
    loss = F.cross_entropy(logits, golds)
    if pairwise_weight:
        positive = logits.gather(1, golds.view(-1, 1))
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, golds.view(-1, 1), True)
        # Mean log-sigmoid over the seven negatives. The positive column is excluded.
        gaps = F.logsigmoid(positive - logits)
        loss = loss + float(pairwise_weight) * (-gaps.masked_fill(mask, 0.0).sum() / (logits.size(0) * 7))
    return loss


def conditional_and_highconf(rows, picks, scored):
    hit_new = []
    high = 0
    n = len(rows)
    for (text, truth), picked, record in zip(rows, picks, scored):
        if truth not in picked:
            if record["confidence"] >= 0.70:
                high += 1
            continue
        hit_new.append(1.0 if record["prediction"] == truth else 0.0)
        if record["prediction"] != truth and record["confidence"] >= 0.70:
            high += 1
    return hit_new, high / n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.load(open(os.path.join(ROOT, "artifacts", "release", "manifest.json"), encoding="utf-8"))
    print("loading", device, flush=True)
    bundle = load_from_manifest(manifest, device, {})
    agent = bundle.agent
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    labels = list(bundle.labels)
    crit = {label: text for label, text in zip(labels, bundle.label_texts)}
    train = load_banking("train")
    rows = train[RERANKER_TRAIN[0]:RERANKER_TRAIN[1]]
    print("ranking reranker train", len(rows), flush=True)
    ranked = rank_all(bundle, [row[0] for row in rows])
    pool = []
    absent = 0
    for offset, ((text, truth), top) in enumerate(zip(rows, ranked)):
        index = RERANKER_TRAIN[0] + offset
        assert_training_id(example_id("train", index))
        picked = [labels[i] for i in top]
        if truth not in picked:
            absent += 1
            continue
        pool.append((index, text, truth, picked))
    print("trainable rows", len(pool), "target absent", absent, flush=True)

    reranker = K8Reranker()
    reranker.clone_from_agent(agent)
    reranker.to(agent.device)
    reranker.eval()
    probe_picked = pool[0][3]
    probe_criteria = {label: crit[label] for label in probe_picked}
    from veya.release_bundle import raw_choice_logits
    reference = raw_choice_logits(agent, pool[0][1], INSTRUCTIONS, probe_criteria)[0]
    copied = reranker.choice_logits(agent, pool[0][1], INSTRUCTIONS, probe_criteria)[0]
    if not np.allclose(reference, copied, atol=1e-4):
        raise SystemExit("cloned head does not match the base head: max abs %s" % float(np.max(np.abs(reference - copied))))
    print("clone matches base head", flush=True)

    val_records = json.load(open(os.path.join(ART, "val_predictions.json"), encoding="utf-8"))
    val_rows = [(train[RERANKER_VAL[0] + i][0], record["target"]) for i, record in enumerate(val_records)]
    val_picks = [record["retrieved_top8"] for record in val_records]
    base_hits = [1.0 if record["correct"] else 0.0 for record in val_records if record["target_in_top8"]]
    # Argmax does not depend on a positive temperature. Confidence does. Compare
    # high-confidence mistakes at T=1, before any new calibration is fit.
    base_high = 0.0
    if val_records:
        base_high = sum(
            1 for record in val_records
            if (not record["correct"]) and float(softmax_row(record["reranker_logits"], 1.0).max()) >= 0.70
        ) / len(val_records)

    optimizer = torch.optim.AdamW(reranker.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    best = {"accuracy": -1.0, "state": None, "epoch": None}
    history = []
    for epoch in range(MAX_EPOCHS):
        reranker.train()
        order = list(range(len(pool)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        step = 0
        batch_texts, batch_picks, batch_golds = [], [], []
        for position, pool_index in enumerate(order):
            index, text, truth, shortlist = pool[pool_index]
            rng = random.Random(SEED + 10007 * epoch + index)
            if index % 5 == 0:
                picked = random_set(labels, truth, rng)
            else:
                picked = list(shortlist)
                rng.shuffle(picked)
            batch_texts.append(text)
            batch_picks.append(picked)
            batch_golds.append(picked.index(truth))
            if len(batch_texts) < BATCH_SIZE and position != len(order) - 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on_batch(
                agent, reranker, batch_texts, batch_picks, torch.tensor(batch_golds), crit, args.pairwise_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_texts, batch_picks, batch_golds = [], [], []
            step += 1
            if step % 100 == 0:
                print("  epoch %d step %d loss %.4f" % (epoch, step, float(np.mean(losses[-100:]))), flush=True)
        reranker.eval()
        print("validate epoch", epoch, flush=True)
        scored = score_sets(agent, [row[0] for row in val_rows], val_picks, crit, temperature=1.0, reranker=reranker, batch_size=4)
        hit_new, high_rate = conditional_and_highconf(val_rows, val_picks, scored)
        # Align hit vectors with the baseline, which iterates target_in_top8 in the same order.
        if len(hit_new) != len(base_hits):
            raise SystemExit("hit count changed; the shortlist was not held fixed")
        boot = paired_bootstrap(base_hits, hit_new)
        accuracy = float(np.mean(hit_new))
        print("  conditional", accuracy, "delta", boot["delta"], "ci", boot["ci95_low"], boot["ci95_high"], "high", high_rate, flush=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "conditional_accuracy": accuracy, "bootstrap": boot, "high_conf_rate": high_rate})
        if accuracy > best["accuracy"]:
            best = {
                "accuracy": accuracy,
                "epoch": epoch,
                "state": {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                          for name, module in (("head", reranker.head), ("type_emb", reranker.type_emb), ("scorer", reranker.scorer))},
                "bootstrap": boot,
                "high_conf_rate": high_rate,
            }
        elif epoch >= 1:
            print("early stop", flush=True)
            break

    reranker.head.load_state_dict(best["state"]["head"])
    reranker.type_emb.load_state_dict(best["state"]["type_emb"])
    reranker.scorer.load_state_dict(best["state"]["scorer"])
    reranker.eval()
    promoted, reason = promote(best["bootstrap"]["ci95_low"], best["high_conf_rate"], base_high)
    manifest_note = {
        "variant": "listwise_plus_pairwise" if args.pairwise_weight else "listwise_shortlist_80_random_20",
        "pairwise_weight": args.pairwise_weight,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "best_epoch": best["epoch"],
        "trainable_rows": len(pool),
        "target_absent_rows": absent,
        "init": "phase5_decision_head",
        "encoder": "frozen",
        "split": "banking77_train_seed17_800_8600",
    }
    out_dir = os.path.join(ART, "candidates")
    os.makedirs(out_dir, exist_ok=True)
    artifact_name = "k8_pairwise.pt" if args.pairwise_weight else "k8_listwise.pt"
    side = save_reranker(
        os.path.join(out_dir, artifact_name),
        reranker,
        bundle.checkpoint_hash,
        bundle.shortlist_sha256,
        manifest_note,
    )
    decision = {
        "promoted": promoted,
        "reason": reason,
        "best": {key: value for key, value in best.items() if key != "state"},
        "baseline_conditional_hits": len(base_hits),
        "baseline_high_conf_rate": base_high,
        "history": history,
        "artifact": "artifacts/experiments/phase6/candidates/" + artifact_name,
        "artifact_sha256": side["file_sha256"],
    }
    decision_path = os.path.join(ART, "train_decision_pairwise.json" if args.pairwise_weight else "train_decision.json")
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: decision[key] for key in ("promoted", "reason", "best")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
