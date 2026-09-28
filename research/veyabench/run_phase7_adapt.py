"""Upper-layer encoder adaptation for hard K=8 Choice.

The shortlist and the release test stay frozen. Every run restores the
Phase-6 control weights before the next one. A checkpoint is written for a
later sealed test only when the validation gate passes.
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
    RERANKER_TRAIN, RERANKER_VAL, assert_training_id, example_id, paired_bootstrap,
)
from research.veyabench.phase6_score import score_sets, softmax_row
from research.veyabench.run_phase3_eval import ROOT, SEED, choice_logits, load_ag_news, load_banking, metrics
from research.veyabench.run_release_eval import load_from_manifest
from veya.common import QTYPES, build_sequence, collate_items
from veya.release_bundle import INSTRUCTIONS

ART = os.path.join(ROOT, "artifacts", "experiments", "phase7")
PROBE_LAYERS = (6, 13, 20, 27)
SHORT_STEPS = 300
SHORT_POOL = 512
VAL_SHORT = 200
# The first ladder at 2e-5 / 1e-4 spiked (pre-clip grad norms above 70).
# These rates are the validation correction from that unstable short run.
ENCODER_LR = 1e-6
HEAD_LR = 5e-6
RETRY_ENCODER_LR = 5e-6
RETRY_HEAD_LR = 1e-5


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def convert(value):
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        raise TypeError(type(value).__name__)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert)
        handle.write("\n")


def snapshot(modules):
    return [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()} for module in modules]


def restore(modules, states):
    for module, state in zip(modules, states):
        module.load_state_dict(state)


def encode_suffix(encoder, input_ids, attention_mask, start_layer):
    """Frozen prefix under no_grad. Gradients start at start_layer."""
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        hidden = encoder.embeddings(input_ids=input_ids)
        mask_kwargs = {"config": encoder.config, "inputs_embeds": hidden, "attention_mask": attention_mask}
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        positions = {}
        for layer_type in set(encoder.config.layer_types):
            positions[layer_type] = encoder.rotary_emb(hidden, position_ids, layer_type)
        for layer in encoder.layers[:start_layer]:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_embeddings=positions[layer.attention_type],
            )
        hidden = hidden.detach()
    for layer in encoder.layers[start_layer:]:
        hidden = layer(
            hidden,
            attention_mask=masks[layer.attention_type],
            position_embeddings=positions[layer.attention_type],
        )
    return encoder.final_norm(hidden)


def score_hidden(model, hidden, attention_mask, marker_pos, marker_mask, qtype):
    h = hidden + model.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    gathered = torch.gather(h, 1, index)
    logits = model.scorer(gathered).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def batch_items(agent, texts, picks, crit):
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    items = []
    for text, picked in zip(texts, picks):
        criteria = {label: crit[label] for label in picked}
        question = {"t": "choice", "ins": INSTRUCTIONS, "crit": criteria}
        seq, markers = build_sequence(agent.tok, text, question, max_len, head_max_len)
        if len(markers) != 8:
            raise RuntimeError("row did not keep 8 candidates")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
    return collate_items([items], agent.tok.pad_token_id)


def trainable_modules(agent, n_layers):
    encoder = agent.model.encoder
    depth = len(encoder.layers)
    start = depth - int(n_layers)
    modules = list(encoder.layers[start:]) + [encoder.final_norm, agent.model.head, agent.model.type_emb, agent.model.scorer]
    return start, modules


def set_trainable(agent, n_layers):
    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    start, modules = trainable_modules(agent, n_layers)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return start


def optimizer_for(agent, n_layers, encoder_lr, head_lr):
    start, _modules = trainable_modules(agent, n_layers)
    encoder = agent.model.encoder
    encoder_params = [parameter for layer in encoder.layers[start:] for parameter in layer.parameters()]
    encoder_params += list(encoder.final_norm.parameters())
    head_params = list(agent.model.head.parameters()) + list(agent.model.type_emb.parameters()) + list(agent.model.scorer.parameters())
    return torch.optim.AdamW(
        [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=0.01,
    )


def random_set(labels, truth, rng):
    others = [label for label in labels if label != truth]
    rng.shuffle(others)
    picked = [truth] + others[:7]
    rng.shuffle(picked)
    return picked


def hot_labels(limit_count=5):
    path = os.path.join(ROOT, "artifacts", "experiments", "phase6", "train_confusion.json")
    edges = json.load(open(path, encoding="utf-8"))["top_edges"]
    return {edge["true"] for edge in edges if edge["count"] >= limit_count}


def build_pool(bundle, labels, crit):
    train = load_banking("train")
    rows = train[RERANKER_TRAIN[0]:RERANKER_TRAIN[1]]
    queries = bundle.project(bundle.embed([row[0] for row in rows]))
    matrix = bundle.project(bundle.embed(bundle.label_texts))
    ranked = torch.tensor(np.asarray(queries @ matrix.T)).topk(8, dim=1).indices.tolist()
    pool = []
    absent = 0
    for offset, ((text, truth), top) in enumerate(zip(rows, ranked)):
        index = RERANKER_TRAIN[0] + offset
        assert_training_id(example_id("train", index))
        picked = [labels[i] for i in top]
        if truth not in picked:
            absent += 1
            continue
        pool.append({"index": index, "text": text, "truth": truth, "shortlist": picked})
    return pool, absent


def one_loss(agent, batch, start_layer):
    hidden = encode_suffix(agent.model.encoder, batch["input_ids"].to(agent.device), batch["attention_mask"].to(agent.device), start_layer)
    logits = score_hidden(
        agent.model,
        hidden,
        batch["attention_mask"].to(agent.device),
        batch["marker_pos"].to(agent.device),
        batch["marker_mask"].to(agent.device),
        batch["qtype"].to(agent.device),
    )
    return F.cross_entropy(logits[:, :8], batch["gold"].to(agent.device))


def run_steps(agent, pool, labels, crit, n_layers, steps, encoder_lr, head_lr, mix, oversample, seed, batch_size=2):
    start = set_trainable(agent, n_layers)
    agent.model.encoder.eval()
    agent.model.head.train()
    opt = optimizer_for(agent, n_layers, encoder_lr, head_lr)
    hot = hot_labels() if oversample else set()
    work = []
    for row in pool:
        work.append(row)
        if row["truth"] in hot:
            work.append(row)
    losses = []
    grad_norms = []
    for step in range(steps):
        texts, picks, golds = [], [], []
        for slot in range(batch_size):
            row = work[(seed + step * batch_size + slot) % len(work)]
            rng = random.Random(seed + 10007 * step + slot + row["index"])
            mode = mix
            if mix == "mixed":
                mode = "generic" if ((step + slot) % 2 == 0) else "hard"
            if mode == "generic":
                picked = random_set(labels, row["truth"], rng)
            else:
                picked = list(row["shortlist"])
                rng.shuffle(picked)
            texts.append(row["text"])
            picks.append(picked)
            golds.append(picked.index(row["truth"]))
        batch = batch_items(agent, texts, picks, crit)
        batch["gold"] = torch.tensor(golds)
        opt.zero_grad(set_to_none=True)
        loss = one_loss(agent, batch, start)
        loss.backward()
        params = [parameter for parameter in agent.model.parameters() if parameter.requires_grad and parameter.grad is not None]
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % 50 == 0:
            print("  step %d loss %.4f grad %.3f lr_enc %g lr_head %g" % (step, losses[-1], grad_norms[-1], encoder_lr, head_lr), flush=True)
    agent.model.head.eval()
    early = float(np.mean(losses[:20]))
    late = float(np.mean(losses[-20:]))
    return {
        "start_layer": start,
        "steps": steps,
        "loss_first20": early,
        "loss_last20": late,
        "loss_dropped": (early - late) / max(early, 1e-6),
        "max_loss": float(np.max(losses)),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "trainable": int(sum(parameter.numel() for parameter in agent.model.parameters() if parameter.requires_grad)),
    }


def conditional_on(records, scored):
    base, new, high_base, high_new = [], [], 0, 0
    for record, fresh in zip(records, scored):
        base_conf = float(softmax_row(record["reranker_logits"], 1.0).max())
        if not record["correct"] and base_conf >= 0.70:
            high_base += 1
        if not fresh["correct_flag"] and fresh["confidence"] >= 0.70:
            high_new += 1
        if record["target_in_top8"]:
            base.append(1.0 if record["correct"] else 0.0)
            new.append(1.0 if fresh["correct_flag"] else 0.0)
    return base, new, high_base / len(records), high_new / len(records)


def evaluate(agent, rows, picks, truths, crit):
    scored = score_sets(agent, rows, picks, crit, temperature=1.0, batch_size=4)
    out = []
    for record, truth in zip(scored, truths):
        out.append({
            "correct_flag": record["prediction"] == truth,
            "confidence": record["confidence"],
            "margin": record["margin"],
            "prediction": record["prediction"],
            "probabilities": record["probabilities"],
            "candidates": record["candidates"],
        })
    return out


def layer_probe(agent, rows, picks, truths, crit):
    encoder = agent.model.encoder
    captured = {index: [] for index in PROBE_LAYERS}
    hooks = []
    for index in PROBE_LAYERS:
        def _hook(module, inputs, output, index=index):
            captured[index].append(output.detach())
        hooks.append(encoder.layers[index].register_forward_hook(_hook))
    bags = {index: [] for index in PROBE_LAYERS}
    golds = []
    kept = 0
    with torch.no_grad():
        for text, picked, truth in zip(rows, picks, truths):
            if truth not in picked:
                continue
            for index in PROBE_LAYERS:
                captured[index].clear()
            batch = batch_items(agent, [text], [picked], crit)
            agent.model.encoder(
                input_ids=batch["input_ids"].to(agent.device),
                attention_mask=batch["attention_mask"].to(agent.device),
            )
            for index in PROBE_LAYERS:
                hidden = captured[index][0][0]
                markers = batch["marker_pos"][0].tolist()
                bags[index].append(torch.stack([hidden[pos].cpu() for pos in markers[:8]]))
            golds.append(picked.index(truth))
            kept += 1
    for hook in hooks:
        hook.remove()
    golds = torch.tensor(golds)
    report = {"n": kept}
    for index in PROBE_LAYERS:
        data = torch.stack(bags[index])
        split = max(1, int(0.8 * len(data)))
        probe = torch.nn.Linear(data.shape[-1], 1)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        for _epoch in range(40):
            opt.zero_grad()
            loss = F.cross_entropy(probe(data[:split]).squeeze(-1), golds[:split])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(data[split:]).squeeze(-1).argmax(-1)
            accuracy = float((pred == golds[split:]).float().mean()) if len(golds[split:]) else None
        report["layer_%d" % index] = {"held_out_probe_accuracy": accuracy}
    return report


def small_choice(agent, rows, labels, crit):
    correct = []
    temperature = float(agent.temperature_by_options.get("choice:3-5", agent.temperature[0]))
    for i, (text, truth) in enumerate(rows):
        rng = random.Random(SEED + 333 + i)
        others = [label for label in labels if label != truth]
        rng.shuffle(others)
        picked = [truth] + others[:3]
        rng.shuffle(picked)
        criteria = {label: crit[label] for label in picked}
        logits, _tokens = __import__("veya.release_bundle", fromlist=["raw_choice_logits"]).raw_choice_logits(
            agent, text, INSTRUCTIONS, criteria,
        )
        pred = picked[int(np.argmax(softmax_row(logits, temperature)))]
        correct.append(pred == truth)
    return float(np.mean(correct)) if correct else None


def anchors(agent):
    noul = agent.system_one("I was charged twice for the transfer", {"q": {"type": "noul", "instructions": "Was the customer charged twice?"}})
    score = agent.system_one("The card has not arrived and I travel tomorrow", {"q": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}})
    choice = agent.system_one("Where is my refund?", {"q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {"billing": "billing", "shipping": "shipping", "technical": "technical"}}})
    return {
        "noul": noul["answers"]["q"]["noul"],
        "score": score["answers"]["q"]["score"],
        "choice": choice["answers"]["q"]["choice"],
        "choice_confidence": choice["answers"]["q"]["confidence"],
    }


def ag_news(agent):
    rows = load_ag_news()[400:520]
    names = ["World", "Sports", "Business", "Sci/Tech"]
    crit = {name: "news about %s" % name for name in names}
    parts = [choice_logits(agent, [row[0] for row in rows[start:start + 8]], crit) for start in range(0, len(rows), 8)]
    return metrics(np.concatenate(parts), np.array([row[1] for row in rows]), float(agent.temperature_by_options["choice:3-5"]))


def paraphrase_agreement(agent, rows, picks, crit):
    agree = 0
    used = 0
    for (text, truth), picked in list(zip(rows, picks))[:40]:
        if truth not in picked:
            continue
        original = {label: crit[label] for label in picked}
        paraphrased = {label: "a customer request about %s" % label.replace("_", " ") for label in picked}
        base = score_sets(agent, [text], [picked], original, temperature=1.0, batch_size=1)[0]
        # score_sets looks up crit[label], so pass paraphrased as the crit map via picked order.
        alt_crit = dict(crit)
        alt_crit.update(paraphrased)
        alt = score_sets(agent, [text], [picked], alt_crit, temperature=1.0, batch_size=1)[0]
        agree += int(base["prediction"] == alt["prediction"])
        used += 1
    return {"n": used, "agreement": (agree / used) if used else None}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-layers", type=int, default=0)
    parser.add_argument("--full-mix", default="hard")
    parser.add_argument("--encoder-lr", type=float, default=ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    args = parser.parse_args()
    os.makedirs(ART, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.load(open(os.path.join(ROOT, "artifacts", "release", "manifest.json"), encoding="utf-8"))
    print("loading", device, flush=True)
    bundle = load_from_manifest(manifest, device, {})
    agent = bundle.agent
    labels = list(bundle.labels)
    crit = {label: text for label, text in zip(labels, bundle.label_texts)}
    encoder = agent.model.encoder
    if not hasattr(encoder, "layers"):
        raise SystemExit("encoder has no layers: %s" % type(encoder).__name__)
    depth = len(encoder.layers)
    print("encoder layers", depth, flush=True)
    control_modules = list(encoder.layers[-4:]) + [encoder.final_norm, agent.model.head, agent.model.type_emb, agent.model.scorer]
    control = snapshot(control_modules)
    dump(os.path.join(ART, "control.json"), {
        "release_manifest": "artifacts/release/manifest.json",
        "checkpoint_sha256": bundle.checkpoint_hash,
        "shortlist_sha256": bundle.shortlist_sha256,
        "route_temperature": bundle.route_temperature,
        "reranker_id": bundle.reranker_id,
        "choice_router_hash": manifest["routing"]["choice_router_hash"],
        "packed_calibration_hash": manifest["calibration"]["packed_choice_hash"],
        "hierarchical_calibration_hash": manifest["calibration"]["hierarchical_choice_hash"],
        "release_test": [1300, 3080],
        "note": "Release-test predictions stay in artifacts/experiments/phase6/frozen_test. This run does not read them.",
    })

    # Match the stock encoder forward before any update.
    probe_batch = batch_items(agent, ["Where is my card?"], [labels[:8]], crit)
    with torch.no_grad():
        stock = encoder(input_ids=probe_batch["input_ids"].to(device), attention_mask=probe_batch["attention_mask"].to(device)).last_hidden_state
        copied = encode_suffix(encoder, probe_batch["input_ids"].to(device), probe_batch["attention_mask"].to(device), depth)
    if not torch.allclose(stock, copied, atol=1e-4):
        raise SystemExit("suffix encoder does not match the stock forward")
    print("encoder path matches", flush=True)

    print("building hard pool", flush=True)
    pool, absent = build_pool(bundle, labels, crit)
    print("pool", len(pool), "absent", absent, flush=True)
    val_records = json.load(open(os.path.join(ROOT, "artifacts", "experiments", "phase6", "val_predictions.json"), encoding="utf-8"))
    train_rows = load_banking("train")
    val_rows = [(train_rows[RERANKER_VAL[0] + i][0], record["target"]) for i, record in enumerate(val_records)]
    val_picks = [record["retrieved_top8"] for record in val_records]
    short_records = val_records[:VAL_SHORT]
    short_rows = val_rows[:VAL_SHORT]
    short_picks = val_picks[:VAL_SHORT]

    if args.full_layers:
        print("full epoch only", args.full_layers, args.full_mix, flush=True)
        restore(control_modules, control)
        base_anchors = anchors(agent)
        base_ag = ag_news(agent)
        base_small = small_choice(agent, short_rows, labels, crit)
        print("control ag", base_ag["accuracy"], "small4", base_small, flush=True)
        stats = run_steps(
            agent, pool, labels, crit, args.full_layers, max(1, len(pool) // 2),
            args.encoder_lr, args.head_lr, args.full_mix, False, SEED + 99,
        )
        fresh = evaluate(agent, [row[0] for row in val_rows], val_picks, [row[1] for row in val_rows], crit)
        base, new, high_base, high_new = conditional_on(val_records, fresh)
        boot = paired_bootstrap(base, new)
        ag = ag_news(agent)
        small = small_choice(agent, short_rows, labels, crit)
        after_anchors = anchors(agent)
        paraphrase = paraphrase_agreement(agent, short_rows, short_picks, crit)
        reasons = []
        if boot["ci95_low"] <= 0:
            reasons.append("conditional interval includes zero")
        if high_new > high_base + 0.02:
            reasons.append("high-confidence errors rose")
        if base_ag["accuracy"] - ag["accuracy"] > 0.02:
            reasons.append("AG News dropped by more than 0.02")
        if base_small - small > 0.03:
            reasons.append("4-way Choice dropped by more than 0.03")
        if abs(after_anchors["noul"] - base_anchors["noul"]) > 0.10 or abs(after_anchors["score"] - base_anchors["score"]) > 0.15:
            reasons.append("Noul or Score anchor moved past the preservation gate")
        full = {
            "n_layers": args.full_layers,
            "mix": args.full_mix,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "train": stats,
            "conditional": float(np.mean(new)),
            "bootstrap": boot,
            "high_conf_rate_T1": high_new,
            "baseline_high_conf_rate_T1": high_base,
            "ag_news": ag,
            "baseline_ag_news": base_ag,
            "small4": small,
            "baseline_small4": base_small,
            "anchors": after_anchors,
            "baseline_anchors": base_anchors,
            "paraphrase": paraphrase,
            "promoted": not reasons,
            "reasons": reasons,
        }
        if full["promoted"]:
            start = depth - args.full_layers
            pieces = [("layer_%d" % index, encoder.layers[index]) for index in range(start, depth)]
            pieces += [("final_norm", encoder.final_norm), ("head", agent.model.head), ("type_emb", agent.model.type_emb), ("scorer", agent.model.scorer)]
            cpu_delta = {name: {key: value.detach().cpu() for key, value in module.state_dict().items()} for name, module in pieces}
            torch.save({
                "format": "veya-encoder-adapt-v1",
                "n_layers": args.full_layers,
                "start_layer": start,
                "checkpoint_hash": bundle.checkpoint_hash,
                "shortlist_sha256": bundle.shortlist_sha256,
                "state_dict": cpu_delta,
            }, os.path.join(ART, "finalist.pt"))
        dump(os.path.join(ART, "full_run.json"), full)
        print(json.dumps({"promoted": full["promoted"], "reasons": reasons, "conditional": full["conditional"], "delta": boot["delta"], "ci": [boot["ci95_low"], boot["ci95_high"]]}, indent=2), flush=True)
        return

    print("layer probe", flush=True)
    probe = layer_probe(
        agent,
        [row[0] for row in short_rows],
        short_picks,
        [row[1] for row in short_rows],
        crit,
    )
    dump(os.path.join(ART, "layer_probe.json"), probe)
    print(json.dumps(probe), flush=True)

    print("baseline anchors and AG News", flush=True)
    base_anchors = anchors(agent)
    base_ag = ag_news(agent)
    base_small = small_choice(agent, short_rows, labels, crit)
    dump(os.path.join(ART, "control_behavior.json"), {"anchors": base_anchors, "ag_news_400_520": base_ag, "small4": base_small})

    def eval_short():
        fresh = evaluate(agent, [row[0] for row in short_rows], short_picks, [row[1] for row in short_rows], crit)
        base, new, high_base, high_new = conditional_on(short_records, fresh)
        boot = paired_bootstrap(base, new, n=400, seed=SEED)
        return {"conditional": float(np.mean(new)), "bootstrap": boot, "high_conf_rate": high_new, "baseline_high_conf_rate": high_base}

    experiments = []

    def short_experiment(name, n_layers, mix, encoder_lr, head_lr, oversample=False):
        restore(control_modules, control)
        print("short", name, "layers", n_layers, "mix", mix, flush=True)
        stats = run_steps(agent, pool[:SHORT_POOL], labels, crit, n_layers, SHORT_STEPS, encoder_lr, head_lr, mix, oversample, SEED + n_layers)
        metrics_out = eval_short()
        row = {"name": name, "n_layers": n_layers, "mix": mix, "encoder_lr": encoder_lr, "head_lr": head_lr, "oversample": oversample, **stats, **metrics_out}
        experiments.append(row)
        print("  conditional", row["conditional"], "delta", row["bootstrap"]["delta"], "loss_drop", row["loss_dropped"], flush=True)
        restore(control_modules, control)
        return row

    e1 = short_experiment("E1_hard", 1, "hard", ENCODER_LR, HEAD_LR)
    e2 = short_experiment("E2_hard", 2, "hard", ENCODER_LR, HEAD_LR)
    ran_e4 = e2["conditional"] + 0.005 >= e1["conditional"] and e2["loss_dropped"] >= e1["loss_dropped"] - 0.05
    if ran_e4:
        short_experiment("E4_hard", 4, "hard", ENCODER_LR, HEAD_LR)
    else:
        print("skip top-4; top-2 did not improve on top-1", flush=True)
    hard_rows = [row for row in experiments if row["mix"] == "hard"]
    best_hard = max(hard_rows, key=lambda row: row["conditional"])
    short_experiment("mix_generic", best_hard["n_layers"], "generic", ENCODER_LR, HEAD_LR)
    short_experiment("mix_half", best_hard["n_layers"], "mixed", ENCODER_LR, HEAD_LR)
    learning = [row for row in experiments if row["loss_dropped"] >= 0.10 and row["max_loss"] < 3.0]
    if not learning:
        print("retry higher lr on best hard depth", flush=True)
        short_experiment("retry_lr", best_hard["n_layers"], "hard", RETRY_ENCODER_LR, RETRY_HEAD_LR)
        learning = [row for row in experiments if row["loss_dropped"] >= 0.10 and row["max_loss"] < 3.0]
    dump(os.path.join(ART, "short_runs.json"), experiments)

    full = None
    stable = [row for row in learning if row["mean_grad_norm"] < 30]
    learning = stable or learning
    if learning:
        chosen = max(learning, key=lambda row: row["conditional"])
        print("full epoch", chosen["name"], flush=True)
        restore(control_modules, control)
        # One pass over the full hard or mixed pool. Step count equals the pool size.
        steps = max(1, len(pool) // 2)
        stats = run_steps(
            agent, pool, labels, crit, chosen["n_layers"], steps,
            chosen["encoder_lr"], chosen["head_lr"], chosen["mix"], False, SEED + 99,
        )
        fresh = evaluate(agent, [row[0] for row in val_rows], val_picks, [row[1] for row in val_rows], crit)
        base, new, high_base, high_new = conditional_on(val_records, fresh)
        boot = paired_bootstrap(base, new)
        ag = ag_news(agent)
        small = small_choice(agent, short_rows, labels, crit)
        after_anchors = anchors(agent)
        paraphrase = paraphrase_agreement(agent, short_rows, short_picks, crit)
        full = {
            "chosen": chosen["name"],
            "train": stats,
            "conditional": float(np.mean(new)),
            "bootstrap": boot,
            "high_conf_rate_T1": high_new,
            "baseline_high_conf_rate_T1": high_base,
            "ag_news": ag,
            "small4": small,
            "anchors": after_anchors,
            "anchor_abs_delta": {
                "noul": abs(after_anchors["noul"] - base_anchors["noul"]),
                "score": abs(after_anchors["score"] - base_anchors["score"]),
            },
            "paraphrase": paraphrase,
            "n_hits": len(base),
        }
        reasons = []
        if boot["ci95_low"] <= 0:
            reasons.append("conditional interval includes zero")
        if high_new > high_base + 0.02:
            reasons.append("high-confidence errors rose")
        if base_ag["accuracy"] - ag["accuracy"] > 0.02:
            reasons.append("AG News dropped by more than 0.02")
        if base_small - small > 0.03:
            reasons.append("4-way Choice dropped by more than 0.03")
        if full["anchor_abs_delta"]["noul"] > 0.10 or full["anchor_abs_delta"]["score"] > 0.15:
            reasons.append("Noul or Score anchor moved past the preservation gate")
        full["promoted"] = not reasons
        full["reasons"] = reasons
        if full["promoted"]:
            start, modules = trainable_modules(agent, chosen["n_layers"])
            delta = {module_name: module.state_dict() for module_name, module in (
                [("layer_%d" % index, encoder.layers[index]) for index in range(start, depth)]
                + [("final_norm", encoder.final_norm), ("head", agent.model.head), ("type_emb", agent.model.type_emb), ("scorer", agent.model.scorer)]
            )}
            # state_dict tensors stay on device; move them.
            cpu_delta = {name: {key: value.detach().cpu() for key, value in state.items()} for name, state in delta.items()}
            torch.save({
                "format": "veya-encoder-adapt-v1",
                "n_layers": chosen["n_layers"],
                "start_layer": start,
                "checkpoint_hash": bundle.checkpoint_hash,
                "shortlist_sha256": bundle.shortlist_sha256,
                "state_dict": cpu_delta,
            }, os.path.join(ART, "finalist.pt"))
        dump(os.path.join(ART, "full_run.json"), full)
        print(json.dumps({"promoted": full["promoted"], "reasons": reasons, "conditional": full["conditional"], "bootstrap": boot}, indent=2), flush=True)
    else:
        dump(os.path.join(ART, "full_run.json"), {"promoted": False, "reasons": ["no short run reduced loss by 10 percent"]})
        print("no configuration learned; control unchanged", flush=True)
    dump(os.path.join(ART, "short_runs.json"), experiments)


if __name__ == "__main__":
    main()
