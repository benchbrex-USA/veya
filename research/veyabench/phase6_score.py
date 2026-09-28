"""Shared K=8 scoring. The release test is not referenced here."""
import numpy as np
import torch

from veya.common import QTYPES, build_sequence, collate_items
from veya.release_bundle import INSTRUCTIONS


def softmax_row(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _forward(agent, items, reranker):
    batch = collate_items([items], agent.tok.pad_token_id)
    device = agent.device
    with torch.no_grad():
        if reranker is None:
            logits, _act = agent.model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        else:
            hidden = agent.model.encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).last_hidden_state
            target = next(reranker.parameters()).device
            logits = reranker.score_hidden(
                hidden.to(target),
                batch["attention_mask"].to(target),
                batch["marker_pos"].to(target),
                batch["marker_mask"].to(target),
                batch["qtype"].to(target),
            )
    return logits.float().cpu().numpy(), batch["attention_mask"].sum(dim=1).tolist()


def score_sets(agent, texts, picked_lists, crit, temperature=1.0, reranker=None, batch_size=4,
               head_max_len=None, option_token_cap=48, instructions=INSTRUCTIONS):
    """Score each example's ordered candidate list. One probability row per example."""
    max_len = int(agent.cfg.get("max_len", 512))
    if head_max_len is None:
        head_max_len = int(agent.cfg.get("head_max_len", 192))
    records = []
    cursor = 0
    total = len(texts)
    while cursor < total:
        chunk_texts = texts[cursor:cursor + batch_size]
        chunk_picks = picked_lists[cursor:cursor + batch_size]
        items = []
        widths = []
        for text, picked in zip(chunk_texts, chunk_picks):
            criteria = {label: crit[label] for label in picked}
            question = {"t": "choice", "ins": instructions, "crit": criteria}
            seq, markers = build_sequence(
                agent.tok, text, question, max_len, int(head_max_len), option_token_cap=int(option_token_cap),
            )
            if len(markers) != len(picked):
                raise RuntimeError("kept %d of %d candidates" % (len(markers), len(picked)))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
            widths.append(len(picked))
        logits, tokens = _forward(agent, items, reranker)
        for row, picked, width, n_tokens in zip(logits, chunk_picks, widths, tokens):
            raw = row[:width]
            probs = softmax_row(raw, temperature)
            pred = picked[int(np.argmax(probs))]
            order = np.argsort(-probs)
            margin = float(probs[order[0]] - probs[order[1]]) if width >= 2 else float(probs[0])
            records.append({
                "logits": [float(value) for value in raw],
                "probabilities": [float(value) for value in probs],
                "prediction": pred,
                "confidence": float(probs.max()),
                "margin": margin,
                "input_tokens": int(n_tokens),
                "candidates": list(picked),
            })
        cursor += batch_size
        if total >= 50 and (cursor % 400 == 0 or cursor >= total):
            print("  scored %d/%d" % (min(cursor, total), total), flush=True)
    return records
