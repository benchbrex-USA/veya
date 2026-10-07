"""Route-specific K=8 packed reranker.

The base decision head still scores packed Choice, Score, and Noul. This module
is a copy of that head, loaded only for the shortlist route. It does not write
into the base checkpoint.
"""

import torch
import torch.nn as nn

from .common import QTYPES, build_sequence, collate_items
from .shortlist_artifact import ROUTE, file_sha256

FORMAT = "veya-k8-reranker-v1"


def _new_modules(dim=1024, layers=2, dropout=0.1):
    nhead = max(1, dim // 64)
    layer = nn.TransformerEncoderLayer(dim, nhead, 4 * dim, dropout, batch_first=True, norm_first=True)
    head = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    type_emb = nn.Embedding(3, dim)
    scorer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    return head, type_emb, scorer


class K8Reranker(nn.Module):
    def __init__(self, dim=1024, layers=2, dropout=0.1):
        super().__init__()
        head, type_emb, scorer = _new_modules(dim, layers, dropout)
        self.head = head
        self.type_emb = type_emb
        self.scorer = scorer
        self.meta = {}

    def clone_from_agent(self, agent):
        """Copy the production decision head. The agent's own modules stay put."""
        self.head.load_state_dict(agent.model.head.state_dict())
        self.type_emb.load_state_dict(agent.model.type_emb.state_dict())
        self.scorer.load_state_dict(agent.model.scorer.state_dict())
        return self

    def score_hidden(self, hidden, attention_mask, marker_pos, marker_mask, qtype):
        h = hidden + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        gathered = torch.gather(h, 1, index)
        logits = self.scorer(gathered).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)

    def choice_logits(self, agent, state, instructions, criteria, max_len=None, head_max_len=None, option_token_cap=48):
        question = {"t": "choice", "ins": instructions, "crit": criteria}
        max_len = int(agent.cfg.get("max_len", 512) if max_len is None else max_len)
        head_max_len = int(agent.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len)
        seq, markers = build_sequence(
            agent.tok, state, question, max_len, head_max_len, option_token_cap=int(option_token_cap),
        )
        if len(markers) != len(criteria):
            raise ValueError("packed head kept %d of %d options" % (len(markers), len(criteria)))
        items = [{"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}]
        batch = collate_items([items], agent.tok.pad_token_id)
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            hidden = agent.model.encoder(
                input_ids=batch["input_ids"].to(agent.device),
                attention_mask=batch["attention_mask"].to(agent.device),
            ).last_hidden_state
            logits = self.score_hidden(
                hidden.to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        return logits[0, : len(criteria)].float().cpu().numpy(), int(batch["attention_mask"].sum())


def save_reranker(path, reranker, checkpoint_hash, shortlist_sha256, training_manifest):
    blob = {
        "format": FORMAT,
        "route": ROUTE,
        "k": 8,
        "checkpoint_hash": checkpoint_hash,
        "shortlist_sha256": shortlist_sha256,
        "training_manifest": training_manifest,
        "state_dict": {
            "head": {key: value.detach().cpu() for key, value in reranker.head.state_dict().items()},
            "type_emb": {key: value.detach().cpu() for key, value in reranker.type_emb.state_dict().items()},
            "scorer": {key: value.detach().cpu() for key, value in reranker.scorer.state_dict().items()},
        },
    }
    torch.save(blob, path)
    side = {key: value for key, value in blob.items() if key != "state_dict"}
    side["file_sha256"] = file_sha256(path)
    import json
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump(side, handle, indent=2)
        handle.write("\n")
    return side


def load_reranker(path, agent, checkpoint_hash, shortlist_sha256):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != FORMAT:
        raise ValueError("unsupported reranker format %r" % blob.get("format"))
    if blob.get("route") != ROUTE or int(blob.get("k", -1)) != 8:
        raise ValueError("reranker is not the K=8 shortlist route")
    if blob.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("reranker was trained for a different checkpoint")
    if blob.get("shortlist_sha256") != shortlist_sha256:
        raise ValueError("reranker was trained for a different shortlist")
    reranker = K8Reranker()
    reranker.head.load_state_dict(blob["state_dict"]["head"])
    reranker.type_emb.load_state_dict(blob["state_dict"]["type_emb"])
    reranker.scorer.load_state_dict(blob["state_dict"]["scorer"])
    reranker.meta = {key: value for key, value in blob.items() if key != "state_dict"}
    reranker.meta["file_sha256"] = file_sha256(path)
    reranker.eval()
    return reranker
