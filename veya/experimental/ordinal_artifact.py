"""Artifact schema for an ordinal Score head.

The schema can round-trip a state dict. It cannot be marked production.
Categorical Score remains the served head until a held-out ordinal comparison
exists.
"""
import json

import torch

FORMAT = "veya-ordinal-score-v1"


def save_ordinal(path, head, label_order):
    if getattr(head, "n_levels", None) != len(label_order):
        raise ValueError("label_order length does not match the ordinal head")
    blob = {
        "format": FORMAT,
        "status": "research",
        "production": False,
        "label_order": list(label_order),
        "n_levels": int(head.n_levels),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
    }
    torch.save(blob, path)
    side = {
        "format": FORMAT,
        "status": "research",
        "production": False,
        "label_order": list(label_order),
        "n_levels": int(head.n_levels),
    }
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump(side, handle, indent=2)
        handle.write("\n")
    return side


def load_ordinal(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != FORMAT:
        raise ValueError("unsupported ordinal artifact %r" % blob.get("format"))
    if blob.get("production"):
        raise ValueError("ordinal Score is not a production artifact")
    if blob.get("status") != "research":
        raise ValueError("ordinal artifact must stay status=research")
    return blob
