"""Saved K=8 shortlist projection bound to one base checkpoint.

The projection is a bias-free linear map. Query and label vectors are projected
with the same matrix and scored by a dot product. A file trained for another
checkpoint, or marked for another K, is refused.
"""
import hashlib
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn

ROUTE = "shortlist_k8_packed_rerank"
FORMAT = "veya-shortlist-v1"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_shortlist(path, projection, labels, label_texts, checkpoint_hash, k=8, seed=17):
    projection = projection.cpu().eval()
    weight = projection.weight.detach()
    blob = {
        "format": FORMAT,
        "route": ROUTE,
        "k": int(k),
        "seed": int(seed),
        "checkpoint_hash": checkpoint_hash,
        "input_dim": int(weight.shape[1]),
        "output_dim": int(weight.shape[0]),
        "labels": list(labels),
        "label_texts": list(label_texts),
        "state_dict": {"weight": weight},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    torch.save(blob, path)
    side = path + ".json"
    meta = {key: value for key, value in blob.items() if key != "state_dict"}
    meta["file_sha256"] = file_sha256(path)
    with open(side, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta


def load_shortlist(path, checkpoint_hash, k=8):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT:
        raise ValueError("unsupported shortlist format %r" % blob.get("format"))
    if blob.get("route") != ROUTE:
        raise ValueError("shortlist route %r is not %s" % (blob.get("route"), ROUTE))
    if int(blob.get("k", -1)) != int(k):
        raise ValueError("shortlist K %s does not match requested K %s" % (blob.get("k"), k))
    if blob.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("shortlist was trained for a different checkpoint")
    layer = nn.Linear(blob["input_dim"], blob["output_dim"], bias=False)
    layer.load_state_dict(blob["state_dict"])
    layer.eval()
    return layer, blob


def save_route_calibration(path, temperature, sample_count, shortlist_sha256, checkpoint_hash, fit_split,
                           fit_split_hash=None, scorer_id="base_decision_head"):
    artifact = {
        "format": "veya-route-calibration-v1",
        "route": ROUTE,
        "primitive": "choice",
        "k": 8,
        "method": "temperature-nll",
        "temperature": float(temperature),
        "sample_count": int(sample_count),
        "fitting_split": fit_split,
        "fitting_role": "calibration",
        "fit_split_hash": fit_split_hash,
        "scorer_id": scorer_id,
        "shortlist_sha256": shortlist_sha256,
        "checkpoint_hash": checkpoint_hash,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    return artifact


def load_route_calibration(path, checkpoint_hash, shortlist_sha256, scorer_id="base_decision_head"):
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("route") != ROUTE:
        raise ValueError("calibration route %r cannot be applied to %s" % (artifact.get("route"), ROUTE))
    if artifact.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("route calibration belongs to a different checkpoint")
    if artifact.get("shortlist_sha256") != shortlist_sha256:
        raise ValueError("route calibration belongs to a different shortlist file")
    if artifact.get("fitting_split") in (None, "", "test"):
        raise ValueError("route calibration must name a non-test fitting split")
    if artifact.get("fitting_role", "calibration") != "calibration":
        raise ValueError("route calibration fitting_role must be calibration")
    if artifact.get("scorer_id", "base_decision_head") != scorer_id:
        raise ValueError("route calibration belongs to a different scorer")
    if not artifact.get("fit_split_hash"):
        raise ValueError("route calibration must include a fit-split hash")
    return artifact
