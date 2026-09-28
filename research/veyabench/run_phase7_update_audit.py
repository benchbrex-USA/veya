"""Prove the Phase-7 upper-layer training path changes real checkpoint parameters.

This is a one-step diagnostic in an isolated process.  It never writes model
weights and does not evaluate or inspect the frozen blind set.
"""
import json
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from research.veyabench.run_phase3_eval import ROOT  # noqa: E402
from research.veyabench.run_phase7_adapt import (  # noqa: E402
    batch_items,
    one_loss,
    optimizer_for,
    set_trainable,
)
from veya.release_bundle import load_release_manifest  # noqa: E402


def main():
    manifest_path = os.path.join(ROOT, "artifacts", "release", "manifest.json")
    bundle, manifest = load_release_manifest(manifest_path, device="cpu", root=ROOT)
    labels = list(bundle.labels)
    criteria = dict(zip(labels, bundle.label_texts))
    picks = [labels[:8], list(reversed(labels[:8]))]
    texts = [
        "The card transfer completed and the money was returned the next day.",
        "The cash withdrawal used the wrong exchange rate.",
    ]
    start_layer = set_trainable(bundle.agent, 1)
    batch = batch_items(bundle.agent, texts, picks, criteria)
    batch["gold"] = torch.tensor([0, 0])
    parameters = {
        name: parameter
        for name, parameter in bundle.agent.model.named_parameters()
        if parameter.requires_grad
    }
    before = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    optimizer = optimizer_for(bundle.agent, 1, encoder_lr=1e-6, head_lr=5e-6)
    optimizer.zero_grad(set_to_none=True)
    loss = one_loss(bundle.agent, batch, start_layer)
    loss.backward()
    with_grad = [parameter for parameter in parameters.values() if parameter.grad is not None]
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(with_grad, 1.0))
    optimizer.step()

    changed = 0
    max_abs_delta = 0.0
    squared_delta = 0.0
    for name, parameter in parameters.items():
        delta = parameter.detach().cpu() - before[name]
        local_max = float(delta.abs().max())
        if local_max > 0.0:
            changed += 1
        max_abs_delta = max(max_abs_delta, local_max)
        squared_delta += float((delta.double() ** 2).sum())
    report = {
        "format": "veya-phase7-update-audit-v1",
        "checkpoint_hash": manifest["base_checkpoint_hash"],
        "process_isolated": True,
        "weights_written": False,
        "start_layer": start_layer,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters.values())),
        "trainable_tensor_count": len(parameters),
        "tensors_with_nonzero_delta": changed,
        "max_abs_parameter_delta": max_abs_delta,
        "parameter_delta_l2": squared_delta ** 0.5,
        "gradient_norm_before_clip": gradient_norm,
        "loss": float(loss.detach()),
        "encoder_learning_rate": 1e-6,
        "head_learning_rate": 5e-6,
        "conclusion": "TRAINING PATH UPDATES PARAMETERS" if changed else "INVALID: NO PARAMETER UPDATE",
    }
    if changed == 0:
        raise SystemExit(json.dumps(report, indent=2))
    output = os.path.join(ROOT, "artifacts", "experiments", "phase7", "parameter_update_audit.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
