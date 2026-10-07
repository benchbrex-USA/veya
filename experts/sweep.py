"""Staged Veya training search. This module does not launch training.

The grids are the search space. A configuration is not optimal unless a
validation result beats the baseline, holds on a second seed, and does not
lose the suffix-robustness check. Blind rows are rejected.
"""

from __future__ import annotations

from typing import Any

REFERENCE = {
    "epochs": 4,
    "micro_batch": 8,
    "grad_accum": 4,
    "group_size": 4,
    "lr_encoder": 2.5e-5,
    "lr_head": 1.0e-4,
    "weight_decay": 0.01,
    "sigma_start": 0.4,
    "sigma_end": 0.1,
    "w_sph": 0.75,
    "w_rps": 1.0,
    "rl_weight": 1.0,
    "ce_weight": 1.0,
    "act_weight": 0.0,
    "grad_clip": 1.0,
    "max_len": 1024,
    "head_max_len": 256,
    "source": "veya/train.py TrainConfig defaults",
    "status": "reference_not_reconfirmed_this_change",
}

ACT_HEAD = {
    "reference_act_weight": 0.0,
    "trained_by_this_change": False,
    "reason": (
        "TrainConfig defaults act_weight to 0, and the loss term is act_weight times "
        "the action-head sum. No held-out defer label set is wired to that term, so "
        "the head stays off. Defer remains a policy decision."
    ),
}

_ENCODER_LR = (5e-6, 1e-5, 2.5e-5, 5e-5)
_HEAD_LR = (5e-5, 1e-4, 2e-4)
_EPOCHS = (2, 4, 6)
_DECAY = (0.0, 0.01, 0.05)
_GROUP = (4, 8)
_RL = (0.0, 0.5, 1.0)
_SIGMA = ((0.3, 0.05), (0.4, 0.1), (0.5, 0.1))


def staged_plan() -> dict[str, list[dict[str, Any]]]:
    """Small staged grids. Not the Cartesian product of every axis."""
    phase_a = []
    for lr_encoder in _ENCODER_LR:
        for lr_head in _HEAD_LR:
            phase_a.append({**_base(), "phase": "A", "lr_encoder": lr_encoder, "lr_head": lr_head, "epochs": 2})
    phase_b = []
    for rl_weight in _RL:
        for sigma_start, sigma_end in _SIGMA:
            phase_b.append(
                {
                    **_base(),
                    "phase": "B",
                    "rl_weight": rl_weight,
                    "sigma_start": sigma_start,
                    "sigma_end": sigma_end,
                }
            )
    phase_c = [
        {**_base(), "phase": "C", "weight_decay": decay, "group_size": group}
        for decay in _DECAY
        for group in _GROUP
    ]
    phase_d = [{**_base(), "phase": "D", "seed": seed} for seed in (17, 23)]
    return {"A": phase_a, "B": phase_b, "C": phase_c, "D": phase_d}


def cartesian_size() -> int:
    return len(_ENCODER_LR) * len(_HEAD_LR) * len(_EPOCHS) * len(_DECAY) * len(_GROUP) * len(_RL) * len(_SIGMA)


def judge(result: dict[str, Any]) -> dict[str, Any]:
    """Decide whether one recorded run may be called a validation finalist."""
    reasons = []
    if result.get("split") != "validation":
        reasons.append("not_a_validation_split")
    if result.get("split") == "blind":
        reasons.append("blind_split_rejected")
    if result.get("k") not in (None, 8):
        reasons.append("k_changed")
    baseline = result.get("baseline_recall_at_8")
    recall = result.get("recall_at_8")
    if not isinstance(baseline, (int, float)) or not isinstance(recall, (int, float)):
        reasons.append("recall_missing")
    elif recall <= baseline:
        reasons.append("recall_not_improved")
    suffix = result.get("long_suffix_recall_at_8")
    if not isinstance(suffix, (int, float)) or suffix < 0.2:
        reasons.append("robustness_not_cleared")
    seeds = result.get("seed_recalls")
    if not isinstance(seeds, list) or len(seeds) < 2:
        reasons.append("replicate_missing")
    elif isinstance(baseline, (int, float)) and any(item <= baseline for item in seeds):
        reasons.append("replicate_did_not_hold")
    if result.get("wrote_production_checkpoint") is True:
        reasons.append("production_checkpoint_write_forbidden")
    return {
        "optimal": False,
        "validation_finalist": not reasons,
        "reasons": reasons,
        "promoted_checkpoint": False,
    }


def _base() -> dict[str, Any]:
    return {
        "micro_batch": REFERENCE["micro_batch"],
        "grad_accum": REFERENCE["grad_accum"],
        "group_size": REFERENCE["group_size"],
        "lr_encoder": REFERENCE["lr_encoder"],
        "lr_head": REFERENCE["lr_head"],
        "weight_decay": REFERENCE["weight_decay"],
        "sigma_start": REFERENCE["sigma_start"],
        "sigma_end": REFERENCE["sigma_end"],
        "rl_weight": REFERENCE["rl_weight"],
        "ce_weight": 1.0,
        "act_weight": 0.0,
        "epochs": REFERENCE["epochs"],
        "k": 8,
        "output_dir": "artifacts/experiments/sweep",
        "writes_production_checkpoint": False,
    }
