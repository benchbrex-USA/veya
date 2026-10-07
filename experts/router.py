"""Deterministic capability router.

Constraints run before any learned preference. A larger model is not selected
because it exists. Ising is not replaced by a general vision model.
"""

from __future__ import annotations

from typing import Any

from experts.registry import (
    DEEPSEEK,
    EXPENSIVE,
    GLM,
    GPT_OSS,
    ISING,
    KIMI,
    LIGHTNING,
    SAFETY,
    ULTRA,
    VEYA,
    profile,
)

TIERS = ("fast", "standard", "deep", "max")
TASKS = (
    "bounded_decision",
    "general_text",
    "local_reasoning",
    "multimodal",
    "quantum_calibration",
    "safety",
    "planning",
)


class RouteError(ValueError):
    pass


def route(request: dict[str, Any], enabled: set[str] | None = None) -> dict[str, Any]:
    """Return the selected model and a capability-compatible fallback chain."""
    task = request.get("task")
    if task not in TASKS:
        raise RouteError("unknown task")
    if "model" in request or "base_url" in request:
        raise RouteError("caller cannot select a model or url")
    tier = request.get("tier") or "standard"
    if tier not in TIERS:
        raise RouteError("unknown tier")
    modality = request.get("modality") or "text"
    if modality not in {"text", "image", "multimodal"}:
        raise RouteError("unknown modality")
    privacy = request.get("privacy") or "hosted"
    if privacy not in {"hosted", "private"}:
        raise RouteError("unknown privacy")
    budget = request.get("budget") or "standard"
    if budget not in {"low", "standard", "frontier"}:
        raise RouteError("unknown budget")
    allowed = enabled if enabled is not None else set()
    primary, chain, reason = _choose(task, tier, modality, privacy, budget)
    selected, skipped = _first_available(primary, chain, allowed, task)
    spec = profile(selected) if selected != "defer" else None
    effort = _effort(selected, tier, task)
    return {
        "selected": selected,
        "callable": selected == VEYA or selected == "defer" or selected in allowed,
        "role": None if spec is None else spec["role"],
        "profile": None if spec is None else _with_effort(spec["profile"], effort),
        "fallback": [item for item in chain if item != selected],
        "skipped_unavailable": skipped,
        "reason": reason,
        "tier": tier,
        "invokes_external": selected not in {VEYA, "defer"},
        "veya_remains_policy_gate": True,
        "hardware_control": False,
    }


def _choose(task: str, tier: str, modality: str, privacy: str, budget: str) -> tuple[str, list[str], str]:
    if task == "safety":
        return SAFETY, ["defer"], "safety classification is not a truth or calibration judgment"
    if task == "quantum_calibration" or (modality == "image" and task == "quantum_calibration"):
        return ISING, ["defer"], "quantum plots stay on the calibration specialist"
    if task == "bounded_decision":
        return VEYA, [], "bounded decision stays on Veya Core"
    if privacy == "private" and modality == "text":
        return GPT_OSS, ["defer"], "private text reasoning stays on the local specialist"
    if modality in {"image", "multimodal"} or task == "multimodal":
        if tier == "max" and budget == "frontier":
            return KIMI, [GLM, DEEPSEEK, "defer"], "max multimodal tier with a frontier budget"
        if tier == "deep":
            return GLM, [DEEPSEEK, "defer"], "deep multimodal tier stops before the expensive escalation"
        return DEEPSEEK, [GLM, "defer"], "fast multimodal default"
    if task == "planning" and tier in {"deep", "max"} and budget == "frontier":
        return ULTRA, [LIGHTNING, GPT_OSS, "defer"], "frontier text planning"
    if task == "local_reasoning" or budget == "low":
        return GPT_OSS if task == "local_reasoning" else LIGHTNING, [GPT_OSS, "defer"], "cost-sensitive text path"
    return LIGHTNING, [GPT_OSS, "defer"], "default text worker"


def _first_available(primary: str, chain: list[str], enabled: set[str], task: str) -> tuple[str, list[str]]:
    order = [primary, *chain]
    skipped = []
    for model_id in order:
        if model_id in {VEYA, "defer"} or model_id in enabled:
            if model_id in EXPENSIVE and model_id != primary:
                continue
            if task == "quantum_calibration" and model_id not in {ISING, "defer"}:
                skipped.append(model_id)
                continue
            return model_id, skipped
        skipped.append(model_id)
    return "defer", skipped


def _effort(model_id: str, tier: str, task: str) -> str | None:
    if model_id == GLM:
        return {"fast": "low", "standard": "low", "deep": "high", "max": "high"}[tier]
    if model_id == KIMI:
        return "max" if tier == "max" else "high"
    if model_id == ULTRA:
        return "on" if tier in {"deep", "max"} else "off"
    if model_id == LIGHTNING:
        return "low" if tier == "fast" else "standard"
    if task == "safety":
        return None
    return None


def _with_effort(profile_body: dict[str, Any], effort: str | None) -> dict[str, Any]:
    copied = dict(profile_body)
    if effort is not None:
        copied["reasoning_effort"] = effort
    if str(copied.get("profile_name", "")).startswith("kimi"):
        copied["top_p"] = 1.0 if effort == "max" else 0.95
    return copied
