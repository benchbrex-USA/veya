"""Allowlisted model roles and starting inference profiles.

Sampling numbers are provider starting points. None of them is a BenchBrex
optimum. `benchbrex_tuned` stays false until a held-out sweep says otherwise.
"""

from __future__ import annotations

from typing import Any

VEYA = "veya-core"
LIGHTNING = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEEPSEEK = "deepseek-ai/deepseek-v4.1-flash"
GLM = "z-ai/glm-5.3-flash"
GPT_OSS = "openai/gpt-oss-20b"
KIMI = "moonshotai/kimi-k3"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
ISING = "nvidia/ising-calibration-1.5-31b"
SAFETY = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

PROFILE_VERSION = "experts-profile-v1"
EXPENSIVE = {KIMI, ULTRA}

# context_tokens is recorded only where a provider card states it.
# None means this registry does not claim a number.
_PROFILES: dict[str, dict[str, Any]] = {
    VEYA: {
        "role": "decision",
        "modalities": ["text"],
        "tasks": ["bounded_decision", "routing", "defer"],
        "context_tokens": 512,
        "latency_class": "low",
        "cost_class": "low",
        "privacy_class": "local",
        "image": False,
        "tools": False,
        "reasoning": False,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "veya-bounded-v1",
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "reasoning_effort": None,
            "timeout_s": 30,
            "retries": 0,
            "source": "veya_decision_head",
            "benchbrex_tuned": False,
        },
    },
    LIGHTNING: {
        "role": "default_text_worker",
        "modalities": ["text"],
        "tasks": ["general_text", "tools", "coding"],
        "context_tokens": 1_000_000,
        "latency_class": "standard",
        "cost_class": "standard",
        "privacy_class": "hosted",
        "image": False,
        "tools": True,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "lightning-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "reasoning_effort": "dynamic",
            "timeout_s": 60,
            "retries": 1,
            "source": "nvidia_provider_baseline",
            "benchbrex_tuned": False,
        },
    },
    DEEPSEEK: {
        "role": "fast_multimodal",
        "modalities": ["text", "image"],
        "tasks": ["multimodal", "coding"],
        "context_tokens": 1_000_000,
        "latency_class": "standard",
        "cost_class": "standard",
        "privacy_class": "hosted",
        "image": True,
        "tools": True,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "deepseek-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "reasoning_effort": "dynamic",
            "timeout_s": 60,
            "retries": 1,
            "source": "nvidia_provider_baseline",
            "benchbrex_tuned": False,
        },
    },
    GLM: {
        "role": "multimodal_reasoning",
        "modalities": ["text", "image"],
        "tasks": ["multimodal", "tools"],
        "context_tokens": None,
        "latency_class": "standard",
        "cost_class": "standard",
        "privacy_class": "hosted",
        "image": True,
        "tools": True,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "glm-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "reasoning_effort": "selected",
            "timeout_s": 90,
            "retries": 1,
            "source": "nvidia_provider_baseline",
            "benchbrex_tuned": False,
        },
    },
    GPT_OSS: {
        "role": "private_reasoning",
        "modalities": ["text"],
        "tasks": ["local_reasoning", "general_text"],
        "context_tokens": None,
        "latency_class": "low",
        "cost_class": "low",
        "privacy_class": "local",
        "image": False,
        "tools": False,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "gpt-oss-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "reasoning_effort": None,
            "timeout_s": 60,
            "retries": 1,
            "source": "provider_starting_point",
            "benchbrex_tuned": False,
        },
    },
    KIMI: {
        "role": "multimodal_escalation",
        "modalities": ["text", "image"],
        "tasks": ["multimodal", "long_horizon"],
        "context_tokens": None,
        "latency_class": "high",
        "cost_class": "high",
        "privacy_class": "hosted",
        "image": True,
        "tools": True,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "kimi-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_p_agentic": 1.0,
            "max_tokens": 8192,
            "reasoning_effort": "high_only_when_justified",
            "timeout_s": 120,
            "retries": 0,
            "source": "provider_starting_point",
            "benchbrex_tuned": False,
        },
    },
    ULTRA: {
        "role": "frontier_text",
        "modalities": ["text"],
        "tasks": ["planning", "long_context"],
        "context_tokens": 1_000_000,
        "latency_class": "high",
        "cost_class": "high",
        "privacy_class": "hosted",
        "image": False,
        "tools": True,
        "reasoning": True,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "ultra-provider-baseline-v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 8192,
            "reasoning_effort": "dynamic",
            "timeout_s": 120,
            "retries": 0,
            "source": "nvidia_provider_baseline",
            "benchbrex_tuned": False,
        },
    },
    ISING: {
        "role": "quantum_specialist",
        "modalities": ["text", "image"],
        "tasks": ["quantum_calibration"],
        "context_tokens": 128_000,
        "latency_class": "high",
        "cost_class": "high",
        "privacy_class": "hosted",
        "image": True,
        "tools": False,
        "reasoning": False,
        "scientific_judge": False,
        "safety_only": False,
        "profile": {
            "profile_name": "ising-provider-baseline-v1",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 8192,
            "icl_max_tokens": 32767,
            "reasoning_effort": None,
            "timeout_s": 120,
            "retries": 0,
            "source": "nvidia_provider_baseline",
            "benchbrex_tuned": False,
        },
    },
    SAFETY: {
        "role": "safety",
        "modalities": ["text"],
        "tasks": ["safety"],
        "context_tokens": None,
        "latency_class": "low",
        "cost_class": "low",
        "privacy_class": "hosted",
        "image": False,
        "tools": False,
        "reasoning": False,
        "scientific_judge": False,
        "safety_only": True,
        "profile": {
            "profile_name": "safety-guard-v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "reasoning_effort": None,
            "timeout_s": 20,
            "retries": 0,
            "source": "safety_taxonomy",
            "benchbrex_tuned": False,
        },
    },
}


def ids() -> tuple[str, ...]:
    return tuple(_PROFILES)


def profile(model_id: str) -> dict[str, Any] | None:
    spec = _PROFILES.get(model_id)
    if spec is None:
        return None
    return {"model_id": model_id, "profile_version": PROFILE_VERSION, **spec}


def public_registry() -> list[dict[str, Any]]:
    rows = []
    for model_id in _PROFILES:
        item = profile(model_id)
        if item is not None:
            rows.append(item)
    return rows
