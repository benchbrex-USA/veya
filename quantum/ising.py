"""Ising request text and response normalization.

The prompt asks for one JSON object. The endpoint is not assumed to enforce
that. Malformed text becomes a rejected observation, not a guessed diagnosis.
"""

from __future__ import annotations

import json
from typing import Any

from quantum.config import ISING_MODEL, PROMPT_VERSION, SCHEMA_VERSION
from quantum.observation import extract_json_object, validate_observation

_SCHEMA_HINT = {
    "schema_version": SCHEMA_VERSION,
    "experiment_family": "<family slug>",
    "status": {"label": "insufficient_evidence", "evidence": ["<short text>"]},
    "plot": {"description": "<short text>", "signal_detected": None, "fit_visible": None},
    "fit": {"available": False, "reliable": None, "reason": "<short text>"},
    "parameters": {},
    "diagnosis": {"code": "unspecified", "summary": "<short text>", "evidence": []},
    "recommended_actions": [{"code": "defer", "reason": "<short text>"}],
    "source": {"provider": "nvidia_nim", "model": ISING_MODEL, "mode": "zero_shot", "request_id": ""},
}


def build_prompt(evidence: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    """Text half of the Ising request. Images are attached separately."""
    mode = "icl" if examples else "zero_shot"
    lines = [
        "You are a quantum-calibration plot specialist.",
        "Return one JSON object and no other text.",
        "Use this shape. Do not add keys.",
        json.dumps(_SCHEMA_HINT, sort_keys=True),
        "Rules:",
        "- status.label must be one of expected_behavior, suboptimal_parameters, anomalous_behavior, apparatus_issue, insufficient_evidence.",
        "- Put a number in parameters only when it is visibly exact. Otherwise use null. Do not write approximations as numbers.",
        "- Do not invent precision. Do not propose hardware commands outside recommended_actions codes.",
        "- experiment_family must be %s." % evidence["experiment_family"],
        "- source.mode must be %s." % mode,
        "Caller evidence follows. MEASURED and FIT_DERIVED lines are instrument data. Do not overwrite them.",
        json.dumps(
            {
                "experiment_family": evidence["experiment_family"],
                "measured": evidence.get("measured"),
                "fit_derived": evidence.get("fit_derived"),
                "human_verified": evidence.get("human_verified"),
                "settings": evidence.get("settings"),
            },
            sort_keys=True,
            default=_json_default,
        ),
    ]
    if examples:
        lines.append("Historical examples are context, not the current measurement.")
        lines.append(json.dumps(examples, sort_keys=True, default=_json_default))
    lines.append("Prompt version %s." % PROMPT_VERSION)
    return "\n".join(lines)


def observation_from_text(
    text: str,
    *,
    expected_family: str,
    model: str,
    mode: str,
    request_id: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = extract_json_object(text)
    if payload is None:
        return None, ["malformed_external_response"]
    return validate_observation(
        payload,
        expected_family=expected_family,
        source={
            "provider": "nvidia_nim",
            "model": model,
            "mode": mode,
            "request_id": request_id or "",
        },
    )


def _json_default(value: Any) -> str:
    return ""
