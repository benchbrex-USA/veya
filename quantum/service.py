"""Quantum analyze and decide pipeline.

Analyze is external perception. Decide asks Veya. The policy gate does not
execute hardware, and a missing specialist does not become the action authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any

from quantum.config import ISING_MODEL, SAFETY_MODEL, QuantumConfig, is_slug
from quantum.decision import (
    build_questions,
    build_state,
    family_actions,
    filter_actions,
    interpret_veya,
    policy_gate,
)
from quantum.examples import ExampleIndex, ObservationCache, cache_key
from quantum.images import MAX_IMAGES, MAX_TOTAL_IMAGE_BYTES, decode_image
from quantum.ising import build_prompt, observation_from_text
from quantum.provider import ExternalModelProvider, ProviderError
from quantum.research import disagreement, run_critics, safety_prompt, safety_verdict

_PROJECT = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")
_FORBIDDEN = {
    "model",
    "base_url",
    "api_key",
    "token",
    "authorization",
    "prompt",
    "system",
    "messages",
    "upstream",
    "url",
}
_ALLOWED = {
    "experiment_family",
    "project_id",
    "device_type",
    "images",
    "measurements",
    "fit",
    "settings",
    "instrument",
    "notes",
    "human_verified",
    "example_ids",
    "retrieve_examples",
    "mode",
    "questions",
}


_CACHE: ObservationCache | None = None
_CACHE_LOCK = threading.Lock()


def shared_cache(config: QuantumConfig) -> ObservationCache | None:
    """Process-local cache. Disabled unless the operator turns it on."""
    global _CACHE
    if not config.cache_enabled:
        return None
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE.max_entries != config.cache_entries:
            _CACHE = ObservationCache(config.cache_entries)
        return _CACHE


class RequestError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def handle_request(
    kind: str,
    data: Any,
    *,
    config: QuantumConfig,
    provider: ExternalModelProvider,
    veya: Any,
    store: ExampleIndex | None = None,
    cache: ObservationCache | None = None,
    request_id: str,
) -> dict[str, Any]:
    if kind not in {"analyze", "decide"}:
        raise RequestError("not_found", "Not found.")
    view = parse_request(data)
    if not config.enabled:
        body = _body(
            request_id,
            enabled=False,
            mode=view["mode"],
            deferred=True,
            defer_reason="quantum_disabled",
            observation=None,
            decision=None,
            warnings=[],
            policy=policy_gate("normal", "quantum_disabled"),
            disagreement_report=None,
            external_model_score=None,
            family=view["experiment_family"],
        )
        return {"http_status": 200, "body": body, "telemetry": _telemetry(body, config)}
    if view["mode"] == "research_ensemble" and not config.critic_models:
        raise RequestError("invalid_input", "Research ensemble mode requires configured critic models.")
    if (view["example_ids"] or view["retrieve_examples"]) and config.icl_k == 0:
        raise RequestError("invalid_input", "In-context examples are disabled.")
    if (view["example_ids"] or view["retrieve_examples"]) and not view["project_id"]:
        raise RequestError("invalid_input", "A project id is required to use examples.")

    warnings: list[str] = []
    if config.safety_enabled and view["free_text"]:
        try:
            result = provider.analyze_text(SAFETY_MODEL, safety_prompt("\n".join(view["free_text"])))
        except ProviderError:
            return _done(
                request_id, config, view, True, "safety_unavailable", None, None, warnings, None, None, "normal"
            )
        verdict = safety_verdict(result.text)
        if verdict is None:
            return _done(
                request_id, config, view, True, "safety_unavailable", None, None, warnings, None, None, "normal"
            )
        if verdict:
            return _done(
                request_id, config, view, True, "safety_flagged", None, None, warnings, None, None, "high"
            )

    examples: list[dict[str, Any]] = []
    if store is not None and config.icl_k and (view["example_ids"] or view["retrieve_examples"]):
        examples, missing = store.resolve(
            project_id=view["project_id"],
            example_ids=view["example_ids"],
            family=view["experiment_family"],
            device_type=view["device_type"],
            k=config.icl_k,
            retrieve=view["retrieve_examples"],
        )
        if missing:
            warnings.append("example_not_found")
    mode = "icl" if examples else "zero_shot"
    evidence = _evidence(view)
    key = cache_key(
        model=config.ising_model,
        project_id=view["project_id"],
        image_hashes=view["image_hashes"],
        evidence=evidence,
        example_ids=[str(item["id"]) for item in examples],
        mode=mode,
    )
    observation = cache.get(key) if cache is not None else None
    external_score = None
    external_error = None
    provider_request_id = None
    if observation is None:
        try:
            completion = provider.analyze_multimodal(
                config.ising_model or ISING_MODEL,
                build_prompt(evidence, examples),
                view["images"],
                max_tokens=config.max_tokens,
            )
        except ProviderError as exc:
            external_error = _public_external_code(exc.code)
        else:
            provider_request_id = completion.request_id
            external_score = completion.external_model_score
            observation, parse_warnings = observation_from_text(
                completion.text,
                expected_family=view["experiment_family"],
                model=config.ising_model,
                mode=mode,
                request_id=provider_request_id,
            )
            warnings.extend(parse_warnings)
            if observation is None:
                external_error = "invalid_observation"
            elif cache is not None:
                cache.put(key, observation)
    if external_error and kind == "analyze":
        return _done(
            request_id,
            config,
            view,
            True,
            "external_model_unavailable" if external_error != "invalid_observation" else "invalid_observation",
            None,
            None,
            warnings,
            None,
            external_score,
            "normal",
            external_error=external_error,
            inference_mode=mode,
        )

    critiques: list[dict[str, Any]] = []
    if view["mode"] == "research_ensemble" and observation is not None:
        critiques = run_critics(provider, config.critic_models, _critic_text(evidence, observation))
    family, allowed = family_actions(view["experiment_family"])
    if family == "generic" and view["experiment_family"] != "generic":
        warnings.append("generic_action_set")
    kept, action_warnings = filter_actions(observation, allowed)
    warnings.extend(action_warnings)
    if kind == "analyze":
        critic_disagrees, critic_split = _critic_flags(critiques)
        report = disagreement(
            zero_shot_code=_code(observation) if mode == "zero_shot" else None,
            icl_code=_code(observation) if mode == "icl" else None,
            veya_action=None,
            ising_action=kept[0]["code"] if kept else None,
            critic_disagrees=critic_disagrees,
            critic_split=critic_split,
            human_correction=False,
        )
        return _done(
            request_id,
            config,
            view,
            observation is None,
            None if observation else "invalid_observation",
            observation,
            None,
            warnings,
            report,
            external_score,
            str(report["review_priority"]),
            inference_mode=mode,
            critiques=critiques,
            examples=examples,
        )

    state = build_state(evidence, observation, kept)
    questions = build_questions(allowed)
    transport = veya.ask(state, questions)
    decision = None
    veya_error = None
    if not transport.get("ok"):
        veya_error = "veya_unavailable"
    else:
        decision = interpret_veya(transport["payload"])
        if decision is None:
            veya_error = "veya_unavailable"
    veya_action = None
    if decision and not decision["deferred"]:
        next_action = decision["answers"].get("next_action") or {}
        label = next_action.get("label") if isinstance(next_action, dict) else None
        veya_action = label if isinstance(label, str) else None
    critic_disagrees, critic_split = _critic_flags(critiques)
    report = disagreement(
        zero_shot_code=_code(observation) if mode == "zero_shot" else None,
        icl_code=_code(observation) if mode == "icl" else None,
        veya_action=veya_action,
        ising_action=kept[0]["code"] if kept else None,
        critic_disagrees=critic_disagrees,
        critic_split=critic_split,
        human_correction=False,
    )
    if veya_error:
        reason = "veya_unavailable"
    elif external_error == "invalid_observation":
        reason = "invalid_observation"
    elif external_error:
        reason = "external_model_unavailable"
    elif decision and decision.get("defer_reason") == "unsupported_language":
        reason = "unsupported_language"
    else:
        reason = "automation_not_validated"
    return _done(
        request_id,
        config,
        view,
        True,
        reason,
        observation,
        decision,
        warnings,
        report,
        external_score,
        str(report["review_priority"]),
        external_error=external_error,
        inference_mode=mode,
        critiques=critiques,
        examples=examples,
        veya_error=veya_error,
    )


def parse_request(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RequestError("invalid_input", "Request body must be a JSON object.")
    if set(data) & _FORBIDDEN:
        raise RequestError("invalid_input", "The request names a field the gateway does not accept.")
    if set(data) - _ALLOWED:
        raise RequestError("invalid_input", "Unexpected request field.")
    family = data.get("experiment_family")
    if not is_slug(family):
        raise RequestError("invalid_input", "Experiment family must be a short slug.")
    project = data.get("project_id", "")
    if project and (not isinstance(project, str) or _PROJECT.match(project) is None):
        raise RequestError("invalid_input", "Project id is not an accepted identifier.")
    device = data.get("device_type")
    if device is not None and not is_slug(device):
        raise RequestError("invalid_input", "Device type must be a short slug.")
    mode = data.get("mode") or "minimal"
    if mode == "teacher_factory":
        raise RequestError("invalid_input", "Teacher factory is an offline research path.")
    if mode not in {"minimal", "research_ensemble"}:
        raise RequestError("invalid_input", "Unknown operating mode.")
    questions = data.get("questions")
    if questions is not None:
        if not isinstance(questions, list) or len(questions) > 5:
            raise RequestError("invalid_input", "Questions are not an accepted list.")
        for item in questions:
            if item not in {
                "experiment_status",
                "fit_reliable",
                "severity",
                "next_action",
                "automation_eligible",
            }:
                raise RequestError("invalid_input", "Unknown decision question.")
    images, hashes = _images(data.get("images"))
    measured = _measured(data.get("measurements"))
    fit = _fit(data.get("fit"))
    settings = _object(data.get("settings"), "Settings")
    instrument = _object(data.get("instrument"), "Instrument")
    human = _object(data.get("human_verified"), "Human verified")
    notes = data.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500 or "\x00" in notes:
        raise RequestError("invalid_input", "Notes must be text of at most 500 characters.")
    example_ids = data.get("example_ids") or []
    if not isinstance(example_ids, list) or len(example_ids) > 8:
        raise RequestError("invalid_input", "Too many example ids.")
    for example_id in example_ids:
        if not isinstance(example_id, str) or not _PROJECT.match(example_id):
            raise RequestError("invalid_input", "Example id is not an accepted identifier.")
    retrieve = data.get("retrieve_examples", False)
    if not isinstance(retrieve, bool):
        raise RequestError("invalid_input", "retrieve_examples must be a boolean.")
    _size_guard(measured, fit, settings, instrument, human)
    _no_secrets(measured)
    _no_secrets(fit)
    _no_secrets(settings)
    _no_secrets(instrument)
    _no_secrets(human)
    return {
        "experiment_family": family,
        "project_id": project,
        "device_type": device,
        "mode": mode,
        "images": images,
        "image_hashes": hashes,
        "measured": measured,
        "fit": fit,
        "settings": settings,
        "instrument": instrument,
        "human_verified": human,
        "notes": notes,
        "example_ids": example_ids,
        "retrieve_examples": retrieve,
        "free_text": [notes] if notes.strip() else [],
    }


def _images(value: Any) -> tuple[list[tuple[str, str]], list[str]]:
    if value is None:
        return [], []
    if not isinstance(value, list) or len(value) > MAX_IMAGES:
        raise RequestError("invalid_input", "Too many images.")
    images = []
    hashes = []
    total = 0
    for item in value:
        if not isinstance(item, dict) or set(item) - {"mime", "data_base64"}:
            raise RequestError("invalid_input", "Image object is not accepted.")
        mime = item.get("mime")
        encoded = item.get("data_base64")
        if not isinstance(mime, str) or not isinstance(encoded, str):
            raise RequestError("invalid_input", "Image object is not accepted.")
        try:
            raw, digest = decode_image(mime, encoded)
        except ValueError:
            raise RequestError("invalid_input", "Image is outside the accepted limits.") from None
        total += len(raw)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise RequestError("invalid_input", "Image is outside the accepted limits.")
        images.append((mime, encoded))
        hashes.append(digest)
    return images, hashes


def _measured(value: Any) -> dict[str, Any]:
    if value is None:
        return {"series": [], "metrics": {}}
    if not isinstance(value, dict) or set(value) - {"series", "metrics"}:
        raise RequestError("invalid_input", "Measurements are not an accepted object.")
    series = value.get("series") or []
    metrics = value.get("metrics") or {}
    if not isinstance(series, list) or len(series) > 4 or not isinstance(metrics, dict) or len(metrics) > 16:
        raise RequestError("invalid_input", "Measurements are outside the accepted limits.")
    cleaned = []
    for item in series:
        if not isinstance(item, dict) or set(item) - {"name", "x", "y", "x_unit", "y_unit"}:
            raise RequestError("invalid_input", "A measurement series is not accepted.")
        if not is_slug(item.get("name")):
            raise RequestError("invalid_input", "A measurement series is not accepted.")
        cleaned.append(
            {
                "name": item["name"],
                "x": _series(item.get("x")),
                "y": _series(item.get("y")),
                "x_unit": _unit(item.get("x_unit")),
                "y_unit": _unit(item.get("y_unit")),
            }
        )
    return {"series": cleaned, "metrics": _metrics(metrics)}


def _series(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) > 256:
        raise RequestError("invalid_input", "A measurement series is outside the accepted limits.")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RequestError("invalid_input", "A measurement series must be numeric.")
        out.append(float(item))
    return out


def _metrics(value: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, item in value.items():
        if not is_slug(key) or isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RequestError("invalid_input", "A measurement metric must be numeric.")
        out[key] = float(item)
    return out


def _unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 32 or not value.isascii():
        raise RequestError("invalid_input", "A unit is not accepted.")
    return value


def _fit(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"available", "r2", "parameters", "residuals_summary"}:
        raise RequestError("invalid_input", "Fit metadata is not an accepted object.")
    available = value.get("available")
    if not isinstance(available, bool):
        raise RequestError("invalid_input", "Fit metadata is not an accepted object.")
    r2 = value.get("r2")
    if r2 is not None and (isinstance(r2, bool) or not isinstance(r2, (int, float))):
        raise RequestError("invalid_input", "Fit metric must be numeric.")
    summary = value.get("residuals_summary", "")
    if not isinstance(summary, str) or len(summary) > 300:
        raise RequestError("invalid_input", "Fit summary is too long.")
    parameters = value.get("parameters") or {}
    if not isinstance(parameters, dict) or len(parameters) > 16:
        raise RequestError("invalid_input", "Fit parameters are outside the accepted limits.")
    cleaned = {}
    for name, raw in parameters.items():
        if not is_slug(name) or not isinstance(raw, dict) or set(raw) - {"value", "unit"}:
            raise RequestError("invalid_input", "A fit parameter is not accepted.")
        number = raw.get("value")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise RequestError("invalid_input", "A fit parameter must be numeric.")
        cleaned[name] = {"value": float(number), "unit": _unit(raw.get("unit"))}
    return {
        "available": available,
        "r2": None if r2 is None else float(r2),
        "parameters": cleaned,
        "residuals_summary": summary,
    }


def _object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RequestError("invalid_input", "%s must be an object." % label)
    return value


def _no_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"api_key", "token", "secret", "authorization", "password"}:
                raise RequestError("invalid_input", "The request names a field the gateway does not accept.")
            _no_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _no_secrets(item)


def _size_guard(*values: Any) -> None:
    try:
        raw = json.dumps(values, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise RequestError("invalid_input", "Metadata must be JSON data.") from None
    if len(raw) > 32_000:
        raise RequestError("invalid_input", "Metadata exceeds 32 KB.")


def _evidence(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_family": view["experiment_family"],
        "measured": view["measured"],
        "fit_derived": view["fit"],
        "human_verified": view["human_verified"],
        "settings": view["settings"],
        "instrument": view["instrument"],
    }


def _public_external_code(code: str) -> str:
    if code in {"timeout", "rate_limited", "overloaded", "upstream_error", "response_too_large", "unavailable", "invalid_response"}:
        return code
    return "unavailable"


def _code(observation: dict[str, Any] | None) -> str | None:
    if observation is None:
        return None
    code = observation["diagnosis"]["code"]
    return code if isinstance(code, str) else None


def _critic_flags(critiques: list[dict[str, Any]]) -> tuple[bool, bool]:
    verdicts = []
    for note in critiques:
        critique = note.get("critique") or {}
        agrees = critique.get("agrees_with_diagnosis")
        if isinstance(agrees, bool):
            verdicts.append(agrees)
    return any(item is False for item in verdicts), len(set(verdicts)) > 1


def _critic_text(evidence: dict[str, Any], observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "measured": evidence.get("measured"),
            "fit_derived": evidence.get("fit_derived"),
            "ising_diagnosis": observation["diagnosis"]["code"],
            "ising_status": observation["status"]["label"],
        },
        sort_keys=True,
    )


def _done(
    request_id: str,
    config: QuantumConfig,
    view: dict[str, Any],
    deferred: bool,
    defer_reason: str | None,
    observation: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    warnings: list[str],
    report: dict[str, Any] | None,
    external_score: dict[str, object] | None,
    priority: str,
    *,
    external_error: str | None = None,
    inference_mode: str = "zero_shot",
    critiques: list[dict[str, Any]] | None = None,
    examples: list[dict[str, Any]] | None = None,
    veya_error: str | None = None,
) -> dict[str, Any]:
    policy_reason = defer_reason or "automation_not_validated"
    body = _body(
        request_id,
        enabled=True,
        mode=view["mode"],
        deferred=deferred,
        defer_reason=defer_reason,
        observation=observation,
        decision=decision,
        warnings=warnings,
        policy=policy_gate(priority, policy_reason),
        disagreement_report=report,
        external_model_score=external_score,
        family=view["experiment_family"],
        external_error=external_error,
        inference_mode=inference_mode,
        critiques=critiques or [],
        example_ids=[str(item["id"]) for item in (examples or [])],
        veya_error=veya_error,
        measured=view["measured"],
        fit=view["fit"],
        human=view["human_verified"],
    )
    return {"http_status": 200, "body": body, "telemetry": _telemetry(body, config)}


def _body(
    request_id: str,
    *,
    enabled: bool,
    mode: str,
    deferred: bool,
    defer_reason: str | None,
    observation: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    warnings: list[str],
    policy: dict[str, object],
    disagreement_report: dict[str, Any] | None,
    external_model_score: dict[str, object] | None,
    family: str,
    external_error: str | None = None,
    inference_mode: str = "zero_shot",
    critiques: list[dict[str, Any]] | None = None,
    example_ids: list[str] | None = None,
    veya_error: str | None = None,
    measured: dict[str, Any] | None = None,
    fit: dict[str, Any] | None = None,
    human: dict[str, Any] | None = None,
) -> dict[str, Any]:
    veya_probability: dict[str, float | None] = {}
    if decision:
        veya_probability = decision.get("veya_probability") or {}
    return {
        "request_id": request_id,
        "schema_version": "1",
        "enabled": enabled,
        "mode": mode,
        "deferred": deferred,
        "defer_reason": defer_reason,
        "experiment_family": family,
        "evidence": {
            "measured": measured or {},
            "fit_derived": fit,
            "ising_inferred": observation,
            "human_verified": human or {},
        },
        "observation": observation,
        "decision": decision,
        "validation_warnings": warnings,
        "provenance": {
            "external_provider": "nvidia_nim" if enabled else None,
            "external_model": ISING_MODEL if enabled else None,
            "external_mode": inference_mode if enabled else None,
            "example_ids": example_ids or [],
            "veya_route": "systemone" if decision is not None or veya_error else None,
        },
        "scores": {
            "external_model_score": external_model_score,
            "veya_probability": veya_probability,
            "policy_decision": policy["decision"],
        },
        "policy": policy,
        "disagreement": disagreement_report,
        "critiques": critiques or [],
        "failures": [item for item in (external_error, veya_error) if item],
    }


def _telemetry(body: dict[str, Any], config: QuantumConfig) -> dict[str, object]:
    project_hash = hashlib.sha256(str(body.get("experiment_family", "")).encode("utf-8")).hexdigest()[:16]
    disagreement_report = body.get("disagreement") or {}
    flags = disagreement_report.get("flags") if isinstance(disagreement_report, dict) else None
    return {
        "experiment_family": body.get("experiment_family"),
        "family_hash": project_hash,
        "provider": (body.get("provenance") or {}).get("external_provider"),
        "model": (body.get("provenance") or {}).get("external_model"),
        "mode": (body.get("provenance") or {}).get("external_mode"),
        "operating_mode": body.get("mode"),
        "policy": config.policy,
        "validation": "disabled" if not body.get("enabled") else ("ok" if body.get("observation") else "rejected"),
        "deferred": body.get("deferred"),
        "defer_reason": body.get("defer_reason"),
        "disagreement": bool(flags),
        "veya_route": (body.get("provenance") or {}).get("veya_route"),
    }
