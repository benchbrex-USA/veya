"""Strict QuantumObservation validation.

Invalid model text is rejected. Scientific numbers are never parsed out of
prose. A string such as "approximately 5 GHz" stays null.
"""

from __future__ import annotations

import json
from typing import Any

from quantum.config import SCHEMA_VERSION, is_slug

STATUS_LABELS = (
    "expected_behavior",
    "suboptimal_parameters",
    "anomalous_behavior",
    "apparatus_issue",
    "insufficient_evidence",
)
PARAM_SOURCES = ("visual", "metadata", "fit", "measured", "human_verified", "ising_inferred")
OBSERVATION_KEYS = {
    "schema_version",
    "experiment_family",
    "status",
    "plot",
    "fit",
    "parameters",
    "diagnosis",
    "recommended_actions",
    "source",
}
_SECRET_KEYS = {"api_key", "token", "secret", "authorization", "password", "prompt", "system"}


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the single JSON object in `text`, or None when that is ambiguous."""
    if not isinstance(text, str) or not text.strip() or len(text) > 200_000:
        return None
    stripped = text.strip()
    parsed = _loads_object(stripped)
    if isinstance(parsed, dict):
        return parsed
    objects = _balanced_objects(stripped)
    if len(objects) != 1:
        return None
    return _loads_object(objects[0])


def validate_observation(
    payload: Any,
    *,
    expected_family: str,
    source: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one model object. The returned source is ours, not the model's."""
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return None, ["observation_not_object"]
    if set(payload) - OBSERVATION_KEYS:
        return None, ["observation_unexpected_field"]
    if _has_secret_key(payload):
        return None, ["observation_secret_field"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None, ["schema_version"]
    if payload.get("experiment_family") != expected_family or not is_slug(expected_family):
        return None, ["experiment_family"]
    status = _status(payload.get("status"))
    plot = _plot(payload.get("plot"))
    fit, fit_warnings = _fit(payload.get("fit"))
    parameters, parameter_warnings = _parameters(payload.get("parameters"))
    diagnosis = _diagnosis(payload.get("diagnosis"))
    actions = _actions(payload.get("recommended_actions"))
    warnings.extend(fit_warnings)
    warnings.extend(parameter_warnings)
    if status is None or plot is None or fit is None or parameters is None or diagnosis is None or actions is None:
        return None, ["observation_invalid"]
    if not _source_ok(source):
        return None, ["source_invalid"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": expected_family,
        "status": status,
        "plot": plot,
        "fit": fit,
        "parameters": parameters,
        "diagnosis": diagnosis,
        "recommended_actions": actions,
        "source": {
            "provider": source["provider"],
            "model": source["model"],
            "mode": source["mode"],
            "request_id": source.get("request_id") or "",
        },
    }, warnings


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_number)
    except (json.JSONDecodeError, ValueError, UnicodeError):
        return None
    if isinstance(value, dict):
        return value
    return None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate")
        out[key] = value
    return out


def _bad_number(_: str) -> Any:
    raise ValueError("non-finite")


def _balanced_objects(text: str) -> list[str]:
    found: list[str] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    found.append(text[start : index + 1])
                    start = None
                    if len(found) > 1:
                        return found
    return found


def _has_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS or _has_secret_key(item):
                return True
    elif isinstance(value, list):
        return any(_has_secret_key(item) for item in value)
    return False


def _text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        return None
    if _unsafe_text(value):
        return None
    return value


def _unsafe_text(value: str) -> bool:
    lowered = value.lower()
    if "://" in lowered or lowered.startswith(("/", "\\")) or ".." in value:
        return True
    return False


def _status(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) - {"label", "evidence"}:
        return None
    if value.get("label") not in STATUS_LABELS:
        return None
    evidence = _string_list(value.get("evidence"), 8, 300)
    if evidence is None:
        return None
    return {"label": value["label"], "evidence": evidence}


def _plot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) - {"description", "signal_detected", "fit_visible"}:
        return None
    description = _text(value.get("description"), 600)
    if description is None:
        return None
    signal = _optional_bool(value.get("signal_detected"))
    visible = _optional_bool(value.get("fit_visible"))
    if signal is _BAD or visible is _BAD:
        return None
    return {"description": description, "signal_detected": signal, "fit_visible": visible}


def _fit(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(value, dict) or set(value) - {"available", "reliable", "reason"}:
        return None, []
    available = _optional_bool(value.get("available"))
    if available is _BAD or not isinstance(available, bool):
        return None, []
    reliable, warnings = _strict_optional_bool(value.get("reliable"), "fit_reliable_not_boolean")
    reason = _text(value.get("reason"), 400)
    if reason is None:
        return None, []
    return {"available": available, "reliable": reliable, "reason": reason}, warnings


def _parameters(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(value, dict) or len(value) > 16:
        return None, []
    warnings: list[str] = []
    out: dict[str, Any] = {}
    for name, raw in value.items():
        if not is_slug(name) or not isinstance(raw, dict) or set(raw) - {"value", "unit", "source"}:
            return None, []
        number, noted = _number_or_null(raw.get("value"))
        warnings.extend(noted)
        unit = raw.get("unit")
        if unit is not None:
            unit = _text(unit, 32)
            if unit is None:
                return None, []
        source = raw.get("source")
        if source not in PARAM_SOURCES:
            return None, []
        out[name] = {"value": number, "unit": unit, "source": source}
    return out, warnings


def _diagnosis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) - {"code", "summary", "evidence"}:
        return None
    code = value.get("code")
    if code != "unspecified" and not is_slug(code):
        return None
    summary = _text(value.get("summary"), 600)
    evidence = _string_list(value.get("evidence"), 8, 300)
    if summary is None or evidence is None:
        return None
    return {"code": code, "summary": summary, "evidence": evidence}


def _actions(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or len(value) > 8:
        return None
    out = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"code", "reason"}:
            return None
        if not is_slug(item.get("code")):
            return None
        reason = _text(item.get("reason"), 300)
        if reason is None:
            return None
        out.append({"code": item["code"], "reason": reason})
    return out


def _string_list(value: Any, count: int, limit: int) -> list[str] | None:
    if not isinstance(value, list) or len(value) > count:
        return None
    out = []
    for item in value:
        text = _text(item, limit)
        if text is None:
            return None
        out.append(text)
    return out


_BAD = object()


def _optional_bool(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    return _BAD


def _strict_optional_bool(value: Any, warning: str) -> tuple[bool | None, list[str]]:
    if value is None or isinstance(value, bool):
        return value, []
    return None, [warning]


def _number_or_null(value: Any) -> tuple[float | None, list[str]]:
    if value is None:
        return None, []
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, ["parameter_value_not_numeric"]
    if value != value or value in (float("inf"), float("-inf")):
        return None, ["parameter_value_not_numeric"]
    return float(value), []


def _source_ok(source: dict[str, str]) -> bool:
    required = ("provider", "model", "mode")
    if any(not isinstance(source.get(key), str) or not source[key] for key in required):
        return False
    if source["mode"] not in ("zero_shot", "icl"):
        return False
    request_id = source.get("request_id") or ""
    return isinstance(request_id, str) and len(request_id) <= 80
