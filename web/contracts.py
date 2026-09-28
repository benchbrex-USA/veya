"""Public website contract for the audited Veya HTTP service.

The gateway intentionally accepts a smaller, bounded subset of the model API.
No candidate text is rewritten or promoted to instructions.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any, Literal, TypedDict

Primitive = Literal["choice", "noul", "score", "systemone"]
MAX_CHOICE = 100
MAX_QUESTIONS = 8
MAX_STATE_BYTES = 12_000


class RequestError(ValueError):
    pass


class UpstreamContractError(ValueError):
    pass


class PublicRequest(TypedDict, total=False):
    primitive: Primitive
    state: str | dict[str, Any] | list[Any]
    instructions: str
    criteria: dict[str, str] | list[str]
    questions: dict[str, dict[str, Any]]


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise RequestError("Duplicate JSON key.")
        out[key] = value
    return out


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                RequestError("Non-finite JSON number.")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RequestError("Invalid JSON body.") from exc


def _state(value: Any) -> Any:
    if isinstance(value, str):
        valid = bool(value.strip())
    elif isinstance(value, (dict, list)):
        valid = bool(value)
    else:
        valid = False
    if not valid:
        raise RequestError("State must be a nonempty string, object, or list.")
    try:
        size = len(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RequestError("State must be valid JSON data.") from exc
    if size > MAX_STATE_BYTES:
        raise RequestError("State exceeds 12 KB.")
    return value


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RequestError(
            f"{name} must be nonempty text of at most {limit} characters."
        )
    return value


def _criteria(kind: str, value: Any) -> dict[str, str] | list[str] | None:
    if kind == "choice":
        if not isinstance(value, dict) or not 2 <= len(value) <= MAX_CHOICE:
            raise RequestError(f"Choice requires 2–{MAX_CHOICE} candidates.")
        seen = set()
        for key, description in value.items():
            _text(key, "Candidate ID", 80)
            if not isinstance(description, str) or len(description) > 240:
                raise RequestError(
                    "Candidate descriptions must be text of at most 240 characters."
                )
            normalized = _normal(key)
            if normalized in seen:
                raise RequestError(
                    "Candidate IDs must be unique after Unicode normalization."
                )
            seen.add(normalized)
        return value
    if kind == "score":
        if not isinstance(value, list) or not 2 <= len(value) <= 10:
            raise RequestError("Score requires 2–10 ordered labels.")
        for label in value:
            _text(label, "Score label", 100)
        if len({_normal(label) for label in value}) != len(value):
            raise RequestError("Score labels must be unique.")
        return value
    if value is not None:
        raise RequestError("Noul does not accept criteria in this website.")
    return None


def validate(data: Any) -> dict[str, Any]:
    """Validate and return the exact model-service body, without browser-only fields."""
    if not isinstance(data, dict) or data.get("primitive") not in (
        "choice",
        "noul",
        "score",
        "systemone",
    ):
        raise RequestError("Select a supported primitive.")
    kind = data["primitive"]
    allowed = (
        {"primitive", "state", "questions"}
        if kind == "systemone"
        else {"primitive", "state", "instructions", "criteria"}
    )
    if set(data) - allowed:
        raise RequestError("Unexpected request field.")
    state = _state(data.get("state"))
    if kind == "systemone":
        questions = data.get("questions")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
            raise RequestError(f"SystemOne requires 1–{MAX_QUESTIONS} questions.")
        out = {}
        for qid, question in questions.items():
            _text(qid, "Question ID", 64)
            if not isinstance(question, dict) or set(question) - {
                "type",
                "instructions",
                "criteria",
            }:
                raise RequestError(
                    "Each question must be an object with type, instructions, and optional criteria."
                )
            qtype = question.get("type")
            if qtype not in ("choice", "noul", "score"):
                raise RequestError(
                    "SystemOne supports Choice, Noul, and Score questions."
                )
            validated = validate(
                {
                    "primitive": qtype,
                    "state": state,
                    "instructions": question.get("instructions"),
                    "criteria": question.get("criteria"),
                }
            )
            out[qid] = {"type": qtype, "instructions": validated["instructions"]}
            if "criteria" in validated:
                out[qid]["criteria"] = validated["criteria"]
        return {"state": state, "questions": out}
    instructions = _text(data.get("instructions"), "Question", 500)
    result = {"state": state, "instructions": instructions}
    criteria = _criteria(kind, data.get("criteria"))
    if criteria is not None:
        result["criteria"] = criteria
    return result


def _probabilities(value: Any, allowed: set[str] | None = None) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise UpstreamContractError("Missing probability distribution.")
    out = {}
    for label, probability in value.items():
        if (
            not isinstance(label, str)
            or not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise UpstreamContractError("Invalid probability value.")
        if allowed is not None and label not in allowed:
            raise UpstreamContractError("Unexpected probability label.")
        out[label] = float(probability)
    if abs(sum(out.values()) - 1.0) > 0.015:
        raise UpstreamContractError("Probabilities do not sum to one.")
    return out


def _number(value: Any, name: str, low: float, high: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise UpstreamContractError(f"Invalid {name}.")
    return float(value)


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 200:
        raise UpstreamContractError(f"Invalid {name}.")
    return value


def _answer(kind: str, source: dict[str, Any], criteria: Any) -> dict[str, Any]:
    if kind == "choice":
        probabilities = _probabilities(source.get("probabilities"), set(criteria))
        answer = source.get("answer", source.get("choice"))
        if not isinstance(answer, str) or answer not in probabilities:
            raise UpstreamContractError("Choice answer is absent from distribution.")
        return {"answer": answer, "probabilities": probabilities}
    if kind == "noul":
        answer = source.get("answer", source.get("noul"))
        return {
            "answer": _number(answer, "Noul probability", 0, 1),
            "probabilities": None,
        }
    levels = len(criteria)
    answer = source.get("answer", source.get("score"))
    probabilities = _probabilities(
        source.get("probabilities"), {str(i) for i in range(levels)}
    )
    if set(probabilities) != {str(i) for i in range(levels)}:
        raise UpstreamContractError("Score levels are missing.")
    return {
        "answer": _number(answer, "Score", 0, levels - 1),
        "probabilities": probabilities,
        "legend": {str(i): label for i, label in enumerate(criteria)},
    }


def validate_response(
    kind: Primitive, payload: Any, body: dict[str, Any]
) -> dict[str, Any]:
    """Whitelist and validate model output before it reaches the browser."""
    if not isinstance(payload, dict) or payload.get("decision") not in (
        "answer",
        "defer",
    ):
        raise UpstreamContractError("Invalid decision response.")
    decision = payload["decision"]
    if "deferred" in payload and bool(payload["deferred"]) != (decision == "defer"):
        raise UpstreamContractError("Conflicting defer state.")
    result: dict[str, Any] = {
        "decision": decision,
        "deferred": decision == "defer",
        "primitive": kind,
        "model": _optional_text(payload.get("model"), "model"),
        "calibration_version": _optional_text(
            payload.get("calibration_version"), "calibration version"
        ),
    }
    reason = payload.get("defer_reason") or payload.get("reason")
    if decision == "defer":
        result.update(
            {
                "answer": None,
                "probabilities": None,
                "defer_reason": _optional_text(reason, "defer reason")
                or "model_deferred",
            }
        )
        if kind == "systemone":
            result["answers"] = {}
        return result
    result["defer_reason"] = None
    confidence = payload.get("confidence")
    if confidence is not None:
        result["confidence"] = _number(confidence, "confidence", 0, 1)
    if kind == "systemone":
        answers = payload.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
            raise UpstreamContractError("SystemOne answer IDs do not match questions.")
        result["answers"] = {}
        for qid, q in body["questions"].items():
            item = answers[qid]
            if not isinstance(item, dict) or item.get("type") != q["type"]:
                raise UpstreamContractError("SystemOne question type mismatch.")
            normalized = _answer(q["type"], item, q.get("criteria"))
            normalized["type"] = q["type"]
            if q["type"] == "choice":
                normalized["choice"] = normalized["answer"]
            elif q["type"] == "noul":
                normalized["noul"] = normalized["answer"]
            else:
                normalized["score"] = normalized["answer"]
            if item.get("confidence") is not None:
                normalized["confidence"] = _number(
                    item["confidence"], "confidence", 0, 1
                )
            normalized["route"] = _optional_text(item.get("route"), "route")
            result["answers"][qid] = normalized
        routes = payload.get("routes")
        if routes is not None:
            if not isinstance(routes, dict) or set(routes) != set(body["questions"]):
                raise UpstreamContractError("Invalid SystemOne routes.")
            result["routes"] = {
                qid: _optional_text(routes[qid], "route") for qid in body["questions"]
            }
    else:
        result.update(_answer(kind, payload, body.get("criteria")))
        result["route"] = _optional_text(payload.get("route"), "route")
        if kind == "choice":
            shortlist = payload.get("shortlist")
            if shortlist is not None:
                if (
                    not isinstance(shortlist, list)
                    or len(shortlist) > 8
                    or any(
                        not isinstance(label, str) or label not in body["criteria"]
                        for label in shortlist
                    )
                ):
                    raise UpstreamContractError("Invalid shortlist metadata.")
                result["shortlist"] = shortlist
            result["candidate_count"] = len(body["criteria"])
            result["calibration"] = _optional_text(
                payload.get("calibration"), "calibration"
            )
            result["reranker_id"] = _optional_text(
                payload.get("reranker_id"), "reranker"
            )
            if payload.get("automation") is not None:
                result["automation"] = _optional_text(
                    payload["automation"], "automation"
                )
    usage = payload.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {
            key: int(usage[key])
            for key in ("input_tokens", "output_tokens")
            if isinstance(usage.get(key), int) and 0 <= usage[key] <= 1_000_000
        }
    return result
