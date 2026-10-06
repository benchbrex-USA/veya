"""Translate validated quantum evidence into Veya questions and a defer policy.

Veya remains the decision model. This module never treats Ising prose as a
calibrated probability and never authorizes hardware execution.
"""

from __future__ import annotations

import json
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

from quantum.config import is_slug
from quantum.observation import STATUS_LABELS

FAMILIES: dict[str, tuple[str, ...]] = {
    "resonator_spectroscopy": (
        "accept",
        "repeat_measurement",
        "widen_sweep",
        "narrow_sweep",
        "refit",
        "collect_more_data",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
    "qubit_spectroscopy": (
        "accept",
        "repeat_measurement",
        "widen_sweep",
        "narrow_sweep",
        "refit",
        "collect_more_data",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
    "rabi": (
        "accept",
        "repeat_measurement",
        "change_drive_amplitude",
        "change_drive_frequency",
        "refit",
        "collect_more_data",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
    "ramsey": (
        "accept",
        "repeat_measurement",
        "change_drive_frequency",
        "collect_more_data",
        "refit",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
    "readout_calibration": (
        "accept",
        "repeat_measurement",
        "collect_more_data",
        "refit",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
    "generic": (
        "repeat_measurement",
        "collect_more_data",
        "inspect_hardware",
        "escalate_to_operator",
        "defer",
    ),
}

SEVERITY = (
    "nominal",
    "minor",
    "adjustment required",
    "serious anomaly",
    "likely apparatus or system issue",
)

QUESTION_IDS = (
    "experiment_status",
    "fit_reliable",
    "severity",
    "next_action",
    "automation_eligible",
)


def family_actions(family: str) -> tuple[str, tuple[str, ...]]:
    if family in FAMILIES:
        return family, FAMILIES[family]
    return "generic", FAMILIES["generic"]


def filter_actions(observation: dict[str, Any] | None, allowed: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    if observation is None:
        return [], []
    kept = []
    warnings = []
    for action in observation["recommended_actions"]:
        if action["code"] in allowed:
            kept.append(action)
        else:
            warnings.append("unsupported_action")
    return kept, warnings


def build_state(evidence: dict[str, Any], observation: dict[str, Any] | None, actions: list[dict[str, str]]) -> str:
    """English decision state. Provenance labels stay distinct. Images are omitted."""
    lines = [
        "This is the evidence for a bounded calibration decision.",
        "The measured values and the fit values and the specialist notes are separate.",
        "MEASURED data came from the instrument.",
        "FIT_DERIVED data came from the supplied fit.",
        "ISING_INFERRED data came from an external specialist and is not a calibrated probability.",
        "HUMAN_VERIFIED data was supplied as already reviewed.",
        "Experiment family is %s." % evidence["experiment_family"],
        "MEASURED " + _compact(evidence.get("measured")),
        "FIT_DERIVED " + _compact(evidence.get("fit_derived")),
        "HUMAN_VERIFIED " + _compact(evidence.get("human_verified")),
    ]
    if observation is None:
        lines.append("ISING_INFERRED observation is absent. Do not invent a diagnosis.")
    else:
        lines.append("ISING_INFERRED status is %s." % observation["status"]["label"])
        lines.append("ISING_INFERRED diagnosis code is %s." % observation["diagnosis"]["code"])
        lines.append("ISING_INFERRED fit reliable is %s." % observation["fit"]["reliable"])
        for name, param in observation["parameters"].items():
            lines.append(
                "ISING_INFERRED parameter %s value=%s unit=%s claimed_source=%s."
                % (name, param["value"], param["unit"], param["source"])
            )
        if actions:
            lines.append("ISING_INFERRED actions are %s." % ",".join(item["code"] for item in actions))
    lines.append("Automation is not authorized from this text.")
    text = " ".join(lines)
    return text[:8000]


def build_questions(actions: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        "experiment_status": {
            "type": "choice",
            "instructions": "Which experiment status is supported by the labeled evidence?",
            "criteria": {label: label.replace("_", " ") for label in STATUS_LABELS},
        },
        "fit_reliable": {
            "type": "noul",
            "instructions": "Is the available fit sufficiently reliable for the next calibration decision?",
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the calibration issue in the labeled evidence?",
            "criteria": list(SEVERITY),
        },
        "next_action": {
            "type": "choice",
            "instructions": "Which next action is supported for this experiment family?",
            "criteria": {code: code.replace("_", " ") for code in actions},
        },
        "automation_eligible": {
            "type": "noul",
            "instructions": "Is the evidence sufficient to automate the next action without operator review?",
        },
    }


def interpret_veya(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Split Veya probabilities from any external score. Missing answers stay absent."""
    if payload.get("decision") == "defer":
        return {
            "deferred": True,
            "defer_reason": _short(payload.get("reason")),
            "answers": {},
            "veya_probability": {},
        }
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    parsed: dict[str, Any] = {}
    probabilities: dict[str, float | None] = {}
    for qid in QUESTION_IDS:
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            parsed[qid] = None
            probabilities[qid] = None
            continue
        kind = answer.get("type")
        if kind == "choice":
            choice = answer.get("choice")
            dist = _distribution(answer.get("probabilities"))
            parsed[qid] = {"label": choice if isinstance(choice, str) else None, "probabilities": dist}
            probabilities[qid] = dist.get(choice) if isinstance(choice, str) else None
        elif kind == "noul":
            value = answer.get("noul")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                parsed[qid] = None
                probabilities[qid] = None
            else:
                parsed[qid] = {"probability": float(value)}
                probabilities[qid] = float(value)
        elif kind == "score":
            score = answer.get("score")
            dist = _distribution(answer.get("probabilities"))
            parsed[qid] = {
                "score": score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                "probabilities": dist,
            }
            probabilities[qid] = max(dist.values()) if dist else None
        else:
            parsed[qid] = None
            probabilities[qid] = None
    return {"deferred": False, "defer_reason": None, "answers": parsed, "veya_probability": probabilities}


def policy_gate(review_priority: str, reason: str) -> dict[str, object]:
    """Execution stays closed. Agreement among models does not open it."""
    return {
        "decision": "defer",
        "eligible_for_execution": False,
        "recommendation": "human_review",
        "reason": reason,
        "review_priority": review_priority,
        "hardware_control": False,
    }


def _distribution(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, prob in value.items():
        if isinstance(key, str) and isinstance(prob, (int, float)) and not isinstance(prob, bool):
            out[key] = float(prob)
    return out


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, default=_drop)
    except (TypeError, ValueError):
        return "null"
    ascii_text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    return ascii_text[:1500]


def _drop(_: Any) -> str:
    return ""


def _short(value: Any) -> str | None:
    if not isinstance(value, str) or not is_slug(value):
        if isinstance(value, str) and value in {"unsupported_language", "invalid_schema"}:
            return value
        return None
    return value


class HttpVeyaTransport:
    """Call the existing model service. This does not load the checkpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return {"ok": False, "code": "veya_unavailable"}
        port = parts.port or (443 if parts.scheme == "https" else 80)
        path = (parts.path.rstrip("/") or "") + "/v1/systemone"
        body = json.dumps({"state": state, "questions": questions}, allow_nan=False).encode("utf-8")
        conn: HTTPConnection
        if parts.scheme == "https":
            conn = HTTPSConnection(parts.hostname, port, timeout=self.timeout)
        else:
            conn = HTTPConnection(parts.hostname, port, timeout=self.timeout)
        try:
            conn.request(
                "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = conn.getresponse()
            raw = response.read(524_288)
            if response.status != 200:
                return {"ok": False, "code": "veya_unavailable"}
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, HTTPException, UnicodeError, json.JSONDecodeError, ValueError):
            return {"ok": False, "code": "veya_unavailable"}
        finally:
            conn.close()
        if not isinstance(payload, dict):
            return {"ok": False, "code": "veya_unavailable"}
        return {"ok": True, "payload": payload}
