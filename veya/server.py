"""Local HTTP inference for a loaded Veya agent. Stdlib only.

The process that calls ``serve`` owns the agent. Request bodies are JSON.
State text is not written to the server log. One inference lock serializes
forwards: the checkpoint is a single module, and concurrent MPS calls are not
a supported serving mode.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import __version__
from .common import state_has_content

_INFER_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 1024 * 1024


class RequestTooLarge(ValueError):
    pass


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key %r" % key)
        out[key] = value
    return out


def _question(kind: str, body: dict) -> dict:
    question = {"type": kind, "instructions": body["instructions"]}
    if "criteria" in body:
        question["criteria"] = body["criteria"]
    return {"q": question}


def _public_answer(result: dict, qid: str = "q") -> dict:
    answer = (result.get("answers") or {}).get(qid, {})
    return {
        "model": result.get("model"),
        "model_version": __version__,
        "primitive": answer.get("type"),
        "decision": result.get("decision"),
        "answer": answer.get("choice", answer.get("noul", answer.get("score"))),
        "probabilities": answer.get("probabilities"),
        "confidence": answer.get("confidence"),
        "calibration_version": result.get("calibration_version"),
        "deferred": result.get("decision") != "answer",
        "defer_reason": result.get("reason"),
        "usage": result.get("usage"),
    }


def _version_payload(agent, bundle=None, hypothesis=None) -> dict:
    payload = {
        "model_version": __version__,
        "checkpoint": agent.cfg.get("model_name"),
        "encoder": agent.cfg.get("encoder"),
        "calibration_version": getattr(agent, "calibration_version", "checkpoint"),
        "max_len": agent.cfg.get("max_len"),
        "head_max_len": agent.cfg.get("head_max_len"),
        "architecture_version": None,
        "checkpoint_sha256": None,
        "shortlist_sha256": None,
        "route_calibration_temperature": None,
        "choice_router_version": None,
        "reranker_id": None,
        "reranker_sha256": None,
        "hypothesis_scorer_id": None,
        "hypothesis_template_version": None,
        "release": None,
        "release_manifest_sha256": None,
        "selective_policy_sha256": None,
        "supported_languages": None,
        "supported_primitives": None,
        "release_status": None,
        "encoder_adaptation_id": None,
        "hypothesis_calibration_sha256": None,
        "hypothesis_head_sha256": None,
        "encoder_delta_sha256": None,
        "hypothesis_temperature": None,
    }
    active = hypothesis if hypothesis is not None else getattr(bundle, "hypothesis", None)
    if active is not None:
        payload["hypothesis_scorer_id"] = getattr(active, "scorer_id", None)
        payload["hypothesis_template_version"] = getattr(active, "template_version", None)
        payload["encoder_adaptation_id"] = getattr(active, "encoder_adaptation_id", None)
        payload["hypothesis_calibration_sha256"] = getattr(active, "calibration_sha256", None)
        payload["hypothesis_head_sha256"] = getattr(active, "head_sha256", None)
        payload["encoder_delta_sha256"] = getattr(active, "encoder_delta_sha256", None)
        payload["hypothesis_temperature"] = getattr(active, "temperature", None)
    if bundle is not None:
        payload["architecture_version"] = "veya-hypothesis-hce2-v1" if getattr(bundle, "hypothesis", None) else "veya-shortlist-v1"
        payload["checkpoint_sha256"] = bundle.checkpoint_hash
        payload["shortlist_sha256"] = bundle.shortlist_sha256
        payload["route_calibration_temperature"] = bundle.route_temperature
        payload["choice_router_version"] = bundle.router.get("version")
        payload["reranker_id"] = getattr(bundle, "reranker_id", None) or "base_decision_head"
        payload["reranker_sha256"] = getattr(bundle, "reranker_sha256", None)
        manifest = getattr(bundle, "manifest", None)
        if manifest:
            payload["release"] = manifest.get("release")
            payload["release_manifest_sha256"] = getattr(bundle, "manifest_sha256", None)
            payload["selective_policy_sha256"] = manifest.get("selective_policy_hash")
            payload["supported_languages"] = manifest.get("supported_languages")
            payload["supported_primitives"] = manifest.get("supported_primitives")
            payload["release_status"] = manifest.get("release_status")
    return payload


def _validate_request(body, kind=None, systemone=False):
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if not state_has_content(body.get("state")):
        raise ValueError("state must be a non-empty string, object, or list")
    from .agent import Agent
    if systemone:
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        for qid, question in questions.items():
            if not isinstance(qid, str) or not qid:
                raise ValueError("question ids must be non-empty strings")
            Agent._to_internal(question)
        return
    if "instructions" not in body:
        raise ValueError("instructions is required")
    question = {"type": kind, "instructions": body["instructions"]}
    if "criteria" in body:
        question["criteria"] = body["criteria"]
    Agent._to_internal(question)


def make_handler(agent, bundle=None, hypothesis=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length <= 0:
                return {}
            if length > MAX_REQUEST_BYTES:
                raise RequestTooLarge("request exceeds %d bytes" % MAX_REQUEST_BYTES)
            return json.loads(self.rfile.read(length).decode("utf-8"), object_pairs_hook=_strict_object)

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"status": "ok"})
                return
            if path == "/version":
                self._send(200, _version_payload(agent, bundle, hypothesis))
                return
            self._send(404, {"decision": "invalid", "defer_reason": "not_found"})

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._body()
            except RequestTooLarge as exc:
                self._send(413, {"decision": "invalid", "defer_reason": "request_too_large", "detail": str(exc)})
                return
            except (json.JSONDecodeError, UnicodeError, ValueError):
                self._send(400, {"decision": "invalid", "defer_reason": "invalid_json"})
                return
            try:
                if path == "/v1/systemone":
                    _validate_request(body, systemone=True)
                    questions = body["questions"]
                    if bundle is not None:
                        with _INFER_LOCK:
                            result = bundle.answer_questions(body.get("state"), questions)
                    else:
                        with _INFER_LOCK:
                            result = agent.system_one(body.get("state"), questions)
                    result["calibration_version"] = getattr(agent, "calibration_version", "checkpoint")
                    self._send(200, result)
                    return
                if path in ("/v1/choice-hypothesis", "/v1/choice-compare"):
                    if hypothesis is None:
                        self._send(404, {"decision": "invalid", "defer_reason": "hypothesis_not_loaded"})
                        return
                    _validate_request(body, kind="choice")
                    if path == "/v1/choice-compare" and bundle is None:
                        self._send(400, {"decision": "invalid", "defer_reason": "invalid_schema"})
                        return
                    with _INFER_LOCK:
                        experimental = hypothesis.public_choice(
                            body.get("state"), body["instructions"], body.get("criteria"), bundle,
                        )
                        if path == "/v1/choice-compare":
                            packed = bundle.public_choice(body.get("state"), body["instructions"], body.get("criteria"))
                            self._send(200, {"packed": packed, "hypothesis": experimental})
                            return
                    self._send(200, experimental)
                    return
                kind = {"/v1/choice": "choice", "/v1/noul": "noul", "/v1/score": "score"}.get(path)
                if kind is None:
                    self._send(404, {"decision": "invalid", "defer_reason": "not_found"})
                    return
                _validate_request(body, kind=kind)
                if kind == "choice" and bundle is not None:
                    with _INFER_LOCK:
                        public = bundle.public_choice(body.get("state"), body["instructions"], body.get("criteria"))
                    public["calibration_version"] = getattr(agent, "calibration_version", "checkpoint")
                    self._send(200, public)
                    return
                with _INFER_LOCK:
                    result = agent.system_one(body.get("state"), _question(kind, body))
                result["calibration_version"] = getattr(agent, "calibration_version", "checkpoint")
                public = _public_answer(result)
                if kind == "choice":
                    public["route"] = "packed"
                self._send(200, public)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                self._send(400, {"decision": "invalid", "defer_reason": "invalid_schema", "detail": str(exc)})

    return Handler


def serve(agent, host: str = "127.0.0.1", port: int = 8080, bundle=None, hypothesis=None) -> ThreadingHTTPServer:
    if hypothesis is None and bundle is not None:
        hypothesis = getattr(bundle, "hypothesis", None)
    server = ThreadingHTTPServer((host, port), make_handler(agent, bundle, hypothesis))
    return server
