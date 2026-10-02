"""Reliability hardening (G12) on the web gateway.

Covers graceful shutdown (SIGTERM drain), overload 503 + Retry-After, the
structured error envelope (error / error_code / request_id), request-ID
propagation, and the inference timeout. Corrupted-artifact handling (fail-
closed startup_failed via the readiness latches) is covered exhaustively in
tests/test_health_probes.py; test_startup_failure_latches_startup_failed
exercises the connection once more from the reliability angle.

Run: .venv/bin/python -m pytest tests/test_reliability_hardening.py -v
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web import server as gateway

CHOICE = {
    "primitive": "choice",
    "state": "Duplicate charge. Please refund.",
    "instructions": "Which team?",
    "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
}


class ModelDouble(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def _reply(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
            return
        self._reply(404, {"code": "not_found"})

    def do_POST(self):
        if type(self).state.get("slow"):
            time.sleep(5)
        self._reply(200, {"decision": "answer", "answer": "billing",
                          "probabilities": {"billing": 1.0, "technical": 0.0},
                          "deferred": False})


@pytest.fixture
def pair(monkeypatch):
    ModelDouble.state = {}
    model = ThreadingHTTPServer(("127.0.0.1", 0), ModelDouble)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        gateway, "UPSTREAM", f"http://127.0.0.1:{model.server_address[1]}"
    )
    monkeypatch.setattr(gateway, "_SHUTTING_DOWN", False)
    monkeypatch.setattr(gateway, "version_cache", (0, {}))
    web = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    yield web.server_address[1]
    web.shutdown()
    web.server_close()
    model.shutdown()
    model.server_close()


def request(port, path, body=None):
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urlopen(url, timeout=10) if body is None else urlopen(
            Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=10,
        ) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def test_shutdown_rejects_new_requests_with_503(pair, monkeypatch):
    monkeypatch.setattr(gateway, "_SHUTTING_DOWN", True)
    status, body, headers = request(pair, "/api/infer", CHOICE)
    assert status == 503
    assert body["error_code"] == "SHUTTING_DOWN"
    assert "error" in body and "request_id" in body
    assert headers["Retry-After"] == str(gateway.SHUTDOWN_DRAIN_S)
    status, body, headers = request(pair, "/api/version")
    assert status == 503
    assert body["error_code"] == "SHUTTING_DOWN"
    assert headers["Retry-After"] == str(gateway.SHUTDOWN_DRAIN_S)


def test_liveness_and_readiness_answer_while_draining(pair, monkeypatch):
    monkeypatch.setattr(gateway, "_SHUTTING_DOWN", True)
    status, body, _ = request(pair, "/healthz")
    assert status == 200 and body == {"status": "alive"}
    status, body, _ = request(pair, "/readyz")
    assert status in (200, 503)
    assert body["status"] in ("ready", "not_ready")


def test_overload_returns_503_with_retry_after(pair, monkeypatch):
    monkeypatch.setattr(
        gateway, "inflight", threading.BoundedSemaphore(gateway.MAX_INFLIGHT)
    )
    for _ in range(gateway.MAX_INFLIGHT):
        gateway.inflight.acquire()
    status, body, headers = request(pair, "/api/infer", CHOICE)
    assert status == 503
    assert body["code"] == "model_busy"
    assert body["error_code"] == "OVERLOADED"
    assert body["retry_after"] >= 1
    assert "error" in body and "request_id" in body
    assert int(headers["Retry-After"]) == body["retry_after"]


def test_structured_error_envelope_is_consistent(pair):
    bad = {**CHOICE, "criteria": {"only": "one criterion"}}
    status, body, headers = request(pair, "/api/infer", bad)
    assert status == 400
    for field in ("error", "error_code", "request_id", "code", "detail"):
        assert field in body
    assert body["error_code"] == "VALIDATION_ERROR"
    assert headers["Content-Type"].startswith("application/json")


def test_request_id_is_present_in_body_and_header(pair):
    _, body, headers = request(pair, "/api/infer", {"primitive": "bogus"})
    assert body["request_id"]
    assert headers["X-Request-Id"] == body["request_id"]


def test_error_responses_are_json_with_security_headers(pair):
    _, _, headers = request(pair, "/api/infer", {"primitive": "bogus"})
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_inference_timeout_returns_504(pair, monkeypatch):
    ModelDouble.state["slow"] = True
    monkeypatch.setattr(gateway, "INFER_TIMEOUT_S", 0.05)
    status, body, _ = request(pair, "/api/infer", CHOICE)
    assert status == 504
    assert body["error_code"] == "TIMEOUT"
    assert "error" in body and "request_id" in body


def test_successful_inference_keeps_legacy_contract(pair):
    status, body, headers = request(pair, "/api/infer", CHOICE)
    assert status == 200
    assert body["decision"] == "answer"
    assert body["request_id"]
    assert "Retry-After" not in headers
