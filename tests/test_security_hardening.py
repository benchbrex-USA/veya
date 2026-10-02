"""Tests for security hardening (G13).

Covers the website gateway's security headers, the ``veya.security_audit``
logger (hashed client IP, path, error code/reason — never raw input), and
input-handling safety: manifest path traversal, static-file traversal, and
JSON-only deserialization on every request path.

Uses the project's standard double-server pattern: a real HTTP hop from
urllib through the website gateway (``web.server.Handler``) into a
controlled model-service double.
"""
import hashlib
import http.client
import inspect
import json
import logging
import os
import pickle
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web import contracts
from web import server as gateway
from veya.release_bundle import resolve_manifest_path

CHOICE = {
    "primitive": "choice",
    "state": "Duplicate charge. Please refund.",
    "instructions": "Which team?",
    "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
}
HSTS = "max-age=31536000; includeSubDomains"
CLIENT_IP = "127.0.0.1"
CLIENT_HASH = hashlib.sha256(CLIENT_IP.encode()).hexdigest()[:16]


class _AuditCapture(logging.Handler):
    """Collect LogRecords from the audit logger without formatting them."""

    def __init__(self):
        super().__init__(level=0)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def audit_log():
    capture = _AuditCapture()
    audit = logging.getLogger("veya.security_audit")
    audit.addHandler(capture)
    try:
        yield capture
    finally:
        audit.removeHandler(capture)


def _line(record: logging.LogRecord) -> str:
    """Everything an operator could read on one rendered audit line."""
    parts = [record.getMessage()]
    for key in ("status", "client_hash", "path", "error_code", "reason"):
        parts.append("%s=%s" % (key, record.__dict__.get(key)))
    return " ".join(parts)


@pytest.fixture
def http_pair(monkeypatch):
    """Website gateway in front of a controlled model-service double."""
    mode = {"kind": "normal"}

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except BrokenPipeError:
                pass

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok"})
            elif self.path == "/version":
                self.reply(200, {"model_version": "0.3.5", "release": "research-test"})
            else:
                self.reply(404, {"code": "not_found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if mode["kind"] == "unsupported":
                self.reply(
                    200,
                    {
                        "model": "veya-rl-agent",
                        "decision": "defer",
                        "deferred": True,
                        "defer_reason": "unsupported_language",
                        "answer": None,
                        "probabilities": None,
                    },
                )
                return
            labels = list(body["criteria"])
            probs = {
                label: (1.0 if i == 0 else 0.0) for i, label in enumerate(labels)
            }
            self.reply(
                200,
                {
                    "model": "veya-rl-agent",
                    "decision": "answer",
                    "deferred": False,
                    "answer": labels[0],
                    "probabilities": probs,
                },
            )

    model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        gateway, "UPSTREAM", f"http://127.0.0.1:{model.server_address[1]}"
    )
    monkeypatch.setattr(gateway, "version_cache", (0, {}))
    monkeypatch.setattr(gateway, "CAPACITY", 12)
    gateway.rate_hits.clear()
    web = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    yield mode, web.server_address[1]
    web.shutdown()
    web.server_close()
    model.shutdown()
    model.server_close()


def call(port, payload=None, raw=None):
    body = (
        raw
        if raw is not None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    req = Request(
        f"http://127.0.0.1:{port}/api/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


class TestSecurityHeaders:
    """Verify all required security headers are present."""

    def test_hsts_header(self, http_pair):
        _, port = http_pair
        with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            assert (
                response.headers["Strict-Transport-Security"] == HSTS
            )

    def test_nosniff_and_frame_deny(self, http_pair):
        _, port = http_pair
        with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"

    def test_csp_header_already_enforced(self, http_pair):
        _, port = http_pair
        with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            csp = response.headers["Content-Security-Policy"]
            assert "default-src 'self'" in csp
            assert "object-src 'none'" in csp
            assert "frame-ancestors 'none'" in csp

    def test_headers_present_on_error_responses(self, http_pair):
        _, port = http_pair
        with pytest.raises(HTTPError) as exc:
            urlopen(f"http://127.0.0.1:{port}/api/not-a-route", timeout=3)
        for name, expected in (
            ("Strict-Transport-Security", HSTS),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
        ):
            assert exc.value.headers[name] == expected
        assert "default-src 'self'" in exc.value.headers["Content-Security-Policy"]
        status, _, headers = call(port, {**CHOICE, "criteria": {"only": "one"}})
        assert status == 400
        assert headers["Strict-Transport-Security"] == HSTS
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"


class TestAuditLogging:
    """Verify security audit logging works."""

    def test_audit_logger_configuration(self):
        audit = logging.getLogger("veya.security_audit")
        assert audit.level == logging.WARNING
        assert any(
            isinstance(handler, logging.StreamHandler) and handler.formatter
            for handler in audit.handlers
        )

    def test_validation_failure_produces_audit_entry(self, http_pair, audit_log):
        _, port = http_pair
        status, out, _ = call(port, {**CHOICE, "criteria": {"only": "one candidate"}})
        assert status == 400 and out["code"] == "invalid_input"
        assert len(audit_log.records) == 1
        record = audit_log.records[0]
        assert record.name == "veya.security_audit"
        assert record.levelno == logging.WARNING
        assert record.status == 400
        assert record.error_code == "invalid_input"
        assert record.path == "/api/infer"
        assert record.reason

    def test_malformed_request_body_produces_audit_entry(self, http_pair, audit_log):
        _, port = http_pair
        status, out, _ = call(port, raw=b'{"primitive": "choice",')
        assert status == 400 and out["code"] == "invalid_input"
        record = audit_log.records[-1]
        assert record.error_code == "invalid_input"
        assert record.reason == "Invalid JSON body."

    def test_rate_limit_hit_produces_audit_entry(
        self, http_pair, monkeypatch, audit_log
    ):
        _, port = http_pair
        monkeypatch.setattr(gateway, "CAPACITY", 1)
        gateway.rate_hits.clear()
        status, _, _ = call(port, CHOICE)
        assert status == 200
        assert not audit_log.records
        status, out, _ = call(port, CHOICE)
        assert status == 429 and out["code"] == "rate_limited"
        record = audit_log.records[-1]
        assert record.status == 429 and record.error_code == "rate_limited"

    def test_unsupported_language_rejection_is_audited(self, http_pair, audit_log):
        mode, port = http_pair
        mode["kind"] = "unsupported"
        status, out, _ = call(port, CHOICE)
        assert status == 200 and out["deferred"]
        assert out["defer_reason"] == "unsupported_language"
        record = audit_log.records[-1]
        assert record.error_code == "unsupported_language"
        assert record.path == "/api/infer"

    def test_oversize_request_and_outage_are_audited(
        self, http_pair, monkeypatch, audit_log
    ):
        _, port = http_pair
        status, out, _ = call(
            port, CHOICE, raw=b" " * (gateway.MAX_REQUEST_BYTES + 1)
        )
        assert status == 413 and out["code"] == "request_too_large"
        record = audit_log.records[-1]
        assert record.status == 413 and record.error_code == "request_too_large"
        monkeypatch.setattr(gateway, "UPSTREAM", "http://127.0.0.1:1")
        status, out, _ = call(port, CHOICE)
        assert status == 503 and out["code"] == "model_unavailable"
        record = audit_log.records[-1]
        assert record.status == 503 and record.error_code == "model_unavailable"

    def test_not_found_responses_are_audited(self, http_pair, audit_log):
        _, port = http_pair
        with pytest.raises(HTTPError):
            urlopen(f"http://127.0.0.1:{port}/api/not-a-route", timeout=3)
        assert audit_log.records[-1].status == 404
        assert audit_log.records[-1].error_code == "not_found"
        with pytest.raises(HTTPError):
            urlopen(f"http://127.0.0.1:{port}/not-a-page", timeout=3)
        assert audit_log.records[-1].status == 404
        assert audit_log.records[-1].error_code == "not_found"

    def test_audit_log_does_not_contain_raw_request_body(self, http_pair, audit_log):
        _, port = http_pair
        sentinel = "AUDIT-SENTINEL-candidate-候选-<script>alert(1)</script>"
        status, _, _ = call(
            port,
            {
                **CHOICE,
                "state": sentinel,
                "instructions": "   ",
                "criteria": {"billing": sentinel, "technical": "Bugs"},
            },
        )
        assert status == 400
        rendered = "\n".join(_line(record) for record in audit_log.records)
        assert sentinel not in rendered
        assert "billing" not in rendered

    def test_client_ip_is_hashed_not_raw(self, http_pair, audit_log):
        _, port = http_pair
        status, _, _ = call(port, raw=b"not json")
        assert status == 400
        record = audit_log.records[-1]
        assert record.client_hash == CLIENT_HASH
        assert record.client_hash != CLIENT_IP
        assert CLIENT_IP not in _line(record)

    def test_audit_line_renders_structured_fields(self, http_pair, audit_log):
        _, port = http_pair
        status, _, _ = call(port, raw=b"{")
        assert status == 400
        record = audit_log.records[-1]
        formatter = next(
            handler.formatter
            for handler in logging.getLogger("veya.security_audit").handlers
            if isinstance(handler, logging.StreamHandler) and handler.formatter
        )
        line = formatter.format(record)
        assert f"client={CLIENT_HASH}" in line
        assert "path=/api/infer" in line
        assert "code=invalid_input" in line
        assert CLIENT_IP not in line


class TestInputSafety:
    """Verify input handling safety."""

    def test_manifest_path_traversal_rejected(self, tmp_path):
        root = tmp_path / "release"
        (root / "inner").mkdir(parents=True)
        (tmp_path / "outside.json").write_text("{}")
        escape = tmp_path / "escape"
        escape.mkdir()
        os.symlink(escape, root / "link")
        for hostile in (
            "../outside.json",
            "inner/../../outside.json",
            "/etc/passwd",
            "",
            123,
            "link/secret.json",
        ):
            with pytest.raises(ValueError):
                resolve_manifest_path(str(root), hostile)
        resolved = resolve_manifest_path(str(root), "inner/file.json")
        assert resolved == str((root / "inner" / "file.json").resolve())

    def test_static_path_traversal_blocked(self, http_pair, audit_log):
        _, port = http_pair
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", "/static/../server.py")
        response = connection.getresponse()
        body = response.read()
        connection.close()
        assert response.status == 404
        assert b"rate_hits" not in body
        assert b"not_found" in body
        record = audit_log.records[-1]
        assert record.status == 404 and record.error_code == "not_found"
        assert record.path == "/static/../server.py"

    def test_pickle_body_is_never_deserialized(self, http_pair, audit_log):
        _, port = http_pair
        pickled = pickle.dumps({**CHOICE, "state": "pickle-payload"})
        status, out, _ = call(port, raw=pickled)
        assert status == 400 and out["code"] == "invalid_input"
        assert out["detail"] == "Invalid JSON body."
        record = audit_log.records[-1]
        assert "pickle-payload" not in _line(record)

    def test_yaml_body_is_never_parsed(self, http_pair):
        _, port = http_pair
        status, out, _ = call(port, raw=b"primitive: choice\nstate: yaml")
        assert status == 400 and out["code"] == "invalid_input"

    def test_request_path_modules_use_json_only(self):
        from veya import server as model_server

        unsafe_import = re.compile(
            r"(?m)^\s*(import|from)\s+(pickle|yaml|marshal|shelve)\b"
        )
        unsafe_call = re.compile(r"(?<![.\w])(eval|exec)\(")
        for module in (gateway, contracts, model_server):
            source = inspect.getsource(module)
            assert not unsafe_import.search(source), module.__name__
            assert not unsafe_call.search(source), module.__name__

    def test_telemetry_has_no_raw_input(self, http_pair, monkeypatch):
        _, port = http_pair
        records = []
        monkeypatch.setattr(gateway.logger, "info", lambda entry: records.append(entry))
        payload = {
            **CHOICE,
            "state": "TELEMETRY-SENTINEL-状态",
            "criteria": {
                "billing": "TELEMETRY-SENTINEL",
                "technical": "Bugs and outages",
            },
        }
        status, _, _ = call(port, payload)
        assert status == 200
        joined = "".join(records)
        assert "TELEMETRY-SENTINEL" not in joined
