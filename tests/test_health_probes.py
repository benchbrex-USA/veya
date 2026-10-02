"""Health, readiness, and startup probes (G18) on the web gateway.

The gateway derives deep health from the audited model service: the service
only starts serving after ``load_release_manifest`` verified every release
hash, so /readyz reports ready exactly when the service presents the audited
release bindings. A healthy service that cannot present those bindings is an
integrity failure and must never make the gateway ready (fail-closed).

Run: .venv/bin/python -m pytest tests/test_health_probes.py -v
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from web import server as gateway

RELEASE_HASHES = {
    "release_manifest_sha256": "ab" * 32,
    "checkpoint_sha256": "cd" * 32,
    "shortlist_sha256": "ef" * 32,
    "selective_policy_sha256": "12" * 32,
}
RELEASE_VERSION = {
    "model_version": "0.3.5",
    "release": "0.3.5-audit-baseline.1",
    "architecture_version": "veya-shortlist-v1",
    "route_calibration_temperature": 1.25,
    **RELEASE_HASHES,
}


class ModelDouble(BaseHTTPRequestHandler):
    """Controllable stand-in for the audited model service."""

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
        state = type(self).state
        if self.path == "/health":
            if state.get("health_ok", True):
                self._reply(200, {"status": "ok"})
            else:
                self._reply(503, {"status": "unavailable"})
            return
        if self.path == "/version":
            self._reply(200, state.get("version", {}))
            return
        self._reply(404, {"code": "not_found"})


@pytest.fixture
def probe_pair(monkeypatch):
    state = {"health_ok": True, "version": dict(RELEASE_VERSION)}
    ModelDouble.state = state
    model = ThreadingHTTPServer(("127.0.0.1", 0), ModelDouble)
    model_thread = threading.Thread(target=model.serve_forever, daemon=True)
    model_thread.start()
    monkeypatch.setattr(
        gateway, "UPSTREAM", f"http://127.0.0.1:{model.server_address[1]}"
    )
    monkeypatch.setattr(
        gateway, "_READINESS", {key: False for key in gateway._READINESS}
    )
    monkeypatch.setattr(gateway, "_READINESS_PROBE", {"at": 0.0})
    monkeypatch.setattr(
        gateway, "_RELEASE_BINDINGS", {"model_hash": "", "policy_hash": ""}
    )
    monkeypatch.setattr(gateway, "READINESS_TTL_S", 0.0)
    monkeypatch.setattr(gateway, "version_cache", (0, {}))
    web = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    web_thread = threading.Thread(target=web.serve_forever, daemon=True)
    web_thread.start()
    yield state, web.server_address[1]
    web.shutdown()
    web.server_close()
    model.shutdown()
    model.server_close()


def probe(port, path):
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def test_healthz_is_liveness_only(probe_pair, monkeypatch):
    _, port = probe_pair
    monkeypatch.setattr(gateway, "UPSTREAM", "http://127.0.0.1:1")
    status, body, _ = probe(port, "/healthz")
    assert status == 200
    assert body == {"status": "alive"}


def test_readyz_reports_not_ready_without_components(probe_pair, monkeypatch):
    _, port = probe_pair
    monkeypatch.setattr(gateway, "UPSTREAM", "http://127.0.0.1:1")
    status, body, headers = probe(port, "/readyz")
    assert status == 503
    assert body == {
        "status": "not_ready",
        "components": {
            "model_loaded": False,
            "policy_loaded": False,
            "calibration_loaded": False,
            "shortlist_loaded": False,
            "hashes_valid": False,
            "startup_failed": False,
        },
    }
    assert "X-Veya-Model-Hash" not in headers
    assert "X-Veya-Policy-Hash" not in headers


def test_readyz_reports_ready_for_the_audited_release(probe_pair):
    _, port = probe_pair
    status, body, headers = probe(port, "/readyz")
    assert status == 200
    assert body["status"] == "ready"
    assert body["components"] == {
        "model_loaded": True,
        "policy_loaded": True,
        "calibration_loaded": True,
        "shortlist_loaded": True,
        "hashes_valid": True,
        "startup_failed": False,
    }
    assert headers["X-Veya-Model-Hash"] == RELEASE_HASHES["checkpoint_sha256"]
    assert (
        headers["X-Veya-Policy-Hash"] == RELEASE_HASHES["selective_policy_sha256"]
    )


def test_readyz_reports_partial_component_status(probe_pair):
    state, port = probe_pair
    state["version"] = {
        key: value
        for key, value in RELEASE_VERSION.items()
        if key != "selective_policy_sha256"
    }
    status, body, _ = probe(port, "/readyz")
    assert status == 503
    assert body["status"] == "not_ready"
    components = body["components"]
    assert components["model_loaded"] is True
    assert components["shortlist_loaded"] is True
    assert components["calibration_loaded"] is True
    assert components["hashes_valid"] is True
    assert components["policy_loaded"] is False
    assert components["startup_failed"] is False


def test_readyz_detects_service_without_release_bindings(probe_pair):
    state, port = probe_pair
    state["version"] = {"model_version": "0.3.5"}
    status, body, _ = probe(port, "/readyz")
    assert status == 503
    assert body["components"]["model_loaded"] is True
    assert body["components"]["startup_failed"] is True


def test_malformed_release_digests_fail_closed(probe_pair):
    state, port = probe_pair
    state["version"] = {**RELEASE_VERSION, "checkpoint_sha256": "not-a-digest"}
    status, body, _ = probe(port, "/readyz")
    assert status == 503
    assert body["components"]["startup_failed"] is True


def test_startup_failure_is_sticky_and_never_ready(probe_pair):
    state, port = probe_pair
    state["version"] = {"model_version": "0.3.5"}
    status, body, _ = probe(port, "/readyz")
    assert status == 503
    assert body["components"]["startup_failed"] is True
    # A later, fully valid release must not clear the failure: fail-closed.
    state["version"] = dict(RELEASE_VERSION)
    for path in ("/readyz", "/startupz", "/readyz"):
        status, body, headers = probe(port, path)
        assert status == 503
        assert body["components"]["startup_failed"] is True
        assert "X-Veya-Model-Hash" not in headers
    # Liveness must stay green even while readiness is permanently failed.
    status, body, _ = probe(port, "/healthz")
    assert status == 200
    assert body == {"status": "alive"}


def test_startupz_mirrors_readyz(probe_pair):
    state, port = probe_pair
    state["version"] = {
        key: value
        for key, value in RELEASE_VERSION.items()
        if key != "selective_policy_sha256"
    }
    ready_status, ready_body, _ = probe(port, "/readyz")
    startup_status, startup_body, _ = probe(port, "/startupz")
    assert ready_status == startup_status == 503
    assert ready_body == startup_body
    state["version"] = dict(RELEASE_VERSION)
    ready_status, ready_body, ready_headers = probe(port, "/readyz")
    startup_status, startup_body, startup_headers = probe(port, "/startupz")
    assert ready_status == startup_status == 200
    assert ready_body == startup_body
    assert (
        startup_headers["X-Veya-Model-Hash"] == ready_headers["X-Veya-Model-Hash"]
    )
    assert (
        startup_headers["X-Veya-Policy-Hash"] == ready_headers["X-Veya-Policy-Hash"]
    )


def test_existing_health_and_version_routes_unchanged(probe_pair, monkeypatch):
    _, port = probe_pair
    status, body, _ = probe(port, "/api/health")
    assert status == 200
    assert body == {"status": "ok"}
    status, body, _ = probe(port, "/api/version")
    assert status == 200
    assert body["model_version"] == "0.3.5"
    assert (
        body["release_manifest_sha256"]
        == RELEASE_HASHES["release_manifest_sha256"]
    )
    monkeypatch.setattr(gateway, "UPSTREAM", "http://127.0.0.1:1")
    status, body, _ = probe(port, "/api/health")
    assert status == 503
    assert body["code"] == "model_unavailable"
