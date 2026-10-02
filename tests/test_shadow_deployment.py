"""Shadow deployment (G20): the real gateway process in shadow mode, end to end.

``docs/SHADOW_DEPLOYMENT.md`` documents the production procedure this file
verifies at the boundary the repository can exercise without a cluster:

  * ``VEYA_SHADOW_MODE=1`` starts the real ``web.server`` process, flags every
    telemetry entry with ``"shadow": true``, and answers with
    ``X-Veya-Shadow: true`` (and, in shadow-off mode, does neither).
  * The telemetry entry carries every field that
    ``artifacts/release/shadow_schema.json`` sources from the gateway:
    timestamp, request_id, config_hash, primitive, candidate_count, route,
    confidence, automation, latency_ms, ...
  * ``GET /readyz`` serves the model_hash / policy_hash join keys for records.
  * The joined record feeds ``research/veyabench/drift_monitor.py`` unchanged
    once the schema's nested ``slice_metadata`` is flattened to the top level
    (the transformation is exercised below and documented in the doc).
  * The sample size that clears the risk gate matches
    ``research/veyabench/sample_size.py`` (189 accepted at epsilon=0.02).

The heavy sibling of this test is the manual real-weight run
(``tests/test_local_e2e.py``, model service + ``web.model_server``); this file
runs the same gateway binary as a subprocess against a model-service double so
it can execute in CI.

Run: .venv/bin/python -m pytest tests/test_shadow_deployment.py -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "research" / "veyabench"))

from drift_monitor import drift_report, drift_triggers, period_stats  # noqa: E402
from sample_size import required_n  # noqa: E402

from web import server as gateway  # noqa: E402

CHOICE = {
    "primitive": "choice",
    "state": "Duplicate charge. Please refund.",
    "instructions": "Which team?",
    "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
}
# 64-hex release bindings, same fixtures as tests/test_health_probes.py.
RELEASE_HASHES = {
    "release_manifest_sha256": "ab" * 32,
    "checkpoint_sha256": "cd" * 32,
    "shortlist_sha256": "ef" * 32,
    "selective_policy_sha256": "12" * 32,
}


class ModelDouble(BaseHTTPRequestHandler):
    """Stand-in for the audited model service; records every infer request."""

    requests: list[tuple[str, dict]] = []

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
        if self.path == "/version":
            # Mirrors the release bindings served by the audited stack
            # (veya/server.py _version_payload via load_release_manifest).
            self._reply(
                200,
                {
                    "model_version": "0.3.5",
                    "release": "0.3.5-audit-baseline.1",
                    "architecture_version": "veya-shortlist-v1",
                    "route_calibration_temperature": 1.25,
                    "release_status": {"english_choice": "RESEARCH RELEASE"},
                    **RELEASE_HASHES,
                },
            )
            return
        self._reply(404, {"code": "not_found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        if self.path == "/v1/choice":
            # Mirrors veya/release_bundle.py public_choice: confidence,
            # automation, and max_probability are all part of the served
            # payload. The browser contract still strips max_probability
            # (web/contracts.py); the gateway captures it for telemetry only
            # (web/server.py _telemetry_max_probability, G16).
            self._reply(
                200,
                {
                    "model": "veya-rl-agent",
                    "decision": "answer",
                    "deferred": False,
                    "answer": "billing",
                    "probabilities": {"billing": 0.9, "technical": 0.1},
                    "route": "packed",
                    "confidence": 0.93,
                    "max_probability": 0.9,
                    "automation": "answer",
                    "calibration": "test",
                    "calibration_version": "test",
                    "candidate_count": len(body["criteria"]),
                },
            )
            return
        self._reply(404, {"code": "not_found"})


@pytest.fixture
def model_double():
    ModelDouble.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelDouble)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.25):
                return
        except OSError as exc:
            error = exc
            time.sleep(0.05)
    raise AssertionError(f"gateway did not listen on port {port}: {error}")


class GatewayProcess:
    """The real gateway process, launched exactly as the deployment doc does.

    ``VEYA_SHADOW_MODE=1 VEYA_WEB_HOST=127.0.0.1 .venv/bin/python -m web.server``
    """

    def __init__(self, model_url: str, shadow_mode: str):
        self.port = _free_port()
        env = {
            **os.environ,
            "VEYA_WEB_HOST": "127.0.0.1",
            "VEYA_WEB_PORT": str(self.port),
            "VEYA_MODEL_URL": model_url,
            "VEYA_SHADOW_MODE": shadow_mode,
        }
        self.process = subprocess.Popen(
            [sys.executable, "-m", "web.server"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.lines: list[str] = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        _wait_for_port(self.port)

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.append(line)

    def telemetry(self, request_id: str, timeout: float = 5.0) -> dict:
        """The telemetry entry web/server.py _safe_telemetry logged for a request."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in list(self.lines):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("request_id") == request_id:
                    return entry
            time.sleep(0.05)
        raise AssertionError(
            f"no telemetry entry for {request_id}; captured lines: {self.lines!r}"
        )

    def stop(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=2)


def _get(port: int, path: str):
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def _post_choice(port: int):
    request = Request(
        f"http://127.0.0.1:{port}/api/infer",
        data=json.dumps(CHOICE).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def _flatten_for_drift(record: dict, *, ground_truth, correct) -> dict:
    """shadow_schema.json record -> drift_monitor.py input (documented transform).

    drift_monitor.py reads ``candidate_count`` and ``route`` at the record's
    top level (research/veyabench/drift_monitor.py period_stats), while the
    schema nests them under ``slice_metadata``. ``ground_truth``/``correct``
    arrive from the labeling join later.
    """
    flat = {key: value for key, value in record.items() if key != "slice_metadata"}
    flat.update(
        {
            "candidate_count": record["slice_metadata"]["candidate_count"],
            "route": record["slice_metadata"]["route"],
            "ground_truth": ground_truth,
            "correct": correct,
        }
    )
    return flat


def _schema_record() -> dict:
    """One record in the exact artifacts/release/shadow_schema.json shape."""
    return {
        "timestamp": "2026-03-02T12:00:00+00:00",
        "request_id": "0" * 16,
        "model_hash": RELEASE_HASHES["checkpoint_sha256"],
        "policy_hash": RELEASE_HASHES["selective_policy_sha256"],
        "config_hash": RELEASE_HASHES["release_manifest_sha256"],
        "primitive": "choice",
        "prediction": {
            "answer": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "confidence": 0.93,
        "automation": "answer",
        "selective_decision": "accept",
        "latency_ms": 42,
        "ground_truth": None,
        "correct": None,
        "slice_metadata": {"candidate_count": 2, "route": "packed"},
        "slice_tag": "clean",
        "incumbent_outcome": None,
    }


def _drift_record(i: int, *, correct=True, automation="answer", config_hash="cfg"):
    """A labeled telemetry-shaped record; field names exactly as web/server.py emits."""
    return {
        "timestamp": (
            datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
        ).isoformat(),
        "request_id": f"{i:016x}",
        "primitive": "choice",
        "candidate_count": 2,
        "route": "packed",
        "deferred": automation == "defer",
        "confidence": 0.93 if automation == "answer" else None,
        "automation": automation,
        "config_hash": config_hash,
        "status": 200,
        "latency_ms": 40 + (i % 5),
        "ground_truth": "billing" if correct is not None else None,
        "correct": correct,
    }


def test_shadow_gateway_e2e_telemetry_and_record_fields(model_double):
    """Shadow mode on: header, telemetry entry, schema record, drift-monitor feed."""
    model_url = f"http://127.0.0.1:{model_double.server_address[1]}"
    gateway_process = GatewayProcess(model_url, shadow_mode="1")
    try:
        status, ready, headers = _get(gateway_process.port, "/readyz")
        assert status == 200 and ready["status"] == "ready"
        model_hash = headers["X-Veya-Model-Hash"]
        policy_hash = headers["X-Veya-Policy-Hash"]
        assert model_hash == RELEASE_HASHES["checkpoint_sha256"]
        assert policy_hash == RELEASE_HASHES["selective_policy_sha256"]

        status, inference, headers = _post_choice(gateway_process.port)
        assert status == 200, inference
        assert headers["X-Veya-Shadow"] == "true"

        entry = gateway_process.telemetry(inference["request_id"])
    finally:
        gateway_process.stop()

    # --- telemetry fields sourced from shadow_schema.json record_schema
    assert entry["shadow"] is True
    assert entry["request_id"] == inference["request_id"]
    assert entry["status"] == 200
    assert entry["primitive"] == "choice"
    assert entry["candidate_count"] == 2
    assert entry["route"] == "packed"
    assert entry["deferred"] is False
    assert entry["confidence"] == pytest.approx(0.93)
    assert entry["automation"] == "answer"  # raw selective field drift_monitor reads
    assert entry["config_hash"] == RELEASE_HASHES["release_manifest_sha256"]
    assert isinstance(entry["latency_ms"], int) and entry["latency_ms"] >= 0
    assert entry["model_version"] == "0.3.5"
    assert entry["error_code"] is None
    # G16 closed: the gateway captures the peak probability from the raw
    # model-service payload for the telemetry log; the browser response
    # contract never exposes it to the client.
    assert entry["max_probability"] == pytest.approx(0.9)
    assert "max_probability" not in inference
    timestamp = datetime.fromisoformat(entry["timestamp"])
    assert timestamp.tzinfo is not None

    # privacy: raw request text never reaches telemetry, also in shadow mode
    captured = json.dumps(entry)
    assert "Duplicate charge" not in captured
    assert "Payments and refunds" not in captured

    # --- join into the shadow_schema.json record shape
    record = {
        "timestamp": entry["timestamp"],
        "request_id": entry["request_id"],
        "model_hash": model_hash,
        "policy_hash": policy_hash,
        "config_hash": entry["config_hash"],
        "primitive": entry["primitive"],
        "prediction": {
            "decision": inference["decision"],
            "answer": inference["answer"],
            "probabilities": inference["probabilities"],
        },
        "confidence": entry["confidence"],
        "max_probability": entry["max_probability"],
        "automation": entry["automation"],
        "selective_decision": "accept",  # normalized view of automation == "answer"
        "latency_ms": entry["latency_ms"],
        "ground_truth": None,
        "correct": None,
        "slice_metadata": {
            "candidate_count": entry["candidate_count"],
            "route": entry["route"],
        },
        "slice_tag": "clean",
        "incumbent_outcome": None,
    }
    for field in (
        "timestamp",
        "request_id",
        "model_hash",
        "policy_hash",
        "config_hash",
        "primitive",
        "prediction",
        "confidence",
        "max_probability",
        "selective_decision",
        "latency_ms",
        "ground_truth",
        "slice_metadata",
        "incumbent_outcome",
    ):
        assert field in record
    assert record["ground_truth"] is None and record["incumbent_outcome"] is None

    # --- the joined, flattened record feeds research/veyabench/drift_monitor.py
    drift_record = _flatten_for_drift(
        record, ground_truth=inference["answer"], correct=True
    )
    report = drift_report([drift_record], period="week")
    assert report["format"] == "veya-drift-report-v1"
    stats = report["partitions"][report["first_period"]]
    assert stats["n"] == 1 and stats["accepted_n"] == 1 and stats["deferred_n"] == 0
    assert stats["coverage"] == 1.0
    assert stats["error_n"] == 0 and stats["risk"] == 0.0
    assert stats["candidate_count_distribution"] == {2: 1}
    assert stats["mean_latency_ms"] == pytest.approx(record["latency_ms"])
    assert stats["config_hashes"] == [RELEASE_HASHES["release_manifest_sha256"]]
    # One record cannot support the epsilon claim: the monitor must refuse it
    # (fail-closed), which the required-sample-size test below turns green.
    assert [alert["trigger"] for alert in drift_triggers(report, 0.02)] == [
        "risk_breach"
    ]


def test_non_shadow_gateway_leaves_no_shadow_markers(model_double):
    """Shadow mode off (ConfigMap value "0"): no header, no telemetry flag."""
    model_url = f"http://127.0.0.1:{model_double.server_address[1]}"
    gateway_process = GatewayProcess(model_url, shadow_mode="0")
    try:
        status, inference, headers = _post_choice(gateway_process.port)
        assert status == 200, inference
        assert "X-Veya-Shadow" not in headers
        entry = gateway_process.telemetry(inference["request_id"])
    finally:
        gateway_process.stop()
    assert "shadow" not in entry


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" true ", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("yes", False),
    ],
)
def test_shadow_flag_truth_table(monkeypatch, value, expected):
    monkeypatch.setattr(gateway, "VEYA_SHADOW_MODE", value)
    assert gateway._shadow_enabled() is expected


def test_schema_record_needs_flattening_before_drift_monitor():
    """The schema nests candidate_count/route under slice_metadata; the monitor
    reads them top-level, so the documented flatten transform is required."""
    record = _schema_record()

    nested = {**record, "ground_truth": "billing", "correct": True}
    stats = period_stats([nested])
    assert stats["candidate_count_distribution"] == {None: 1}
    assert stats["input_length_distribution"] is None

    flat = _flatten_for_drift(record, ground_truth="billing", correct=True)
    stats = period_stats([flat])
    assert stats["candidate_count_distribution"] == {2: 1}
    assert stats["input_length_distribution"]["p50"] == pytest.approx(2.0)
    assert stats["coverage"] == 1.0
    assert stats["config_hashes"] == [RELEASE_HASHES["release_manifest_sha256"]]


def test_release_change_mid_period_fires_config_mismatch():
    """A new release rolling out mid-period changes config_hash; the monitor alerts."""
    records = [_drift_record(i) for i in range(20)]
    records += [_drift_record(20 + i, config_hash="next-release") for i in range(20)]
    alerts = drift_triggers(drift_report(records, period="week"), epsilon=0.5)
    assert [alert["trigger"] for alert in alerts] == ["config_mismatch"]
    assert sorted(alerts[0]["hashes"]) == ["cfg", "next-release"]


def test_documented_sample_size_clears_risk_gate():
    """189 accepted zero-error decisions (protocol Section 7) clear epsilon=0.02;
    188 do not, and the monitor flags the shortfall instead of passing vacuously."""
    needed = required_n(epsilon=0.02, anticipated_coverage=1.0, max_errors=0)
    assert needed["accepted"] == 189

    records = [_drift_record(i) for i in range(189)]
    assert drift_triggers(drift_report(records, period="week"), 0.02) == []

    short = records[:-1]
    alerts = drift_triggers(drift_report(short, period="week"), 0.02)
    assert [alert["trigger"] for alert in alerts] == ["risk_breach"]
