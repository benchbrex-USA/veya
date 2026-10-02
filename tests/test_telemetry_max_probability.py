"""Focused tests for the telemetry max_probability capture (G16 residual gap).

Shadow telemetry must log ``max_probability`` so selective-risk and drift
monitoring work (research/veyabench/drift_monitor.py confidence
distributions), while the client-facing HTTP response must never expose the
internal score: web/contracts.py's response whitelist strips it by design.

The gateway therefore captures the peak probability from the raw
model-service payload (web/server.py _telemetry_max_probability) before the
whitelist is applied — the audited bundle serves the field directly
(veya/release_bundle.py public_choice); plain agent responses omit it, so it
is derived from the served distribution the same way the bundle computes it.

Run: .venv/bin/python -m pytest tests/test_telemetry_max_probability.py -v
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.veyabench.drift_monitor import (  # noqa: E402
    drift_report,
    drift_triggers,
    partition_by_time,
    period_stats,
)

from web import server as gateway  # noqa: E402

CHOICE = {
    "primitive": "choice",
    "state": "Duplicate charge. Please refund.",
    "instructions": "Which team?",
    "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
}
# Exactly the shape veya/release_bundle.py public_choice serves for an
# answered packed choice: the explicit max_probability is upstream data the
# browser contract strips but telemetry must keep.
BUNDLE_CHOICE = {
    "model": "veya-rl-agent",
    "decision": "answer",
    "deferred": False,
    "answer": "billing",
    "probabilities": {"billing": 0.87, "technical": 0.13},
    "route": "packed",
    "confidence": 0.74,
    "max_probability": 0.87,
    "automation": "answer",
    "calibration": "test",
    "calibration_version": "test",
    "candidate_count": 2,
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


@pytest.fixture
def http_pair(monkeypatch):
    """Website gateway in front of a model-service double with a mutable response."""
    state = {"response": BUNDLE_CHOICE}

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
            response = state["response"]
            if callable(response):
                response = response(self.path, body)
            code, payload = (
                (500, {"detail": "model exploded"})
                if isinstance(response, int)
                else (200, response)
            )
            self.reply(code, payload)

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
    yield state, web.server_address[1]
    web.shutdown()
    web.server_close()
    model.shutdown()
    model.server_close()


@pytest.fixture
def telemetry(monkeypatch):
    """Capture the gateway's telemetry log lines (logger veya.web)."""
    records = []
    monkeypatch.setattr(gateway.logger, "info", lambda entry: records.append(entry))
    return records


def call(port, payload):
    req = Request(
        f"http://127.0.0.1:{port}/api/infer",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def last_entry(records):
    assert records, "no telemetry entry was logged"
    return json.loads(records[-1])


def test_max_probability_reaches_telemetry_not_client(http_pair, telemetry):
    """The served score is captured for telemetry; clients never see it."""
    _, port = http_pair
    status, out, _ = call(port, CHOICE)
    assert status == 200 and out["answer"] == "billing"
    entry = last_entry(telemetry)
    assert entry["request_id"] == out["request_id"]
    assert entry["status"] == 200
    assert entry["confidence"] == pytest.approx(0.74)
    value = entry["max_probability"]
    assert value is not None, "max_probability must not log null on answered choice"
    assert isinstance(value, float) and not isinstance(value, bool)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(0.87)
    # The browser contract still strips the internal score.
    assert "max_probability" not in out


def test_max_probability_derived_from_distribution_when_upstream_omits_it(http_pair, telemetry):
    """Plain agent responses (no explicit field) derive the peak like the bundle."""
    state, port = http_pair
    state["response"] = {
        key: value
        for key, value in BUNDLE_CHOICE.items()
        if key not in ("max_probability", "confidence", "automation", "route")
    }
    status, out, _ = call(port, CHOICE)
    assert status == 200
    entry = last_entry(telemetry)
    assert entry["max_probability"] == pytest.approx(0.87)
    assert "max_probability" not in out


def test_score_peak_comes_from_levels_not_the_expected_answer(http_pair, telemetry):
    """Score answers are expected values, not probabilities; the peak must come
    from the level distribution (1.5 for three levels must not be misread)."""
    state, port = http_pair
    state["response"] = {
        "model": "veya-rl-agent",
        "decision": "answer",
        "answer": 1.5,
        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
    }
    status, out, _ = call(
        port,
        {
            "primitive": "score",
            "state": "text",
            "instructions": "How much?",
            "criteria": ["Low", "Medium", "High"],
        },
    )
    assert status == 200
    entry = last_entry(telemetry)
    assert entry["primitive"] == "score"
    assert entry["max_probability"] == pytest.approx(0.7)
    assert "max_probability" not in out


@pytest.mark.parametrize(
    ("answer", "expected_peak"),
    [(0.7, 0.7), (0.2, 0.8), (1.0, 1.0)],
)
def test_noul_peak_from_binary_distribution(http_pair, telemetry, answer, expected_peak):
    """Noul answers carry P(yes); the implied binary peak is max(p, 1 - p)."""
    state, port = http_pair
    state["response"] = {
        "model": "veya-rl-agent",
        "decision": "answer",
        "answer": answer,
        "confidence": expected_peak,
    }
    status, out, _ = call(
        port, {"primitive": "noul", "state": "text", "instructions": "Is true?"}
    )
    assert status == 200
    entry = last_entry(telemetry)
    assert entry["max_probability"] == pytest.approx(expected_peak)
    assert "max_probability" not in out


def test_defer_and_error_paths_log_null(http_pair, telemetry):
    """No distribution to report: defer and upstream failures log null."""
    state, port = http_pair
    state["response"] = {
        "model": "veya-rl-agent",
        "decision": "defer",
        "deferred": True,
        "defer_reason": "unsupported_language",
        "answer": None,
        "probabilities": None,
        "max_probability": None,
    }
    status, out, _ = call(port, CHOICE)
    assert status == 200 and out["deferred"]
    assert last_entry(telemetry)["max_probability"] is None

    state["response"] = 500  # upstream failure
    status, out, _ = call(port, CHOICE)
    assert status == 502 and out["code"] == "model_error"
    entry = last_entry(telemetry)
    assert entry["error_code"] == "model_error"
    assert entry["max_probability"] is None


@pytest.mark.parametrize("bad", [1.5, -0.1, "high", True, None])
def test_invalid_upstream_max_probability_falls_back_to_distribution(
    http_pair, telemetry, bad
):
    """A malformed score never breaks serving; the peak is derived instead."""
    state, port = http_pair
    state["response"] = {
        **{
            key: value
            for key, value in BUNDLE_CHOICE.items()
            if key not in ("max_probability", "confidence", "automation")
        },
        "max_probability": bad,
    }
    status, out, _ = call(port, CHOICE)
    assert status == 200
    entry = last_entry(telemetry)
    assert entry["max_probability"] == pytest.approx(0.87)
    assert 0.0 <= entry["max_probability"] <= 1.0
    assert "max_probability" not in out


def test_systemone_has_no_single_top_level_peak(http_pair, telemetry):
    """Multi-question requests mix distributions; telemetry stays null."""
    state, port = http_pair
    state["response"] = {
        "model": "veya-rl-agent",
        "decision": "answer",
        "answers": {
            "z": {"type": "noul", "noul": 0.6},
            "a": {"type": "noul", "noul": 0.6},
        },
        "routes": {"z": "noul", "a": "noul"},
    }
    status, out, _ = call(
        port,
        {
            "primitive": "systemone",
            "state": "text",
            "questions": {
                "z": {"type": "noul", "instructions": "First?"},
                "a": {"type": "noul", "instructions": "Second?"},
            },
        },
    )
    assert status == 200
    entry = last_entry(telemetry)
    assert entry["primitive"] == "systemone"
    assert entry["max_probability"] is None
    assert "max_probability" not in out


def test_telemetry_with_score_still_carries_no_raw_input(http_pair, telemetry):
    """max_probability is a model confidence score, not PII: raw request text
    (state or candidate descriptions) still never reaches the log."""
    state, port = http_pair
    sentinel = "TELEMETRY-SENTINEL-状态"
    payload = {
        **CHOICE,
        "state": sentinel,
        "criteria": {"billing": sentinel, "technical": "Bugs and outages"},
    }
    status, _, _ = call(port, payload)
    assert status == 200
    assert last_entry(telemetry)["max_probability"] == pytest.approx(0.87)
    assert sentinel not in "".join(telemetry)


def test_telemetry_record_feeds_drift_monitor(http_pair, telemetry):
    """A real telemetry line (max_probability populated) joins into a shadow
    record that drift_monitor consumes unchanged — field names must match."""
    state, port = http_pair
    state["response"] = BUNDLE_CHOICE
    status, out, _ = call(port, CHOICE)
    assert status == 200
    entry = last_entry(telemetry)

    # Labeling join per artifacts/release/shadow_schema.json: the telemetry
    # line plus the eventual ground truth / correct fields.
    record = {**entry, "ground_truth": "billing", "correct": True}
    for field in (
        "timestamp",
        "request_id",
        "primitive",
        "candidate_count",
        "route",
        "deferred",
        "status",
        "latency_ms",
        "model_version",
        "error_code",
        "confidence",
        "max_probability",
        "automation",
        "config_hash",
    ):
        assert field in record

    # Records pass through the monitor's partitioning verbatim.
    buckets = partition_by_time([record])
    (key,) = buckets.keys()
    assert buckets[key][0]["max_probability"] == pytest.approx(0.87)

    stats = period_stats([record])
    assert stats["n"] == 1 and stats["accepted_n"] == 1
    assert stats["coverage"] == 1.0 and stats["risk"] == 0.0
    assert sum(stats["confidence_distribution"]) == 1

    report = drift_report([record], period="week")
    # One record cannot support an epsilon claim: fail-closed risk breach.
    alerts = drift_triggers(report, epsilon=0.02)
    assert [alert["trigger"] for alert in alerts] == ["risk_breach"]
