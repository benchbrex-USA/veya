"""Gateway concurrency-model tests (G10): real HTTP against a slow model double.

Verifies the MAX_INFLIGHT=2 admission semaphore, the 503+Retry-After overload
response, sequential traffic, and response integrity under concurrent access.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web import server as gateway

SLOW_S = 0.3  # long enough that 3 simultaneous requests overlap on admission


def _choice(tag: str) -> dict:
    return {
        "primitive": "choice",
        "state": f"State {tag}",
        "instructions": "Which team?",
        "criteria": {f"{tag}-billing": "Payments", f"{tag}-technical": "Bugs"},
    }


@pytest.fixture
def concurrent_pair(monkeypatch):
    """Gateway + model double that sleeps on /v1/choice to widen the race."""
    observed = {"concurrent_upstream": 0, "max_upstream": 0, "lock": threading.Lock()}

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
                self.reply(200, {"model_version": "0.3.5"})
            else:
                self.reply(404, {"code": "not_found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            labels = list(body["criteria"])
            with observed["lock"]:
                observed["concurrent_upstream"] += 1
                observed["max_upstream"] = max(
                    observed["max_upstream"], observed["concurrent_upstream"]
                )
            time.sleep(SLOW_S)
            with observed["lock"]:
                observed["concurrent_upstream"] -= 1
            # First criterion label echoes the request tag: proves no mixing.
            self.reply(
                200,
                {
                    "model": "veya-rl-agent",
                    "decision": "answer",
                    "deferred": False,
                    "answer": labels[0],
                    "probabilities": {labels[0]: 1.0, labels[1]: 0.0},
                    "route": "packed",
                    "calibration_version": "test",
                    "candidate_count": 2,
                },
            )

    model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        gateway, "UPSTREAM", f"http://127.0.0.1:{model.server_address[1]}"
    )
    monkeypatch.setattr(gateway, "version_cache", (0, {}))
    monkeypatch.setattr(gateway, "CAPACITY", 12)
    monkeypatch.setattr(gateway, "INFER_TIMEOUT_S", 20)
    monkeypatch.setattr(gateway, "_SHUTTING_DOWN", False)
    gateway.rate_hits.clear()
    # Fresh admission budget for every test, regardless of import-time state.
    fresh = threading.BoundedSemaphore(gateway.MAX_INFLIGHT)
    monkeypatch.setattr(gateway, "inflight", fresh)

    class Web(ThreadingHTTPServer):
        # Default listen backlog is 5; the mixed-response test opens more
        # simultaneous connections than that.
        request_queue_size = 32

    web = Web(("127.0.0.1", 0), gateway.Handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    yield {"port": web.server_address[1], "observed": observed}
    web.shutdown()
    web.server_close()
    model.shutdown()
    model.server_close()


def call(port, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{port}/api/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response), response.headers
    except HTTPError as exc:
        return exc.code, json.load(exc), exc.headers


def test_two_concurrent_requests_both_succeed(concurrent_pair):
    port = concurrent_pair["port"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: call(port, _choice(f"a{i}")), range(2)))
    for status, out, _ in results:
        assert status == 200, out
        assert out["deferred"] is False and out["answer"]


def test_third_concurrent_request_gets_503(concurrent_pair):
    port = concurrent_pair["port"]
    barrier = threading.Barrier(3)

    def fire(i):
        barrier.wait()  # maximize overlap on the admission check
        return call(port, _choice(f"b{i}"))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fire, range(3)))
    statuses = sorted(status for status, _, _ in results)
    assert statuses == [200, 200, 503]
    busy = next(out for status, out, _ in results if status == 503)
    assert busy["code"] == "model_busy"
    assert busy["error_code"] == "OVERLOADED"
    assert busy["retry_after"] == 5
    # The two admitted requests really overlapped; the third never reached the model.
    assert concurrent_pair["observed"]["max_upstream"] <= gateway.MAX_INFLIGHT


def test_overload_response_includes_retry_after_header(concurrent_pair):
    port = concurrent_pair["port"]
    barrier = threading.Barrier(3)

    def fire(i):
        barrier.wait()
        return call(port, _choice(f"c{i}"))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fire, range(3)))
    rejected = [headers for status, _, headers in results if status == 503]
    assert rejected, "expected at least one overload rejection"
    for headers in rejected:
        assert int(headers["Retry-After"]) == 5


def test_sequential_requests_work_normally(concurrent_pair):
    port = concurrent_pair["port"]
    for i in range(4):
        status, out, _ = call(port, _choice(f"s{i}"))
        assert status == 200
        assert out["answer"] == f"s{i}-billing"


def test_concurrent_responses_are_never_mixed(concurrent_pair):
    port = concurrent_pair["port"]
    n = gateway.MAX_INFLIGHT * 3
    barrier = threading.Barrier(n)

    def fire(i):
        barrier.wait()
        return i, call(port, _choice(f"t{i}"))

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(fire, range(n)))
    answers = {}
    overloaded = 0
    for i, (status, out, _) in results:
        if status == 503:
            overloaded += 1
            assert out["code"] == "model_busy"
            continue
        assert status == 200
        expected = f"t{i}-billing"
        assert out["answer"] == expected, "response crossed between requests"
        answers[i] = expected
    assert overloaded < n, "not a single request succeeded"
    # Every distinct request tag appears exactly once across the successes.
    assert len(set(answers.values())) == len(answers)
    assert set(answers.values()) == {f"t{i}-billing" for i in answers}
