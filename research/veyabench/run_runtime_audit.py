"""Clean-load runtime, API, language-gate, and latency audit for the release manifest."""
import json
import os
import platform
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from research.veyabench.run_phase3_eval import ROOT  # noqa: E402
from veya.release_bundle import INSTRUCTIONS, load_release_manifest  # noqa: E402
from veya.server import serve  # noqa: E402

ENTERED = time.perf_counter()


def percentiles(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "throughput_per_second": float(1000.0 / values.mean()),
    }


def benchmark(function, warmup=3, repeats=20):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return percentiles(samples)


def rss_mb():
    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return round(int(raw.strip()) / 1024.0, 1)


def request(port, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    manifest_path = os.path.join(ROOT, "artifacts", "release", "manifest.json")
    timings = {}
    load_started = time.perf_counter()
    bundle, manifest = load_release_manifest(manifest_path, device="cpu", root=ROOT, timings=timings)
    loaded_s = time.perf_counter() - load_started
    entered_to_loaded_s = time.perf_counter() - ENTERED
    labels = list(bundle.labels)
    criteria = dict(zip(labels, bundle.label_texts))
    small = {label: criteria[label] for label in labels[:4]}
    state = "The card transfer completed yesterday, but the money was returned today."
    noul = {"q": {"type": "noul", "instructions": "Was money returned?"}}
    score = {
        "q": {
            "type": "score",
            "instructions": "How urgent is the request?",
            "criteria": ["low", "medium", "high"],
        }
    }
    mixed = {
        "choice": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": small},
        "noul": noul["q"],
        "score": score["q"],
    }

    latency = {
        "small_packed_choice": benchmark(lambda: bundle.public_choice(state, INSTRUCTIONS, small)),
        "high_n_choice_77": benchmark(lambda: bundle.public_choice(state, INSTRUCTIONS, criteria)),
        "noul": benchmark(lambda: bundle.agent.system_one(state, noul)),
        "score_categorical": benchmark(lambda: bundle.agent.system_one(state, score)),
        "systemone_three_questions": benchmark(lambda: bundle.answer_questions(state, mixed)),
    }
    one = bundle.agent.system_one(state, noul)
    three = bundle.answer_questions(state, mixed)

    server = serve(bundle.agent, host="127.0.0.1", port=0, bundle=bundle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    http = {}
    try:
        http["health"] = request(port, "GET", "/health")
        http["version"] = request(port, "GET", "/version")
        http["choice_small"] = request(port, "POST", "/v1/choice", {
            "state": state, "instructions": INSTRUCTIONS, "criteria": small,
        })
        http["choice_high_n"] = request(port, "POST", "/v1/choice", {
            "state": state, "instructions": INSTRUCTIONS, "criteria": criteria,
        })
        http["noul"] = request(port, "POST", "/v1/noul", {
            "state": state, "instructions": "Was money returned?",
        })
        http["score"] = request(port, "POST", "/v1/score", {
            "state": state, "instructions": "How urgent?", "criteria": ["low", "medium", "high"],
        })
        http["systemone"] = request(port, "POST", "/v1/systemone", {
            "state": state, "questions": mixed,
        })
        non_english = "Je veux un remboursement pour ce paiement qui a été débité deux fois."
        for name, path, payload in (
            ("unsupported_choice", "/v1/choice", {"state": non_english, "instructions": INSTRUCTIONS, "criteria": small}),
            ("unsupported_noul", "/v1/noul", {"state": non_english, "instructions": "Was money returned?"}),
            ("unsupported_score", "/v1/score", {"state": non_english, "instructions": "How urgent?", "criteria": ["low", "high"]}),
            ("unsupported_systemone", "/v1/systemone", {"state": non_english, "questions": mixed}),
        ):
            http[name] = request(port, "POST", path, payload)
    finally:
        server.shutdown()

    version = http["version"][1]
    manifest_match = {
        "release": version.get("release") == manifest["release"],
        "manifest_sha256": version.get("release_manifest_sha256") == bundle.manifest_sha256,
        "checkpoint_sha256": version.get("checkpoint_sha256") == manifest["base_checkpoint_hash"],
        "shortlist_sha256": version.get("shortlist_sha256") == manifest["choice"]["shortlist_hash"],
        "router_version": version.get("choice_router_version") == bundle.router["version"],
        "scorer_id": version.get("reranker_id") == manifest["served_scorer_id"],
        "policy_sha256": version.get("selective_policy_sha256") == manifest["selective_policy_hash"],
    }
    report = {
        "format": "veya-runtime-audit-v1",
        "release": manifest["release"],
        "manifest_sha256": bundle.manifest_sha256,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "device": str(bundle.agent.device),
        },
        "cold_start": {
            "module_enter_to_loaded_s": round(entered_to_loaded_s, 3),
            "manifest_load_s": round(loaded_s, 3),
            "stages_s": {key: round(value, 3) for key, value in timings.items()},
            "rss_mb_after_benchmark": rss_mb(),
        },
        "latency": latency,
        "multi_question_redundancy": {
            "one_question_input_tokens": one["usage"]["input_tokens"],
            "three_question_input_tokens": three["usage"]["input_tokens"],
            "ratio": three["usage"]["input_tokens"] / max(1, one["usage"]["input_tokens"]),
            "request_scoped_state_cache": False,
            "reason": "State representations are question-conditioned; reuse would change attention semantics.",
        },
        "http_status": {name: row[0] for name, row in http.items()},
        "http_routes": {
            "choice_small": http["choice_small"][1].get("route"),
            "choice_high_n": http["choice_high_n"][1].get("route"),
        },
        "unsupported_language_reasons": {
            name: row[1].get("defer_reason", row[1].get("reason"))
            for name, row in http.items() if name.startswith("unsupported_")
        },
        "version_matches_manifest": manifest_match,
        "all_version_fields_match": all(manifest_match.values()),
    }
    output = os.path.join(ROOT, "artifacts", "release", "runtime_audit.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
