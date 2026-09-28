"""Clean-process release evaluation.

Loads every component from the release manifest. Does not train, does not fit a
temperature, and does not read Banking77 rows that chose K or the temperature.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ENTER_S = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.veyabench.run_phase3_eval import ROOT, SEED, choice_logits, load_ag_news, load_banking, metrics, subset_criteria  # noqa: E402
from veya.common import ece_score  # noqa: E402
from veya.release_bundle import (  # noqa: E402
    INSTRUCTIONS,
    load_release_manifest,
    recommend_packed_cutoff,
    resolve_manifest_path,
)
from veya.selective import apply_report, risk_coverage_curve, score_probabilities, select_threshold  # noqa: E402
from veya.server import serve  # noqa: E402

GRID = [2, 4, 8, 12, 16, 20, 32, 50, 77]
POLICY_EPS = ("0.02", "0.05", "0.10")


def split_hash(rows):
    digest = hashlib.sha256()
    for text, label in rows:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def softmax_row(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def percentiles(samples):
    arr = np.asarray(samples, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "n": int(len(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def rss_mb():
    try:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return round(int(raw.strip()) / 1024.0, 1)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def rank_indices(bundle, texts, label_texts, k=8):
    queries = bundle.project(bundle.embed(list(texts)))
    labels = bundle.project(bundle.embed(list(label_texts)))
    scores = queries @ labels.T
    index = torch.tensor(scores).topk(k, dim=1).indices.tolist()
    return index, scores


def score_shortlist(bundle, rows, labels, crit, temperature):
    texts = [row[0] for row in rows]
    started = time.perf_counter()
    top, scores = rank_indices(bundle, texts, [crit[label] for label in labels], 8)
    records = []
    for row_i, ((text, truth), picked_i) in enumerate(zip(rows, top)):
        picked = [labels[j] for j in picked_i]
        subset = {label: crit[label] for label in picked}
        logits, n_tokens = _logits(bundle, text, subset)
        hit = truth in picked
        gold = picked.index(truth) if hit else None
        probs = softmax_row(logits, temperature)
        pred = picked[int(np.argmax(probs))]
        nll = -math.log(float(probs[gold]) + 1e-12) if gold is not None else None
        if gold is None:
            brier = None
        else:
            truth_vec = np.zeros_like(probs)
            truth_vec[gold] = 1.0
            brier = float(np.sum((probs - truth_vec) ** 2))
        records.append({
            "hit": hit,
            "correct": pred == truth,
            "prediction": pred,
            "truth": truth,
            "shortlist": picked,
            "max_probability": float(probs.max()),
            "neg_entropy": score_probabilities(probs, "neg_entropy"),
            "margin": score_probabilities(probs, "margin"),
            "nll": nll,
            "brier": brier,
            "input_tokens": n_tokens,
            "logits": logits,
        })
        if row_i and row_i % 100 == 0:
            print("  shortlist %d/%d" % (row_i, len(rows)), flush=True)
    elapsed = time.perf_counter() - started
    return records, scores, elapsed


def _logits(bundle, text, criteria):
    from veya.release_bundle import raw_choice_logits
    return raw_choice_logits(bundle.agent, text, INSTRUCTIONS, criteria)


def rescore(records, temperature):
    """Apply a temperature to stored logits. Ranking uses the new probabilities."""
    out = []
    for row in records:
        probs = softmax_row(row["logits"], temperature)
        picked = row["shortlist"]
        pred = picked[int(np.argmax(probs))]
        gold = picked.index(row["truth"]) if row["hit"] else None
        nll = -math.log(float(probs[gold]) + 1e-12) if gold is not None else None
        if gold is None:
            brier = None
        else:
            truth_vec = np.zeros_like(probs)
            truth_vec[gold] = 1.0
            brier = float(np.sum((probs - truth_vec) ** 2))
        copied = dict(row)
        copied.update({
            "correct": pred == row["truth"],
            "prediction": pred,
            "max_probability": float(probs.max()),
            "neg_entropy": score_probabilities(probs, "neg_entropy"),
            "margin": score_probabilities(probs, "margin"),
            "nll": nll,
            "brier": brier,
        })
        out.append(copied)
    return out


def aggregate(records, elapsed=None):
    correct = np.array([1.0 if row["correct"] else 0.0 for row in records], dtype=np.float64)
    honest = np.array([row["max_probability"] for row in records], dtype=np.float64)
    zeroed = np.array([0.0 if not row["hit"] else row["max_probability"] for row in records], dtype=np.float64)
    hits = [row for row in records if row["hit"]]
    nlls = [row["nll"] for row in hits if row["nll"] is not None]
    briers = [row["brier"] for row in hits if row["brier"] is not None]
    rerank_ok = sum(1 for row in hits if row["correct"])
    return {
        "n": len(records),
        "recall": (len(hits) / len(records)) if records else None,
        "retrieval_failures": len(records) - len(hits),
        "rerank_accuracy_given_hit": (rerank_ok / len(hits)) if hits else None,
        "end_to_end_accuracy": float(correct.mean()) if len(records) else None,
        "nll_on_hits": float(np.mean(nlls)) if nlls else None,
        "brier_on_hits": float(np.mean(briers)) if briers else None,
        "ece_e2e_misses_zeroed": ece_score(zeroed, correct) if len(records) else None,
        "ece_e2e_honest": ece_score(honest, correct) if len(records) else None,
        "mean_max_probability": float(honest.mean()) if len(records) else None,
        "mean_max_probability_correct": float(honest[correct == 1].mean()) if np.any(correct == 1) else None,
        "mean_max_probability_incorrect": float(honest[correct == 0].mean()) if np.any(correct == 0) else None,
        "mean_input_tokens": float(np.mean([row["input_tokens"] for row in records])) if records else None,
        "seconds": None if elapsed is None else round(elapsed, 3),
    }


def decompose(records):
    retrieval_miss = sum(1 for row in records if not row["hit"])
    rerank_miss = sum(1 for row in records if row["hit"] and not row["correct"])
    correct = sum(1 for row in records if row["correct"])
    overconfident = sum(1 for row in records if (not row["correct"]) and row["max_probability"] >= 0.7)
    n = len(records)
    errors = retrieval_miss + rerank_miss
    return {
        "n": n,
        "correct": correct,
        "retrieval_miss": retrieval_miss,
        "reranker_miss_target_present": rerank_miss,
        "calibration_overconfidence_maxprob_ge_0.7": overconfident,
        "routing_mistake": 0,
        "invalid_or_ambiguous_label": 0,
        "error_share_retrieval": (retrieval_miss / errors) if errors else None,
        "error_share_reranker": (rerank_miss / errors) if errors else None,
    }


def score_vectors(records, kind):
    return np.array([row[kind] for row in records], dtype=np.float64)


def correct_vector(records):
    return np.array([1.0 if row["correct"] else 0.0 for row in records], dtype=np.float64)


def fit_policy(records):
    correct = correct_vector(records)
    kinds = ("max_probability", "neg_entropy", "margin")
    compared = {}
    for kind in kinds:
        scores = score_vectors(records, kind)
        thresholds = {eps: select_threshold(scores, correct, float(eps)) for eps in POLICY_EPS}
        compared[kind] = {
            "thresholds": thresholds,
            "validation": {eps: apply_report(scores, correct, thresholds[eps]) for eps in POLICY_EPS},
        }
    winner = "max_probability"
    best = compared[winner]["validation"]["0.05"]["coverage"] or 0.0
    for kind in kinds[1:]:
        coverage = compared[kind]["validation"]["0.05"]["coverage"] or 0.0
        if coverage > best + 0.02:
            winner = kind
            best = coverage
    return winner, compared


def apply_policy(records, kind, thresholds):
    scores = score_vectors(records, kind)
    correct = correct_vector(records)
    return {eps: apply_report(scores, correct, thresholds[eps]) for eps in POLICY_EPS}


def packed_records(agent, rows, labels, crit, temperature, batch_size=4):
    texts = [row[0] for row in rows]
    parts = []
    started = time.perf_counter()
    try:
        for start in range(0, len(texts), batch_size):
            parts.append(choice_logits(agent, texts[start:start + batch_size], crit))
    except RuntimeError as exc:
        if batch_size == 1 or "memory" not in str(exc).lower():
            raise
        print("packed batch %d ran out of memory; retrying one at a time" % batch_size, flush=True)
        return packed_records(agent, rows, labels, crit, temperature, batch_size=1)
    logits = np.concatenate(parts, axis=0)
    elapsed = time.perf_counter() - started
    index = {label: i for i, label in enumerate(labels)}
    records = []
    for row, raw in zip(rows, logits):
        probs = softmax_row(raw, temperature)
        pred = labels[int(np.argmax(probs))]
        gold = index[row[1]]
        truth_vec = np.zeros_like(probs)
        truth_vec[gold] = 1.0
        records.append({
            "correct": pred == row[1],
            "max_probability": float(probs.max()),
            "nll": -math.log(float(probs[gold]) + 1e-12),
            "brier": float(np.sum((probs - truth_vec) ** 2)),
        })
    return records, elapsed, logits


def packed_aggregate(records, elapsed):
    correct = np.array([1.0 if row["correct"] else 0.0 for row in records])
    conf = np.array([row["max_probability"] for row in records])
    return {
        "n": len(records),
        "accuracy": float(correct.mean()),
        "nll": float(np.mean([row["nll"] for row in records])),
        "brier": float(np.mean([row["brier"] for row in records])),
        "ece": ece_score(conf, correct),
        "mean_max_probability": float(conf.mean()),
        "seconds": round(elapsed, 3),
    }


def crossover(bundle, rows, labels):
    route_t = bundle.route_temperature
    table = []
    for n in GRID:
        packed_ok = []
        hier_ok = []
        packed_s = []
        hier_s = []
        packed_tokens = []
        hier_tokens = []
        print("  crossover N=%d" % n, flush=True)
        for i, (text, truth) in enumerate(rows):
            if n >= len(labels):
                crit = {label: text_value for label, text_value in zip(labels, bundle.label_texts)}
            else:
                rng = random.Random(SEED + n * 10007 + i)
                crit, _gold = subset_criteria(labels, truth, n, rng)
            started = time.perf_counter()
            logits, n_tokens = _logits(bundle, text, crit)
            packed_s.append(time.perf_counter() - started)
            packed_tokens.append(n_tokens)
            temperature = bundle.temperature_for("packed", len(crit))
            pred = list(crit.keys())[int(np.argmax(softmax_row(logits, temperature)))]
            packed_ok.append(pred == truth)
            items = list(crit.items())
            started = time.perf_counter()
            if len(items) > 8:
                picked, _stage = bundle.shortlist(text, items, 8)
            else:
                picked = [key for key, _value in items]
            subset = {key: crit[key] for key in picked}
            h_logits, h_tokens = _logits(bundle, text, subset)
            hier_s.append(time.perf_counter() - started)
            hier_tokens.append(h_tokens)
            h_pred = picked[int(np.argmax(softmax_row(h_logits, route_t)))]
            hier_ok.append(h_pred == truth)
        table.append({
            "n_candidates": n,
            "examples": len(rows),
            "packed_accuracy": float(np.mean(packed_ok)),
            "hierarchical_accuracy": float(np.mean(hier_ok)),
            "packed_seconds": round(float(np.sum(packed_s)), 3),
            "hierarchical_seconds": round(float(np.sum(hier_s)), 3),
            "packed_mean_input_tokens": float(np.mean(packed_tokens)),
            "hierarchical_mean_rerank_tokens": float(np.mean(hier_tokens)),
        })
    gaps = {row["n_candidates"]: row["hierarchical_accuracy"] - row["packed_accuracy"] for row in table}
    cutoff, reason = recommend_packed_cutoff(gaps, GRID, floor=0.05, current=32)
    return {"rows": [296, 400], "table": table, "recommended_packed_max": cutoff, "reason": reason}


def latency_profile(bundle, sample_state, labels, crit):
    small = {label: crit[label] for label in labels[:4]}
    wide = crit

    def bench(fn, warmup, repeats):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
        return percentiles(samples)

    bundle._cache.clear()
    cold = bundle.score_choice(sample_state, INSTRUCTIONS, wide, timings=True)
    warm_stages = []
    for _ in range(20):
        warm_stages.append(bundle.score_choice(sample_state, INSTRUCTIONS, wide, timings=True)["stages_s"])
    stage_means = {}
    for key in warm_stages[0]:
        stage_means[key] = float(np.mean([row[key] for row in warm_stages])) * 1000.0
    noul_q = {"q": {"type": "noul", "instructions": "Does the customer report a charge?"}}
    score_q = {"q": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}}
    mixed = {
        "choice": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": small},
        "noul": {"type": "noul", "instructions": "Does the customer report a charge?"},
        "score": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
    }
    one = bundle.agent.system_one(sample_state, noul_q)
    three = bundle.agent.system_one(sample_state, {
        "a": noul_q["q"],
        "b": score_q["q"],
        "c": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": small},
    })
    return {
        "warmup": 4,
        "repeats": 30,
        "packed_small_ms": bench(lambda: bundle.score_choice(sample_state, INSTRUCTIONS, small), 4, 30),
        "hierarchical_77_ms": bench(lambda: bundle.score_choice(sample_state, INSTRUCTIONS, wide), 4, 30),
        "noul_ms": bench(lambda: bundle.agent.system_one(sample_state, noul_q), 4, 30),
        "score_ms": bench(lambda: bundle.agent.system_one(sample_state, score_q), 4, 30),
        "systemone_mixed_ms": bench(lambda: bundle.answer_questions(sample_state, mixed), 4, 30),
        "hierarchical_cold_stages_ms": {key: value * 1000.0 for key, value in cold["stages_s"].items()},
        "hierarchical_warm_stage_mean_ms": stage_means,
        "state_reuse": {
            "one_question_input_tokens": one["usage"]["input_tokens"],
            "three_question_input_tokens": three["usage"]["input_tokens"],
            "token_ratio": three["usage"]["input_tokens"] / max(1, one["usage"]["input_tokens"]),
            "implemented_cache": False,
            "reason": "Each question is a separate sequence, and the state tokens attend to that question. A hidden-state cache would change the forward.",
        },
    }


def _call(port, method, path, body=None, raw=None):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
    request = Request(url, data=data, method=method)
    if data is not None and raw is None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        payload = exc.read().decode("utf-8")
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"raw": payload}
        return exc.code, payload


def http_suite(bundle, labels, crit):
    server = serve(bundle.agent, host="127.0.0.1", port=0, bundle=bundle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    cases = []

    def record(name, status, payload, expect):
        ok = status == expect
        cases.append({"name": name, "status": status, "expect": expect, "ok": ok, "body": payload})
        print("  http", name, status, "ok" if ok else "FAIL", flush=True)

    try:
        status, body = _call(port, "GET", "/health")
        record("health", status, body, 200)
        status, body = _call(port, "GET", "/version")
        record("version", status, body, 200)
        version_ok = (
            body.get("checkpoint_sha256") == bundle.checkpoint_hash
            and body.get("shortlist_sha256") == bundle.shortlist_sha256
            and abs(float(body.get("route_calibration_temperature")) - bundle.route_temperature) < 1e-9
            and body.get("reranker_id") == getattr(bundle, "reranker_id", None)
            and "/Volumes" not in json.dumps(body)
        )
        cases.append({"name": "version_matches_loaded_artifacts", "ok": version_ok, "body": body})
        small = {label: crit[label] for label in labels[:4]}
        status, body = _call(port, "POST", "/v1/choice", {"state": "Where is my card?", "instructions": INSTRUCTIONS, "criteria": small})
        record("small_choice", status, body, 200)
        cases.append({"name": "small_choice_route", "ok": body.get("route") == "packed", "body": body.get("route")})
        status, body = _call(port, "POST", "/v1/choice", {"state": "I was charged twice for the transfer", "instructions": INSTRUCTIONS, "criteria": crit})
        record("large_choice", status, body, 200)
        cases.append({
            "name": "large_choice_route",
            "ok": body.get("route") == "shortlist_k8_packed_rerank" and isinstance(body.get("shortlist"), list) and len(body.get("shortlist")) == 8,
            "body": {"route": body.get("route"), "shortlist_n": len(body.get("shortlist") or [])},
        })
        status, body = _call(port, "POST", "/v1/noul", {"state": "I was charged twice", "instructions": "Was the customer charged twice?"})
        record("noul", status, body, 200)
        status, body = _call(port, "POST", "/v1/score", {"state": "I was charged twice", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]})
        record("score", status, body, 200)
        status, body = _call(port, "POST", "/v1/systemone", {
            "state": "I was charged twice",
            "questions": {
                "n": {"type": "noul", "instructions": "Was the customer charged twice?"},
                "s": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
                "c": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": small},
            },
        })
        record("systemone_mixed", status, body, 200)
        status, body = _call(port, "POST", "/v1/systemone", {"state": "hello", "questions": {}})
        record("empty_questions", status, body, 400)
        cases.append({"name": "empty_questions_reason", "ok": body.get("defer_reason") == "invalid_schema", "body": body.get("defer_reason")})
        status, body = _call(port, "POST", "/v1/choice", raw=b"{")
        record("malformed_json", status, body, 400)
        status, body = _call(port, "POST", "/v1/choice", {"state": "hello"})
        record("missing_instructions", status, body, 400)
        status, body = _call(port, "POST", "/v1/choice", {
            "state": "मुझे अपने खाते से पैसे वापस चाहिए",
            "instructions": INSTRUCTIONS,
            "criteria": small,
        })
        record("unsupported_language", status, body, 200)
        cases.append({"name": "unsupported_language_reason", "ok": body.get("defer_reason") == "unsupported_language", "body": body.get("defer_reason")})
        status, body = _call(port, "POST", "/v1/nope", {"state": "hello", "instructions": "x"})
        record("invalid_primitive", status, body, 404)
        status, body = _call(port, "POST", "/v1/systemone", {
            "state": "hello",
            "questions": {"q": {"type": "madeup", "instructions": "x"}},
        })
        record("invalid_question_type", status, body, 400)
        status, body = _call(port, "POST", "/v1/choice", {
            "state": "hello",
            "instructions": INSTRUCTIONS,
            "criteria": ["billing", "billing", "sales"],
        })
        record("duplicate_candidates", status, body, 400)
        cases.append({"name": "duplicate_candidates_rejected", "ok": body.get("defer_reason") == "invalid_schema", "body": body.get("defer_reason")})
        wide = {"opt-%d" % i: "option number %d" % i for i in range(120)}
        status, body = _call(port, "POST", "/v1/choice", {"state": "hello there", "instructions": INSTRUCTIONS, "criteria": wide})
        record("very_large_choice", status, body, 200)
        cases.append({"name": "very_large_uses_shortlist", "ok": body.get("route") == "shortlist_k8_packed_rerank", "body": body.get("route")})
        status, body = _call(port, "POST", "/v1/choice", {"state": "", "instructions": INSTRUCTIONS, "criteria": small})
        record("empty_state", status, body, 400)
        conc = {}
        payload = {"state": "Where is my card?", "instructions": INSTRUCTIONS, "criteria": small}
        for workers in (1, 2, 4, 8):
            errors = []
            latencies = []
            lock = threading.Lock()

            def work(repeats=4):
                local = []
                local_err = 0
                for _ in range(repeats):
                    started = time.perf_counter()
                    try:
                        status_i, body_i = _call(port, "POST", "/v1/choice", payload)
                        if status_i != 200 or body_i.get("route") != "packed":
                            local_err += 1
                    except Exception:
                        local_err += 1
                    local.append((time.perf_counter() - started) * 1000.0)
                with lock:
                    latencies.extend(local)
                    errors.append(local_err)

            threads = [threading.Thread(target=work) for _ in range(workers)]
            wall0 = time.perf_counter()
            for item in threads:
                item.start()
            for item in threads:
                item.join()
            wall = time.perf_counter() - wall0
            total = workers * 4
            conc[str(workers)] = {
                "requests": total,
                "errors": int(sum(errors)),
                "error_rate": sum(errors) / total,
                "throughput_rps": total / wall,
                "latency_ms": percentiles(latencies),
                "rss_mb": rss_mb(),
            }
            print("  concurrency", workers, conc[str(workers)]["throughput_rps"], "err", conc[str(workers)]["errors"], flush=True)
    finally:
        server.shutdown()
    failed = [case["name"] for case in cases if not case.get("ok")]
    return {"cases": cases, "failed": failed, "concurrency": conc}


def reproduction_ok(got, ref):
    n = got["n"]
    acc_off = abs(got["end_to_end_accuracy"] - ref["end_to_end_accuracy"])
    miss_off = abs(got["retrieval_failures"] - ref["retrieval_failures"])
    nll_off = abs(got["nll_on_hits"] - ref["nll_on_hits"])
    ece_off = abs(got["ece_e2e_misses_zeroed"] - ref["ece_e2e"])
    return {
        "accuracy_abs": acc_off,
        "retrieval_failures_abs": miss_off,
        "nll_on_hits_abs": nll_off,
        "ece_zeroed_abs": ece_off,
        "pass": acc_off <= (1.5 / n) and miss_off <= 1 and nll_off <= 5e-3 and ece_off <= 0.02,
    }


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not serializable: %s" % type(value).__name__)


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def load_from_manifest(manifest, device, timings):
    manifest_path = os.path.join(ROOT, "artifacts", "release", "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        on_disk = json.load(handle)
    if on_disk != manifest:
        raise SystemExit("requested manifest does not match the audited release manifest")
    bundle, _verified = load_release_manifest(
        manifest_path,
        device=device,
        root=ROOT,
        timings=timings,
    )
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fingerprint-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if os.path.isabs(args.manifest) else os.path.join(ROOT, args.manifest)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    timings = {}
    print("loading bundle", device, flush=True)
    bundle = load_from_manifest(manifest, device, timings)
    ready = time.time()
    probe_state = "Where is my card?"
    probe_crit = {bundle.labels[0]: bundle.label_texts[0], bundle.labels[1]: bundle.label_texts[1]}
    started = time.perf_counter()
    bundle.score_choice(probe_state, INSTRUCTIONS, probe_crit)
    first_infer = time.perf_counter() - started
    shell_start = float(os.environ.get("VEYA_PROCESS_START") or ENTER_S)
    cold = {
        "process_enter_minus_shell_s": round(ENTER_S - shell_start, 3),
        "import_to_ready_s": round(ready - ENTER_S, 3),
        "ready_wall_s": round(ready - shell_start, 3),
        "first_inference_s": round(first_infer, 3),
        "stages_s": {key: round(value, 3) for key, value in timings.items()},
        "device": device,
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rss_mb_after_load": rss_mb(),
    }
    print("ready", cold["ready_wall_s"], "first", cold["first_inference_s"], flush=True)
    if args.fingerprint_only:
        report_path = resolve_manifest_path(ROOT, manifest["evaluation_output"])
        with open(report_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        test = load_banking("test")
        labels = list(bundle.labels)
        crit = {label: text for label, text in zip(labels, bundle.label_texts)}
        rows = test[1300:1332]
        records, scores, _elapsed = score_shortlist(bundle, rows, labels, crit, bundle.route_temperature)
        saved_rows = saved["fingerprint_rows"]
        mismatches = 0
        for got, expect in zip(records, saved_rows):
            if got["prediction"] != expect["prediction"] or got["shortlist"] != expect["shortlist"]:
                mismatches += 1
        probe_delta = float(np.max(np.abs(scores[0] - np.array(saved["probe_scores"]))))
        out = {
            "mismatched_rows": mismatches,
            "rows": len(records),
            "probe_score_max_abs": probe_delta,
            "pass": mismatches == 0 and probe_delta < 1e-3,
            "cold_load": cold,
        }
        dump(os.path.join(ROOT, "artifacts", "experiments", "phase5", "fingerprint_check.json"), out)
        print(json.dumps(out, indent=2), flush=True)
        raise SystemExit(0 if out["pass"] else 2)

    test = load_banking("test")
    labels = list(bundle.labels)
    derived = sorted({row[1] for row in test})
    if derived != labels:
        raise SystemExit("saved shortlist labels do not match Banking77 categories")
    crit = {label: text for label, text in zip(labels, bundle.label_texts)}
    ranges = {
        "k_selection": (200, 296),
        "crossover": (296, 400),
        "policy_validation": (880, 1100),
        "temperature_fit": (1100, 1300),
        "locked_test": (1300, 1500),
        "expanded_test": (1500, len(test)),
    }
    occupied = list(ranges.values())
    for i, left in enumerate(occupied):
        for right in occupied[i + 1:]:
            if left[0] < right[1] and right[0] < left[1]:
                raise SystemExit("split overlap %s %s" % (left, right))
    hashes = {name: split_hash(test[start:stop]) for name, (start, stop) in ranges.items()}
    base_t = float(bundle.agent.temperature_by_options["choice:6-10"])
    route_t = bundle.route_temperature
    print("locked test", flush=True)
    locked_rows = test[1300:1500]
    locked, probe_scores, locked_s = score_shortlist(bundle, locked_rows, labels, crit, route_t)
    before = aggregate(rescore(locked, base_t), locked_s)
    after = aggregate(locked, locked_s)
    with open(os.path.join(ROOT, "artifacts", "experiments", "phase5", "release_eval.json"), encoding="utf-8") as handle:
        published = json.load(handle)
    check = reproduction_ok(after, published["k8_after"])
    print("reproduction", json.dumps(check), "acc", after["end_to_end_accuracy"], flush=True)
    fingerprint_rows = [{
        "prediction": row["prediction"],
        "shortlist": row["shortlist"],
        "correct": row["correct"],
        "hit": row["hit"],
    } for row in locked[:32]]
    report = {
        "cold_load": cold,
        "split_hashes": hashes,
        "split_ranges": {name: list(span) for name, span in ranges.items()},
        "reproduction": check,
        "k8_temperature_1": before,
        "k8_route_temperature": after,
        "fingerprint_rows": fingerprint_rows,
        "probe_scores": probe_scores[0].tolist(),
    }
    out_path = resolve_manifest_path(ROOT, manifest["evaluation_output"])
    dump(out_path, report)
    if not check["pass"]:
        print("clean load did not reproduce the saved K=8 metrics", flush=True)
        raise SystemExit(2)

    print("packed 77 on locked test", flush=True)
    packed_t = float(bundle.agent.temperature_by_options["choice:11+"])
    packed, packed_s, _logits_packed = packed_records(bundle.agent, locked_rows, labels, crit, packed_t)
    report["packed_77_locked"] = packed_aggregate(packed, packed_s)
    report["packed_77_locked"]["temperature"] = packed_t
    report["packed_77_locked"]["candidate_count"] = 77
    report["k8_route_temperature"]["candidate_count"] = 8

    print("policy validation", flush=True)
    validation, _scores, val_s = score_shortlist(bundle, test[880:1100], labels, crit, route_t)
    winner, compared = fit_policy(validation)
    thresholds = {eps: compared[winner]["thresholds"][eps] for eps in POLICY_EPS}
    # JSON null for a missing threshold. The test split is not used above.
    print("expanded test", len(test) - 1500, flush=True)
    expanded, _scores, exp_s = score_shortlist(bundle, test[1500:], labels, crit, route_t)
    report["k8_expanded"] = aggregate(expanded, exp_s)
    report["k8_expanded"]["rows"] = [1500, len(test)]
    combined = locked + expanded
    report["k8_combined_test"] = aggregate(combined, locked_s + exp_s)
    report["error_decomposition_locked"] = decompose(locked)
    report["error_decomposition_combined"] = decompose(combined)
    report["selective"] = {
        "validation_rows": [880, 1100],
        "validation_seconds": round(val_s, 3),
        "scores_compared_on_validation": {
            kind: {eps: values["validation"][eps] for eps in POLICY_EPS}
            for kind, values in compared.items()
        },
        "chosen_score": winner,
        "choice_rule": "highest validation coverage at epsilon 0.05; keep max_probability unless another score gains more than 0.02 coverage",
        "locked_test": apply_policy(locked, winner, thresholds),
        "expanded_test": apply_policy(expanded, winner, thresholds),
        "combined_test": apply_policy(combined, winner, thresholds),
    }
    curve = risk_coverage_curve(score_vectors(combined, winner), correct_vector(combined))
    deciles = []
    for step in range(1, 11):
        idx = min(len(curve) - 1, max(0, int(round(step / 10 * len(curve))) - 1))
        deciles.append(curve[idx])
    report["risk_coverage_deciles"] = deciles
    dump(os.path.join(ROOT, "artifacts", "experiments", "phase5", "risk_coverage.json"), {
        "score": winner,
        "split": "banking77_test_1300_end",
        "curve": curve,
    })
    policy = {
        "format": "veya-selective-policy-v1",
        "route": "shortlist_k8_packed_rerank",
        "score": winner,
        "operating_epsilon": "0.05",
        "thresholds": thresholds,
        "validation_split": "banking77_test_880_1100",
        "validation_hash": hashes["policy_validation"],
        "test_split": "banking77_test_1300_end",
        "test_hash": hashes["locked_test"],
        "shortlist_sha256": bundle.shortlist_sha256,
        "checkpoint_hash": bundle.checkpoint_hash,
        "chosen_on": "validation",
    }
    policy_path = os.path.join(ROOT, "artifacts", "release", "selective_policy.json")
    dump(policy_path, policy)
    bundle.selective_policy = policy

    print("packed policy validation", flush=True)
    packed_val, packed_val_s, _raw = packed_records(
        bundle.agent, test[880:1100], labels, crit, packed_t, batch_size=4,
    )
    packed_scores = np.array([row["max_probability"] for row in packed_val])
    packed_correct = np.array([1.0 if row["correct"] else 0.0 for row in packed_val])
    packed_thresholds = {eps: select_threshold(packed_scores, packed_correct, float(eps)) for eps in POLICY_EPS}
    report["packed_selective"] = {
        "validation_rows": [880, 1100],
        "validation_seconds": round(packed_val_s, 3),
        "thresholds": packed_thresholds,
        "locked_test": {
            eps: apply_report(
                np.array([row["max_probability"] for row in packed]),
                np.array([1.0 if row["correct"] else 0.0 for row in packed]),
                packed_thresholds[eps],
            )
            for eps in POLICY_EPS
        },
    }

    dump(out_path, _strip(report))
    print("crossover", flush=True)
    report["crossover"] = crossover(bundle, test[296:400], labels)
    print("ag news", flush=True)
    ag = load_ag_news()[400:800]
    names = ["World", "Sports", "Business", "Sci/Tech"]
    ag_crit = {name: "news about %s" % name for name in names}
    ag_parts = []
    for start in range(0, len(ag), 8):
        ag_parts.append(choice_logits(bundle.agent, [row[0] for row in ag[start:start + 8]], ag_crit))
        if start and start % 80 == 0:
            print("  ag", start, flush=True)
    ag_logits = np.concatenate(ag_parts, axis=0)
    ag_t = float(bundle.agent.temperature_by_options["choice:3-5"])
    report["ag_news_400_800"] = metrics(ag_logits, np.array([row[1] for row in ag]), ag_t)
    report["ag_news_400_800"]["regression_vs_phase3_0.955"] = abs(report["ag_news_400_800"]["accuracy"] - 0.955) > 0.02

    print("latency", flush=True)
    report["latency"] = latency_profile(bundle, test[1300][0], labels, crit)
    report["latency"]["rss_mb"] = rss_mb()
    dump(out_path, _strip(report))
    print("http", flush=True)
    try:
        report["http"] = http_suite(bundle, labels, crit)
    except Exception as exc:
        report["http"] = {"failed": ["exception"], "error": "%s: %s" % (type(exc).__name__, exc), "cases": [], "concurrency": {}}
    # Drop per-row logits before the final write. They were only needed in memory.
    report["fingerprint_rows"] = fingerprint_rows
    dump(out_path, _strip(report))
    print("wrote", out_path, "http_failed", report["http"]["failed"], flush=True)
    raise SystemExit(1 if report["http"]["failed"] else 0)


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key != "logits"}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    if isinstance(value, float):
        return value
    return value


if __name__ == "__main__":
    main()
