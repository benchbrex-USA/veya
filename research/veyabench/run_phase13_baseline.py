"""Score the sealed Phase 13 bank once. Do not fit a temperature and do not train.

The public gateway admits few requests per minute, so this runner calls the
local model process that the gateway forwards to. The release id is read from
that process. Predictions land under artifacts/experiments/phase13/.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from research.veyabench.phase13_counterfactual import (
    evaluate,
    load_jsonl,
    verify_seal,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data" / "phase13"
OUTPUT = ROOT / "artifacts" / "experiments" / "phase13"
MODEL_CHOICE = "http://127.0.0.1:8080/v1/choice"
MODEL_VERSION = "http://127.0.0.1:8080/version"
GATEWAY_VERSION = "http://127.0.0.1:3000/api/version"
GATEWAY_INFER = "http://127.0.0.1:3000/api/infer"


def _get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 60) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _choice_body(row: dict, gateway: bool) -> dict:
    body = {
        "state": row["state"],
        "instructions": row["instructions"],
        "criteria": {key: text for key, text in row["criteria"]},
    }
    if gateway:
        body = {"primitive": "choice", **body}
    return body


def score_row(row: dict, url: str = MODEL_CHOICE, gateway: bool = False, attempts: int = 4) -> dict:
    delay = 1.0
    last_error = None
    for attempt in range(attempts):
        try:
            payload = _post_json(url, _choice_body(row, gateway))
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503, 504):
                raise
            last_error = error
            time.sleep(delay)
            delay = min(delay * 2, 15)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as error:
            last_error = error
            time.sleep(delay)
            delay = min(delay * 2, 15)
    else:
        raise RuntimeError(f"{row['id']} failed after retries: {last_error}")
    probabilities = payload.get("probabilities")
    answer = payload.get("answer")
    deferred = bool(payload.get("deferred"))
    return {
        "id": row["id"],
        "target": row["target"],
        "slice": row["slice"],
        "tags": row["tags"],
        "variant": row["variant"],
        "counterfactual_id": row["counterfactual_id"],
        "state": row["state"],
        "answer": answer,
        "probabilities": probabilities,
        "confidence": payload.get("confidence"),
        "max_probability": payload.get("max_probability"),
        "route": payload.get("route"),
        "deferred": deferred,
        "correct": answer == row["target"] and not deferred,
    }


def _load_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") and "answer" in row:
            found[row["id"]] = row
    return found


def _version_snapshot() -> dict:
    snapshot = {"model": _get_json(MODEL_VERSION)}
    try:
        snapshot["gateway"] = _get_json(GATEWAY_VERSION)
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, json.JSONDecodeError) as error:
        snapshot["gateway_error"] = type(error).__name__
    model_release = (snapshot.get("model") or {}).get("release")
    gateway_release = (snapshot.get("gateway") or {}).get("release")
    if model_release and gateway_release and model_release != gateway_release:
        raise RuntimeError(f"model release {model_release} differs from gateway release {gateway_release}")
    return snapshot


def run_baseline(data_dir: Path, out_dir: Path, limit: int | None, force: bool) -> dict:
    seal = verify_seal(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "baseline.json"
    if summary_path.exists() and not force and limit is None:
        raise RuntimeError(f"{summary_path} already exists; refusing to replace the sealed measurement")
    rows = load_jsonl(data_dir / "sealed_eval.jsonl")
    if limit is not None:
        rows = rows[:limit]
    versions = _version_snapshot()
    prediction_path = out_dir / "baseline_predictions.jsonl"
    done = {} if force else _load_predictions(prediction_path)
    mode = "w" if force else "a"
    started = time.time()
    with prediction_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            if row["id"] in done:
                continue
            scored = score_row(row)
            handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            done[row["id"]] = scored
            if index % 200 == 0 or index == len(rows):
                elapsed = time.time() - started
                print(f"scored {index}/{len(rows)} in {elapsed:.0f}s", flush=True)
    predictions = {row_id: done[row_id] for row_id in (row["id"] for row in rows)}
    report = evaluate(rows, predictions)
    spot = None
    transcript = next((row for row in load_jsonl(data_dir / "sealed_eval.jsonl") if "regression" in row["tags"] and row["state"].startswith("Payment succeeded")), None)
    if transcript is not None and limit is None:
        try:
            spot = score_row(transcript, GATEWAY_INFER, gateway=True, attempts=2)
        except (RuntimeError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            spot = {"error": type(error).__name__}
    document = {
        "format": "veya-phase13-baseline-v1",
        "seal_sha256": seal["sha256"],
        "seal_row_count": seal["row_count"],
        "scored_rows": len(rows),
        "temperature_fit": False,
        "train_prohibited": True,
        "scoring_endpoint": MODEL_CHOICE,
        "versions": versions,
        "release": (versions.get("model") or {}).get("release"),
        "reranker_id": (versions.get("model") or {}).get("reranker_id"),
        "checkpoint_sha256": (versions.get("model") or {}).get("checkpoint_sha256"),
        "gateway_spot_check": spot,
        "metrics": report,
    }
    if limit is None:
        temporary = summary_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    document = run_baseline(args.data, args.out, args.limit, args.force)
    gates = document["metrics"]["gates"]
    print(
        json.dumps(
            {
                "release": document["release"],
                "scored_rows": document["scored_rows"],
                "overall_accuracy": document["metrics"]["overall_accuracy"],
                "gates_passed": gates["passed"],
                "temperature_fit": document["temperature_fit"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
