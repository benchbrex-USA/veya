"""Versioned BenchBrex routing eval set.

Rows are authored fixtures with deterministic scores. They are not a
generalization benchmark and they do not use an LLM as the judge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FORMAT = "benchbrex-eval-v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "research/data/experts"


def load_dataset(root: Path | None = None) -> dict[str, Any]:
    folder = dataset_dir(root)
    manifest = json.loads((folder / "benchbrex_eval_v1.json").read_text(encoding="utf-8"))
    raw = (folder / "benchbrex_eval_v1.jsonl").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("evaluation dataset hash does not match the manifest")
    if manifest.get("format") != FORMAT:
        raise ValueError("unexpected evaluation dataset format")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["rows"]:
        raise ValueError("evaluation dataset row count does not match the manifest")
    return {"manifest": manifest, "rows": rows, "sha256": digest}


def score_row(row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Score one prediction. `prediction` is model output, not a second judge."""
    kind = row["scoring"]
    expected = row["expected"]
    if kind == "exact_label":
        correct = prediction.get("label") == expected["label"]
    elif kind == "json_keys":
        obj = prediction.get("object")
        correct = isinstance(obj, dict) and all(key in obj for key in expected["keys"])
    elif kind == "flag":
        correct = prediction.get("flagged") is expected["flagged"]
    else:
        raise ValueError("unknown scoring function")
    return {"id": row["id"], "correct": correct, "scoring": kind, "ground_truth_source": row["ground_truth_source"]}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n": 0, "accuracy": None}
    hits = sum(1 for item in results if item["correct"] is True)
    return {"n": len(results), "correct": hits, "accuracy": hits / len(results)}
