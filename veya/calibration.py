"""Versioned temperature calibration. Fit on a calibration split, evaluate on another.

The checkpoint's own temperatures stay in place until an artifact is loaded. Nothing
in this module invents a temperature: every value is the NLL minimizer on the rows
it was given, or a fallback to a coarser group that had enough rows.
"""
import json
import math
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

from .common import TEMP_MAX, TEMP_MIN, clamp_temperature

FORMAT = "veya-calibration-v1"
MIN_OBSERVATIONS = 32


def cardinality_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    if k <= 20:
        return "11-20"
    if k <= 50:
        return "21-50"
    return "51+"


def _grid() -> np.ndarray:
    return np.exp(np.linspace(math.log(TEMP_MIN), math.log(TEMP_MAX), 48))


def _log_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    logp = _log_probs(logits, temperature)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-logp[np.arange(len(labels)), labels].mean())


def brier(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    logp = _log_probs(logits, temperature)
    p = np.exp(logp)
    labels = np.asarray(labels, dtype=np.int64)
    truth = np.zeros_like(p)
    truth[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((p - truth) ** 2, axis=1)))


def accuracy(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    logp = _log_probs(logits, temperature)
    return float((logp.argmax(axis=1) == np.asarray(labels)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature in [TEMP_MIN, TEMP_MAX] with the lowest NLL on these rows."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on zero rows")
    best_t, best_nll = 1.0, float("inf")
    for t in _grid():
        score = nll(logits, labels, float(t))
        if score < best_nll:
            best_t, best_nll = float(t), score
    return clamp_temperature(best_t)


def fit_groups(
    groups: Dict[str, tuple],
    min_observations: int = MIN_OBSERVATIONS,
) -> Dict[str, float]:
    """Fit a temperature per group key. Groups smaller than the floor are omitted."""
    fitted = {}
    for key, (logits, labels) in groups.items():
        labels = np.asarray(labels)
        if len(labels) < min_observations:
            continue
        fitted[key] = fit_temperature(logits, labels)
    return fitted


def build_artifact(
    groups: Dict[str, tuple],
    checkpoint_id: str,
    dataset_hash: str,
    min_observations: int = MIN_OBSERVATIONS,
    fitting_split: str = "calibration",
    checkpoint_hash: Optional[str] = None,
    fit_split_hashes: Optional[Dict[str, str]] = None,
) -> dict:
    counts = {key: int(len(np.asarray(labels))) for key, (_logits, labels) in groups.items()}
    fitted = fit_groups(groups, min_observations=min_observations)
    primitive: Dict[str, float] = {}
    cardinality: Dict[str, float] = {}
    for key, value in fitted.items():
        if ":" in key:
            cardinality[key] = value
        else:
            primitive[key] = value
    global_rows = [(np.asarray(logits), np.asarray(labels)) for logits, labels in groups.values()]
    if global_rows:
        logits = np.concatenate([row[0] for row in global_rows], axis=0)
        labels = np.concatenate([row[1] for row in global_rows], axis=0)
        global_t = fit_temperature(logits, labels) if len(labels) >= min_observations else 1.0
    else:
        global_t = 1.0
    return {
        "format": FORMAT,
        "checkpoint_id": checkpoint_id,
        "dataset_hash": dataset_hash,
        "method": "temperature-nll",
        "min_observations": min_observations,
        "fitting_split": fitting_split,
        "fitting_role": "calibration",
        "checkpoint_hash": checkpoint_hash,
        "fit_split_hashes": fit_split_hashes or {},
        "scorer_id": "base_decision_head",
        "global_temperature": global_t,
        "primitive_temperatures": primitive,
        "cardinality_temperatures": cardinality,
        "sample_counts": counts,
        "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def resolve_temperature(artifact: dict, primitive: str, k: int, language: Optional[str] = None) -> float:
    """Fall back from language+bucket, to bucket, to primitive, to global."""
    bucket = "%s:%s" % (primitive, cardinality_bucket(k))
    cardinality = artifact.get("cardinality_temperatures") or {}
    primitive_map = artifact.get("primitive_temperatures") or {}
    language_map = artifact.get("language_temperatures") or {}
    if language:
        specific = language_map.get("%s:%s" % (language, bucket))
        if specific is not None:
            return clamp_temperature(specific)
    if bucket in cardinality:
        return clamp_temperature(cardinality[bucket])
    if primitive in primitive_map:
        return clamp_temperature(primitive_map[primitive])
    return clamp_temperature(artifact.get("global_temperature", 1.0))


def dump_calibration(artifact: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_calibration(path: str, checkpoint_hash: Optional[str] = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("format") != FORMAT:
        raise ValueError("unsupported calibration format %r" % artifact.get("format"))
    if artifact.get("fitting_split") in (None, "", "test"):
        raise ValueError("calibration artifact must name a non-test fitting split")
    if artifact.get("fitting_role", "calibration") != "calibration":
        raise ValueError("calibration artifact fitting_role must be calibration")
    if checkpoint_hash is not None and artifact.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("calibration artifact belongs to a different checkpoint")
    if artifact.get("scorer_id", "base_decision_head") != "base_decision_head":
        raise ValueError("packed calibration belongs to a different scorer")
    return artifact


def apply_calibration(agent, artifact: dict) -> None:
    """Overlay a fitted artifact onto the checkpoint temperatures. Unspecified buckets stay.

    Replacing the whole map would drop shipped buckets that this artifact did not refit.
    """
    primitive = artifact.get("primitive_temperatures") or {}
    order = ("choice", "score", "noul")
    temps = [clamp_temperature(value) for value in list(getattr(agent, "temperature", [1.0, 1.0, 1.0]))]
    for index, name in enumerate(order):
        if name in primitive:
            temps[index] = clamp_temperature(primitive[name])
    agent.temperature = temps
    merged = {key: clamp_temperature(value) for key, value in dict(getattr(agent, "temperature_by_options", {})).items()}
    for key, value in (artifact.get("cardinality_temperatures") or {}).items():
        merged[key] = clamp_temperature(value)
    agent.temperature_by_options = merged
    agent.calibration_version = artifact.get("dataset_hash") or artifact.get("fitted_at") or FORMAT
    agent.calibration = artifact
