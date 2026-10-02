"""Freeze-record assembler for the pre-registered evaluation protocol (Section 2).

One JSON document pins every input that could drift between pre-registration
and blind evaluation: the model checkpoint, tokenizer contents, inference code
commit, scorer identity, shortlist configuration, calibration method and
temperature artifact, selective policy, operating epsilon and minimum
acceptable coverage, perturbation and metric implementation versions,
exclusion rules, random seeds, bootstrap procedure, and production
preprocessing version.

The record deliberately carries no timestamp: identical inputs must always
produce an identical ``freeze_record_hash``, so a frozen record stays
re-verifiable after the fact. File-backed items are pinned by content hash,
which also makes the record independent of where the files live on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FREEZE_RECORD_FORMAT = "veya-freeze-record-v1"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

_HASH_FIELDS = (
    "model_checkpoint_sha256",
    "tokenizer_version_sha256",
    "shortlist_config_sha256",
    "temperature_artifact_sha256",
    "selective_policy_sha256",
)

_STRING_FIELDS = (
    "inference_code_commit",
    "scorer_id",
    "calibration_method",
    "selective_policy_algorithm",
    "perturbation_definitions_version",
    "metric_implementation_version",
    "production_preprocessing_version",
)

_REQUIRED_KEYS = (
    "format",
    "epsilon",
    "c_min",
    "exclusion_rules",
    "random_seeds",
    "bootstrap_draws",
    "bootstrap_confidence",
) + _HASH_FIELDS + _STRING_FIELDS


def _sha256(path: str | os.PathLike) -> str:
    """SHA-256 of one file, read in 1 MB chunks (the veya/manifest.py pattern)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_sha256(path: str | os.PathLike) -> str:
    """Content hash of a directory: sorted relative names plus per-file hashes.

    Renaming a file therefore changes the hash, matching the way a tokenizer
    "version" is defined by both its file names and their contents.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"directory not found: {root}")
    files = sorted(child for child in root.rglob("*") if child.is_file())
    if not files:
        raise ValueError(f"directory has no files to freeze: {root}")
    digest = hashlib.sha256()
    for child in files:
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_probability(value: Any, name: str, *, allow_one: bool = False) -> float:
    upper = "1]" if allow_one else "1)"
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number in (0, {upper}") from error
    if not (0.0 < number < 1.0 or (allow_one and number == 1.0)):
        raise ValueError(f"{name} must be a number in (0, {upper}")
    return number


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_exclusion_rules(value: Any, name: str = "exclusion_rules") -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    rules: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} entries must be non-empty strings")
        rules.append(item)
    return rules


def _require_random_seeds(value: Any, name: str = "random_seeds") -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping of names to integer seeds")
    seeds: dict[str, int] = {}
    for key, seed in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{name}[{key!r}] must be an integer seed")
        seeds[key] = seed
    return seeds


def build_freeze_record(
    *,
    model_path: str | os.PathLike,
    tokenizer_dir: str | os.PathLike,
    git_commit: str,
    scorer_id: str,
    shortlist_config_path: str | os.PathLike,
    calibration_method: str,
    temperature_artifact_path: str | os.PathLike,
    selective_policy_algorithm: str,
    selective_policy_artifact_path: str | os.PathLike,
    epsilon: float,
    c_min: float,
    perturbation_definitions_version: str,
    metric_implementation_version: str,
    exclusion_rules: list[str],
    random_seeds: dict[str, int],
    bootstrap_draws: int,
    bootstrap_confidence: float,
    production_preprocessing_version: str,
) -> dict[str, Any]:
    """Assemble every protocol Section 2 freeze item into one record.

    File-backed items are passed as paths and pinned by SHA-256 content hash,
    so the record is independent of where those files live. The returned dict
    passes ``validate_freeze_record`` by construction and hashes
    deterministically under ``freeze_record_hash``.
    """
    for label, path in (
        ("model_path", model_path),
        ("shortlist_config_path", shortlist_config_path),
        ("temperature_artifact_path", temperature_artifact_path),
        ("selective_policy_artifact_path", selective_policy_artifact_path),
    ):
        if not os.path.isfile(path):
            raise ValueError(f"{label} is not an existing file: {path}")
    record = {
        "format": FREEZE_RECORD_FORMAT,
        "model_checkpoint_sha256": _sha256(model_path),
        "tokenizer_version_sha256": _dir_sha256(tokenizer_dir),
        "inference_code_commit": _require_str(git_commit, "git_commit"),
        "scorer_id": _require_str(scorer_id, "scorer_id"),
        "shortlist_config_sha256": _sha256(shortlist_config_path),
        "calibration_method": _require_str(calibration_method, "calibration_method"),
        "temperature_artifact_sha256": _sha256(temperature_artifact_path),
        "selective_policy_algorithm": _require_str(
            selective_policy_algorithm, "selective_policy_algorithm"
        ),
        "selective_policy_sha256": _sha256(selective_policy_artifact_path),
        "epsilon": _require_probability(epsilon, "epsilon"),
        "c_min": _require_probability(c_min, "c_min", allow_one=True),
        "perturbation_definitions_version": _require_str(
            perturbation_definitions_version, "perturbation_definitions_version"
        ),
        "metric_implementation_version": _require_str(
            metric_implementation_version, "metric_implementation_version"
        ),
        "exclusion_rules": _require_exclusion_rules(exclusion_rules),
        "random_seeds": _require_random_seeds(random_seeds),
        "bootstrap_draws": _require_positive_int(bootstrap_draws, "bootstrap_draws"),
        "bootstrap_confidence": _require_probability(
            bootstrap_confidence, "bootstrap_confidence"
        ),
        "production_preprocessing_version": _require_str(
            production_preprocessing_version, "production_preprocessing_version"
        ),
    }
    validate_freeze_record(record)
    return record


def validate_freeze_record(record: Any) -> None:
    """Raise ``ValueError`` unless ``record`` is a complete v1 freeze record."""
    if not isinstance(record, dict):
        raise ValueError("freeze record must be a JSON object")
    if record.get("format") != FREEZE_RECORD_FORMAT:
        raise ValueError(
            "unsupported freeze record format %r (expected %r)"
            % (record.get("format"), FREEZE_RECORD_FORMAT)
        )
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError("freeze record is missing fields: %s" % ", ".join(missing))
    for field in _HASH_FIELDS:
        value = record[field]
        if not isinstance(value, str) or not _HEX64.match(value):
            raise ValueError(f"{field} must be a 64-character lowercase hex SHA-256")
    for field in _STRING_FIELDS:
        _require_str(record[field], field)
    _require_probability(record["epsilon"], "epsilon")
    _require_probability(record["c_min"], "c_min", allow_one=True)
    _require_probability(record["bootstrap_confidence"], "bootstrap_confidence")
    _require_positive_int(record["bootstrap_draws"], "bootstrap_draws")
    _require_exclusion_rules(record["exclusion_rules"])
    _require_random_seeds(record["random_seeds"])


def freeze_record_hash(record: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON serialization of a validated record.

    Canonical form: ``json.dumps(record, sort_keys=True,
    separators=(",", ":"), ensure_ascii=False)`` encoded as UTF-8. Identical
    inputs produce identical hashes; changing any field changes the hash.
    """
    validate_freeze_record(record)
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
