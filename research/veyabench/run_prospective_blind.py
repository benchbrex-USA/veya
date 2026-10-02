"""Single-workflow CLI for the prospective blind verdict protocol (gates G6/G8).

``run_blind_verdict`` (research/veyabench/blind_verdict.py) is the library-only
P1-P6 orchestrator; this module is the one documented way to drive it over a
supplied target-domain labeled dataset:

    .venv/bin/python -m research.veyabench.run_prospective_blind \\
        --config artifacts/release/prospective_blind_config.json \\
        --data /path/to/target-domain-labeled.jsonl \\
        --output artifacts/release/prospective_blind_result.json

The runner never touches the model, never writes per-example outputs, and never
modifies a frozen artifact. It verifies provenance first (config format, pinned
release manifest, frozen selective policy, frozen temperature artifact, dataset
hash, calibration/test split pins), loads the dataset under a strict schema,
then hands a fully validated config to ``run_blind_verdict``, which runs split
provenance and the contamination screens before any statistic.

Run config schema (``veya-prospective-blind-config-v1``), all paths may be
absolute or relative to the working directory / repository root:

    format                     "veya-prospective-blind-config-v1"
    manifest                   {"path", "sha256"} of artifacts/release/manifest.json
    selective_policy_artifact  {"path", "sha256"} of the frozen selective policy
    temperature_artifact       {"path", "sha256"} of the frozen temperature artifact
    freeze_record              a full veya-freeze-record-v1 record
    policy                     the frozen selective policy with only
                               operating_epsilon switched to one frozen threshold
    calibration_data           {"path", "sha256"} of the frozen calibration dataset
    test_data                  {"path", "sha256"} of the external target-domain dataset
    training_texts             inline list of controlled training-side texts, OR
    training_texts_path        .jsonl / .json / .csv / .tsv corpus (or list of them)
    temperature                the frozen calibration temperature
    domain                     free-form domain name for the result language
    coverage_retention_min     optional, default 0.95 (G3)
    collapse_floor             optional, default {"coverage": 0.05, "accuracy": 0.30}
    contamination_tolerance    optional, default 0 (P1)
    operational                optional P6 measurements; null fails P6
    epsilon / c_min            optional; must agree with the freeze record
    notes                      free-form documentation

Dataset record schema (.jsonl: one JSON object per line, blank lines skipped;
.json: an array of records). Unknown fields are rejected. String labels and
predictions must use one consistent type across the whole dataset:

    id           required  non-empty string, unique across the dataset
    text         required  non-empty string
    label        required  non-empty string, or a non-negative integer column index
    scores       required  list of >= 2 numbers in (0, 1] summing to 1.0 (1e-3
                           tolerance); probabilities over the candidate shortlist
    prediction   optional  string/int matching the label type; defaults to
                           argmax(scores) (candidates[top] for string labels)
    correct      optional  boolean; defaults to (prediction == label) and must
                           agree with it when supplied
    candidates   optional  one entry per score; used to derive string predictions
    slice_tag    optional  "clean" (default) or a registered protocol slice
    confidence   optional  number in (0, 1]; defaults to max(scores)

Exit codes: 0 the verdict passed every primary gate, 1 the verdict is a
documented FAIL, 2 any validation or runtime error (fail closed).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from typing import Mapping

from research.veyabench.blind_verdict import (
    KNOWN_SLICE_CONDITIONS,
    run_blind_verdict,
    verdict_summary,
)
from research.veyabench.freeze_record import validate_freeze_record

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTPUT = os.path.join(ROOT, "artifacts", "release", "prospective_blind_result.json")

CONFIG_FORMAT = "veya-prospective-blind-config-v1"
RESULT_FORMAT = "veya-prospective-blind-result-v1"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

PLACEHOLDER_SHA256 = "0" * 64
PLACEHOLDER_MARKERS = ("REPLACE_WITH", "<", ">")
SCORE_SUM_TOLERANCE = 1e-3

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_REQUIRED_CONFIG_KEYS = (
    "manifest",
    "selective_policy_artifact",
    "temperature_artifact",
    "freeze_record",
    "policy",
    "calibration_data",
    "test_data",
    "temperature",
)
_ALLOWED_CONFIG_KEYS = frozenset(_REQUIRED_CONFIG_KEYS) | {
    "format",
    "notes",
    "training_texts",
    "training_texts_path",
    "domain",
    "coverage_retention_min",
    "collapse_floor",
    "contamination_tolerance",
    "operational",
    "epsilon",
    "c_min",
}
_ALLOWED_RECORD_KEYS = frozenset({
    "id",
    "text",
    "label",
    "prediction",
    "correct",
    "scores",
    "candidates",
    "slice_tag",
    "confidence",
})
_RUN_CONFIG_PASSTHROUGH = (
    "domain",
    "coverage_retention_min",
    "collapse_floor",
    "contamination_tolerance",
    "operational",
    "epsilon",
    "c_min",
)

BLOCKED_EXTERNAL_MESSAGE = (
    "test_data is still the unfilled template placeholder. The exact minimum action "
    "that unblocks the G6/G8 prospective blind evaluation: supply a target-domain "
    "labeled dataset matching the documented schema, then fill in test_data.path and "
    "test_data.sha256 in the run config (see docs/PROSPECTIVE_BLIND_WORKFLOW.md)."
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str, what: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError("%s %s is not valid JSON: %s" % (what, path, error)) from None


def _resolve_path(raw: str, what: str, extra_roots=()) -> str:
    """Resolve a pinned path: absolute, then cwd, repository root, extra roots."""
    if os.path.isabs(raw):
        candidates = [raw]
    else:
        candidates = [os.path.join(os.getcwd(), raw), os.path.join(ROOT, raw)]
        candidates.extend(os.path.join(root, raw) for root in extra_roots)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise ValueError(
        "%s path %r does not resolve to an existing file (tried: %s)"
        % (what, raw, ", ".join(sorted(set(os.path.abspath(c) for c in candidates))))
    )


def _validate_pin(entry, what: str) -> tuple[str, str]:
    if not isinstance(entry, Mapping):
        raise ValueError("%s must be a mapping with path and sha256" % what)
    path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("%s.path must be a non-empty string" % what)
    if not isinstance(digest, str) or not _HEX64_RE.match(digest):
        raise ValueError("%s.sha256 must be a 64-character lowercase hex SHA-256" % what)
    return path, digest


def _is_placeholder(path: str, digest: str) -> bool:
    return digest == PLACEHOLDER_SHA256 or any(marker in path for marker in PLACEHOLDER_MARKERS)


def _verify_pinned_file(entry, what: str, freeze_hash=None, extra_roots=()) -> tuple[str, str]:
    """Fail closed unless the pinned file exists and hashes to its pin."""
    path, digest = _validate_pin(entry, what)
    resolved = _resolve_path(path, what, extra_roots)
    actual = _sha256_file(resolved)
    if actual != digest:
        raise ValueError(
            "%s hash mismatch: the config pins %s but %s hashes to %s"
            % (what, digest, resolved, actual)
        )
    if freeze_hash is not None and digest != freeze_hash:
        raise ValueError(
            "%s pins %s but the freeze record pins %s; the run config must reference "
            "the frozen artifact" % (what, digest, freeze_hash)
        )
    return resolved, digest


def _verify_manifest_links(manifest, record) -> None:
    """The freeze record must describe exactly the pinned (re-minted) release."""
    if not isinstance(manifest, Mapping):
        raise ValueError("the pinned release manifest must be a JSON object")
    automation = manifest.get("automation") or {}
    if automation.get("status") != "DISABLED":
        raise ValueError(
            "the pinned release manifest reports automation status %r; the prospective "
            "blind protocol requires selective automation to stay DISABLED"
            % automation.get("status")
        )
    checks = (
        ("manifest.base_checkpoint_hash", manifest.get("base_checkpoint_hash"),
         record["model_checkpoint_sha256"]),
        ("manifest.selective_policy_hash", manifest.get("selective_policy_hash"),
         record["selective_policy_sha256"]),
        ("manifest.choice.shortlist_hash", (manifest.get("choice") or {}).get("shortlist_hash"),
         record["shortlist_config_sha256"]),
        ("manifest.calibration.hierarchical_choice_hash",
         (manifest.get("calibration") or {}).get("hierarchical_choice_hash"),
         record["temperature_artifact_sha256"]),
        ("manifest.git_commit", manifest.get("git_commit"), record["inference_code_commit"]),
    )
    for label, manifest_value, record_value in checks:
        if manifest_value != record_value:
            raise ValueError(
                "%s (%r) does not match the freeze record value %r; the run config must "
                "reference the re-minted release" % (label, manifest_value, record_value)
            )


def _verify_policy_derivation(policy, frozen) -> str:
    """The run policy may only switch operating_epsilon to a frozen threshold."""
    if not isinstance(policy, Mapping) or not isinstance(frozen, Mapping):
        raise ValueError("the run policy and the frozen selective policy must be JSON objects")
    if set(policy) != set(frozen):
        raise ValueError(
            "the run policy must copy the frozen selective policy artifact exactly; only "
            "the operating_epsilon value may change (run keys: %s; frozen keys: %s)"
            % (sorted(policy), sorted(frozen))
        )
    for key, value in frozen.items():
        if key != "operating_epsilon" and policy.get(key) != value:
            raise ValueError(
                "run policy field %r differs from the frozen selective policy artifact; "
                "only operating_epsilon may change (run %r, frozen %r)"
                % (key, policy.get(key), value)
            )
    thresholds = frozen.get("thresholds") or {}
    operating = policy.get("operating_epsilon")
    if not isinstance(operating, str) or operating not in thresholds:
        raise ValueError(
            "run policy operating_epsilon %r must select one of the frozen thresholds: %s"
            % (operating, sorted(thresholds))
        )
    if thresholds.get(operating) is None:
        raise ValueError(
            "run policy operating_epsilon %r selects a null frozen threshold; choose one "
            "of %s" % (operating, sorted(k for k, v in thresholds.items() if v is not None))
        )
    frozen_operating = frozen.get("operating_epsilon")
    if (
        isinstance(frozen_operating, str)
        and frozen_operating in thresholds
        and thresholds.get(frozen_operating) is not None
        and operating != frozen_operating
    ):
        raise ValueError(
            "the frozen selective policy already pins operating_epsilon %r; the run "
            "policy may not switch it" % frozen_operating
        )
    return operating


def _verify_temperature(config: Mapping, artifact) -> float:
    temperature = config.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError(
            "config temperature must be a positive finite number (the frozen calibration "
            "temperature; it is never refit on evaluation data)"
        )
    if not isinstance(artifact, Mapping):
        raise ValueError("the pinned temperature artifact must be a JSON object")
    frozen_temperature = artifact.get("temperature")
    if isinstance(frozen_temperature, bool) or not isinstance(frozen_temperature, (int, float)):
        raise ValueError(
            "the pinned temperature artifact does not expose a numeric top-level "
            "'temperature' field"
        )
    if float(temperature) != float(frozen_temperature):
        raise ValueError(
            "config temperature %r does not match the frozen temperature artifact "
            "value %r" % (temperature, frozen_temperature)
        )
    return float(temperature)


def _read_text_corpus(resolved: str) -> list[str]:
    """One controlled training-side corpus file: .jsonl, .json, .csv, or .tsv."""
    lower = resolved.lower()
    texts: list[str] = []
    if lower.endswith(".jsonl"):
        with open(resolved, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "training corpus %s line %d is not valid JSON: %s"
                        % (resolved, line_no, error)
                    ) from None
                text = value.get("text") if isinstance(value, dict) else value
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        "training corpus %s line %d must be a string or an object with a "
                        "non-empty 'text' field" % (resolved, line_no)
                    )
                texts.append(text)
    elif lower.endswith(".json"):
        document = _load_json(resolved, "training corpus")
        if not isinstance(document, list):
            raise ValueError("training corpus %s must be a JSON array" % resolved)
        for position, value in enumerate(document, 1):
            text = value.get("text") if isinstance(value, dict) else value
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "training corpus %s entry %d must be a string or an object with a "
                    "non-empty 'text' field" % (resolved, position)
                )
            texts.append(text)
    elif lower.endswith(".csv") or lower.endswith(".tsv"):
        delimiter = "\t" if lower.endswith(".tsv") else ","
        with open(resolved, encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                raise ValueError("training corpus %s is empty" % resolved)
            if "text" in header:
                column = header.index("text")
            elif len(header) == 1:
                column = 0
            else:
                raise ValueError(
                    "training corpus %s must carry a 'text' column (header: %s)"
                    % (resolved, header)
                )
            for row_no, row in enumerate(reader, 2):
                if len(row) > column and row[column].strip():
                    texts.append(row[column].strip())
    else:
        raise ValueError(
            "unsupported training corpus format %r (expected .jsonl, .json, .csv, or .tsv)"
            % resolved
        )
    return texts


def _load_training_texts(config: Mapping) -> list[str]:
    inline = config.get("training_texts")
    via_path = config.get("training_texts_path")
    if (inline is None) == (via_path is None):
        raise ValueError(
            "config must declare exactly one of training_texts (an inline list of "
            "strings) or training_texts_path (.jsonl/.json/.csv/.tsv corpus files)"
        )
    if inline is not None:
        if not isinstance(inline, list) or not inline:
            raise ValueError("training_texts must be a non-empty list of strings")
        for position, value in enumerate(inline):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "training_texts[%d] must be a non-empty string" % position
                )
        return list(inline)
    paths = via_path if isinstance(via_path, list) else [via_path]
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("training_texts_path must be a path or a list of non-empty paths")
    texts: list[str] = []
    for raw in paths:
        resolved = _resolve_path(raw, "training_texts_path")
        texts.extend(_read_text_corpus(resolved))
    if not texts:
        raise ValueError("the training-side contamination corpus is empty")
    return texts


def _load_records(path: str) -> list[tuple[int, object]]:
    lower = str(path).lower()
    records: list[tuple[int, object]] = []
    if lower.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append((line_no, json.loads(line)))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "dataset %s line %d is not valid JSON: %s" % (path, line_no, error)
                    ) from None
    elif lower.endswith(".json"):
        document = _load_json(path, "dataset")
        if not isinstance(document, list):
            raise ValueError("dataset %s must be a JSON array of records" % path)
        records = list(enumerate(document, 1))
    else:
        raise ValueError(
            "unsupported dataset format %r: use .jsonl (one record per line) or .json "
            "(an array of records)" % path
        )
    if not records:
        raise ValueError("dataset %s contains no records" % path)
    return records


def _record_to_case(position: int, record) -> tuple[dict, str, str]:
    """Validate one external record strictly and build the blind case contract."""
    where = "record %d" % position
    if not isinstance(record, Mapping):
        raise ValueError("%s must be a JSON object" % where)
    unknown = sorted(set(record) - _ALLOWED_RECORD_KEYS)
    if unknown:
        raise ValueError(
            "%s carries unknown field(s) %s; allowed fields: %s"
            % (where, ", ".join(unknown), ", ".join(sorted(_ALLOWED_RECORD_KEYS)))
        )
    for field in ("id", "text", "label", "scores"):
        if field not in record:
            raise ValueError("%s is missing required field %r" % (where, field))

    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("%s.id must be a non-empty string" % where)
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("%s.text must be a non-empty string" % where)

    label = record["label"]
    label_is_int = isinstance(label, int) and not isinstance(label, bool)
    if label_is_int:
        if label < 0:
            raise ValueError("%s.label must be a non-negative integer index" % where)
    elif not isinstance(label, str) or not label.strip():
        raise ValueError(
            "%s.label must be a non-empty string or a non-negative integer index" % where
        )

    raw_scores = record["scores"]
    if not isinstance(raw_scores, list) or len(raw_scores) < 2:
        raise ValueError("%s.scores must be a list of at least two numbers" % where)
    scores = []
    for index, value in enumerate(raw_scores):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("%s.scores[%d] must be a number in (0, 1]" % (where, index))
        number = float(value)
        if not math.isfinite(number) or not (0.0 < number <= 1.0):
            raise ValueError("%s.scores[%d] must be a number in (0, 1]" % (where, index))
        scores.append(number)
    total = math.fsum(scores)
    if abs(total - 1.0) > SCORE_SUM_TOLERANCE:
        raise ValueError(
            "%s.scores must sum to 1.0 within %g (got %.6f)"
            % (where, SCORE_SUM_TOLERANCE, total)
        )
    if label_is_int and label >= len(scores):
        raise ValueError(
            "%s.label index %d is outside the score width %d"
            % (where, label, len(scores))
        )

    candidates = record.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list) or len(candidates) != len(scores):
            given = len(candidates) if isinstance(candidates, list) else -1
            raise ValueError(
                "%s.candidates must be a list with one entry per score (%d expected, "
                "%d given)" % (where, len(scores), given)
            )
        for index, value in enumerate(candidates):
            if (
                isinstance(value, bool)
                or not isinstance(value, (str, int))
                or (isinstance(value, str) and not value.strip())
            ):
                raise ValueError(
                    "%s.candidates[%d] must be a non-empty string or an integer"
                    % (where, index)
                )

    prediction = record.get("prediction")
    if prediction is None:
        top = scores.index(max(scores))
        if label_is_int:
            prediction = top
        elif candidates is not None and all(
            isinstance(value, str) for value in candidates
        ):
            prediction = candidates[top]
        else:
            raise ValueError(
                "%s must supply prediction (or string candidates) so the model's chosen "
                "candidate can be identified for string labels" % where
            )
    prediction_is_int = isinstance(prediction, int) and not isinstance(prediction, bool)
    if label_is_int != prediction_is_int:
        raise ValueError(
            "%s.prediction %r must use the same type as label %r (both strings or both "
            "integer indices)" % (where, prediction, label)
        )
    if prediction_is_int:
        if not 0 <= prediction < len(scores):
            raise ValueError(
                "%s.prediction index %d is outside the score width %d"
                % (where, prediction, len(scores))
            )
    elif not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("%s.prediction must be a non-empty string" % where)

    predicted_matches_label = prediction == label
    correct = record.get("correct")
    if correct is None:
        correct = predicted_matches_label
    elif not isinstance(correct, bool):
        raise ValueError("%s.correct must be a boolean" % where)
    elif correct != predicted_matches_label:
        raise ValueError(
            "%s.correct=%r contradicts prediction %r against label %r"
            % (where, correct, prediction, label)
        )

    slice_tag = record.get("slice_tag", "clean")
    if not isinstance(slice_tag, str) or slice_tag not in KNOWN_SLICE_CONDITIONS:
        raise ValueError(
            "%s.slice_tag %r is not a registered protocol slice; allowed: %s"
            % (where, slice_tag, ", ".join(KNOWN_SLICE_CONDITIONS))
        )

    confidence = record.get("confidence")
    if confidence is None:
        confidence = float(max(scores))
    elif (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not (0.0 < float(confidence) <= 1.0)
    ):
        raise ValueError("%s.confidence must be a number in (0, 1]" % where)
    else:
        confidence = float(confidence)

    case = {
        "text": text,
        "gold_label": label,
        "prediction": prediction,
        "correct": bool(correct),
        "probabilities": scores,
        "logits": [math.log(value) for value in scores],
        "slice_tag": slice_tag,
        "confidence": confidence,
        "candidate_count": len(scores),
    }
    return case, record_id, ("int" if label_is_int else "str")


def _load_dataset(data_path: str) -> list[dict]:
    """Load and validate every record; ids must be unique, label types uniform."""
    if not os.path.isfile(data_path):
        raise ValueError("--data path %s is not an existing file" % data_path)
    cases: list[dict] = []
    seen_ids: set[str] = set()
    label_types: set[str] = set()
    for position, record in _load_records(data_path):
        case, record_id, label_type = _record_to_case(position, record)
        if record_id in seen_ids:
            raise ValueError("dataset record %d reuses id %r" % (position, record_id))
        seen_ids.add(record_id)
        label_types.add(label_type)
        cases.append(case)
    if len(label_types) > 1:
        raise ValueError(
            "labels and predictions must use one consistent type across the dataset "
            "(all strings or all integer indices)"
        )
    return cases


def run(config_path: str, data_path: str, output_path: str) -> dict:
    """Verify provenance, run the verdict once, and write the result envelope."""
    if not os.path.isfile(config_path):
        raise ValueError("--config path %s is not an existing file" % config_path)

    # 1. Load and validate the frozen config.
    config = _load_json(config_path, "config")
    if not isinstance(config, Mapping):
        raise ValueError("config %s must be a JSON object" % config_path)
    if config.get("format") != CONFIG_FORMAT:
        raise ValueError(
            "unsupported config format %r (expected %r)"
            % (config.get("format"), CONFIG_FORMAT)
        )
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "config carries unknown key(s) %s; allowed keys: %s"
            % (", ".join(unknown), ", ".join(sorted(_ALLOWED_CONFIG_KEYS)))
        )
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError("config is missing required key(s): %s" % ", ".join(missing))
    record = config["freeze_record"]
    validate_freeze_record(record)

    # 2. Provenance: the unfilled template fails closed with the blocker message.
    test_pin_path, test_pin_digest = _validate_pin(config["test_data"], "test_data")
    if _is_placeholder(test_pin_path, test_pin_digest):
        raise ValueError(BLOCKED_EXTERNAL_MESSAGE)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    manifest_path, _ = _verify_pinned_file(config["manifest"], "manifest")
    _verify_manifest_links(_load_json(manifest_path, "manifest"), record)
    policy_path, _ = _verify_pinned_file(
        config["selective_policy_artifact"],
        "selective_policy_artifact",
        freeze_hash=record["selective_policy_sha256"],
    )
    temperature_path, _ = _verify_pinned_file(
        config["temperature_artifact"],
        "temperature_artifact",
        freeze_hash=record["temperature_artifact_sha256"],
    )
    temperature = _verify_temperature(
        config, _load_json(temperature_path, "temperature artifact")
    )
    _verify_policy_derivation(
        config["policy"], _load_json(policy_path, "selective policy artifact")
    )
    _verify_pinned_file(
        config["calibration_data"], "calibration_data", extra_roots=(config_dir,)
    )

    # 3. The supplied dataset must hash to the pinned test_data sha256.
    if not os.path.isfile(data_path):
        raise ValueError("--data path %s is not an existing file" % data_path)
    data_digest = _sha256_file(data_path)
    if data_digest != test_pin_digest:
        raise ValueError(
            "--data file %s hashes to %s but the config pins test_data.sha256 %s"
            % (data_path, data_digest, test_pin_digest)
        )
    cases = _load_dataset(data_path)

    # 4. Controlled training-side corpus for the contamination screens.
    training_texts = _load_training_texts(config)

    # 5. Run the full verdict: split provenance and contamination screens run
    #    before any statistic, then the gates, calibration, and P1-P6 language.
    run_config = {
        "freeze_record": record,
        "cases": cases,
        "training_texts": training_texts,
        "policy": dict(config["policy"]),
        "calibration_data": dict(config["calibration_data"]),
        "test_data": dict(config["test_data"]),
        "temperature": temperature,
    }
    for key in _RUN_CONFIG_PASSTHROUGH:
        if key in config:
            run_config[key] = config[key]
    verdict = run_blind_verdict(run_config)

    # 6. Machine-readable result envelope (written once; never overwritten).
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS" if verdict["all_primary_gates_pass"] else "FAIL",
        "summary": verdict_summary(verdict),
        "config": {
            "path": os.path.abspath(config_path),
            "sha256": _sha256_file(config_path),
        },
        "dataset": {
            "path": os.path.abspath(data_path),
            "sha256": data_digest,
            "n_records": len(cases),
        },
        "training_texts_count": len(training_texts),
        "verdict": verdict,
    }
    with open(output_path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="research.veyabench.run_prospective_blind",
        description=(
            "Run the prospective blind verdict protocol (P1-P6) over a supplied "
            "target-domain labeled dataset. Exits 0 on PASS, 1 on FAIL, 2 on "
            "validation or runtime error."
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help="path to the frozen prospective blind run config JSON",
    )
    parser.add_argument(
        "--data", required=True,
        help="path to the target-domain labeled dataset (.jsonl or .json)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="path of the machine-readable result JSON (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        result = run(args.config, args.data, args.output)
    except FileExistsError:
        print(
            "error: result %s already exists; the prospective blind protocol runs once "
            "per output path (remove the file only under audit authority)" % args.output,
            file=sys.stderr,
        )
        return EXIT_ERROR
    except (ValueError, OSError) as error:
        print("error: %s" % error, file=sys.stderr)
        return EXIT_ERROR
    print(result["summary"])
    return EXIT_PASS if result["status"] == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
