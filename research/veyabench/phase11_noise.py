"""Deterministic, label-agnostic query noise for Phase 11 research."""
from __future__ import annotations

import hashlib
import json
from typing import Any


NOISE_VERSION = "phase11-generic-noise-v1"
SHORT_NOTE = "metadata: request received; display theme blue; internal reference 000000"
LONG_NOTE = "unrelated operational metadata contains no routing signal. " * 80
BOILERPLATE = "system record follows; validate fields before processing"
CONDITIONS = (
    "clean",
    "short_prefix",
    "short_suffix",
    "long_prefix",
    "long_suffix",
    "middle_insertion",
    "duplicate_metadata",
    "reordered_fields",
    "boilerplate_wrapper",
    "json_noise",
    "repeated_low_information",
)
TRAINABLE_CONDITIONS = (
    "short_prefix",
    "short_suffix",
    "middle_insertion",
    "boilerplate_wrapper",
    "long_suffix",
    "repeated_low_information",
)


def _stable_index(row_id: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(
        (NOISE_VERSION + "\0" + namespace + "\0" + row_id).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % size


def perturb(text: str, condition: str, row_id: str) -> str:
    """Apply fixed generic noise while retaining the source text verbatim."""
    if condition not in CONDITIONS:
        raise ValueError("unknown Phase-11 noise condition %r" % condition)
    clean = str(text).strip()
    if not clean:
        raise ValueError("source text must not be empty")
    if condition == "clean":
        return clean
    if condition == "short_prefix":
        return SHORT_NOTE + "\n" + clean
    if condition == "short_suffix":
        return clean + "\n" + SHORT_NOTE
    if condition == "long_prefix":
        return LONG_NOTE + "\n" + clean
    if condition == "long_suffix":
        return clean + "\n" + LONG_NOTE
    if condition == "middle_insertion":
        return "record header\n" + clean + "\n" + SHORT_NOTE + "\nrecord footer"
    if condition == "duplicate_metadata":
        return SHORT_NOTE + "\n" + SHORT_NOTE + "\n" + clean
    if condition == "reordered_fields":
        fields = [
            "request_id: 000000",
            "display_theme: blue",
            "payload: " + clean,
        ]
        rotation = _stable_index(row_id, condition, len(fields))
        fields = fields[rotation:] + fields[:rotation]
        return "\n".join(fields)
    if condition == "boilerplate_wrapper":
        return BOILERPLATE + "\n" + clean + "\nend system record"
    if condition == "json_noise":
        return json.dumps(
            {
                "request_id": "000000",
                "display_theme": "blue",
                "payload": clean,
                "attempt": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if condition == "repeated_low_information":
        return clean + "\n" + ("status pending. " * 120)
    raise AssertionError("condition dispatch is incomplete")


def training_conditions(row_id: str) -> tuple[str, str, str]:
    """Return a deterministic modest-to-long curriculum for one TRAIN row."""
    short = ("short_prefix", "short_suffix", "middle_insertion", "boilerplate_wrapper")
    long = ("long_suffix", "repeated_low_information")
    return (
        "clean",
        short[_stable_index(row_id, "train-short", len(short))],
        long[_stable_index(row_id, "train-long", len(long))],
    )


def audit_record(row: dict[str, Any], condition: str) -> dict[str, Any]:
    text = str(row["text"]).strip()
    augmented = perturb(text, condition, str(row["id"]))
    if condition == "json_noise":
        preserved = json.loads(augmented).get("payload") == text
        occurrences = int(preserved)
    else:
        occurrences = augmented.count(text)
        preserved = occurrences == 1
    return {
        "row_id": str(row["id"]),
        "candidate_id": str(row["candidate_id"]),
        "condition": condition,
        "source_chars": len(text),
        "augmented_chars": len(augmented),
        "source_occurrences": occurrences,
        "label_preserved": preserved,
        "noise_version": NOISE_VERSION,
    }


def validate_augmentation(row: dict[str, Any], condition: str) -> str:
    audit = audit_record(row, condition)
    if not audit["label_preserved"]:
        raise ValueError("augmentation did not preserve source text exactly once")
    return perturb(str(row["text"]), condition, str(row["id"]))
