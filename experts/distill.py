"""Offline teacher records. Generative output is not ground truth."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def teacher_record(
    *,
    teacher_model: str,
    teacher_version: str,
    prompt_version: str,
    sampling: dict[str, Any],
    teacher_output: dict[str, Any],
    accepted_label: dict[str, Any] | None,
    human_verified: bool,
    human_corrections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not human_verified or accepted_label is None:
        raise ValueError("human verification and an accepted label are required")
    teacher = json.loads(json.dumps(teacher_output))
    label = json.loads(json.dumps(accepted_label))
    return {
        "record_type": "expert_teacher_example",
        "teacher_model": teacher_model,
        "teacher_version": teacher_version,
        "prompt_version": prompt_version,
        "sampling": sampling,
        "teacher_output": teacher,
        "accepted_label": label,
        "teacher_is_ground_truth": False,
        "human_verified": True,
        "human_corrections": list(human_corrections or []),
        "input_sha256": _hash(sampling),
        "teacher_sha256": _hash(teacher),
        "label_sha256": _hash(label),
        "trainable": True,
    }


def raw_teacher_output_is_trainable(record: dict[str, Any]) -> bool:
    return bool(record.get("human_verified") and record.get("trainable") and record.get("teacher_is_ground_truth") is False)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
