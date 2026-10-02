"""Deterministic template-paraphrase perturbations (S5) for blind benchmarking."""
from __future__ import annotations

import hashlib
from typing import Any


NOISE_VERSION = "blind-template-paraphrase-v1"
CONDITIONS = (
    "clean",
    "polite_prefix",
    "formal_rewrite",
    "casual_rewrite",
    "question_form",
)

_POLITE_PREFIXES = (
    "I need help with: ",
    "Could you please handle: ",
    "Hi, regarding: ",
    "I'd like assistance with: ",
)

_CASUAL_WRAPPERS = (
    "hey so basically {text}",
    "yo i got this thing - {text}",
)

_QUESTION_FORMS = (
    "What should be done about: {text}?",
    "How would you categorize: {text}?",
)


def _stable_index(row_id: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(
        (NOISE_VERSION + "\0" + namespace + "\0" + row_id).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % size


def perturb(text: str, condition: str, row_id: str) -> str:
    """Wrap the source text in a deterministic paraphrase frame, preserving it verbatim."""
    if condition not in CONDITIONS:
        raise ValueError("unknown paraphrase condition %r" % condition)
    clean = str(text).strip()
    if not clean:
        raise ValueError("source text must not be empty")
    if condition == "clean":
        return clean
    if condition == "polite_prefix":
        prefix = _POLITE_PREFIXES[_stable_index(row_id, "polite", len(_POLITE_PREFIXES))]
        return prefix + clean
    if condition == "formal_rewrite":
        return (
            "The following matter requires attention: "
            + clean
            + ". Please advise accordingly."
        )
    if condition == "casual_rewrite":
        template = _CASUAL_WRAPPERS[
            _stable_index(row_id, "casual", len(_CASUAL_WRAPPERS))
        ]
        return template.format(text=clean)
    if condition == "question_form":
        template = _QUESTION_FORMS[
            _stable_index(row_id, "question", len(_QUESTION_FORMS))
        ]
        return template.format(text=clean)
    raise AssertionError("condition dispatch is incomplete")


def audit_record(row: dict[str, Any], condition: str) -> dict[str, Any]:
    """Build an audit entry, verifying the source text appears exactly once."""
    text = str(row["text"]).strip()
    augmented = perturb(text, condition, str(row["id"]))
    occurrences = augmented.count(text)
    return {
        "row_id": str(row["id"]),
        "condition": condition,
        "source_chars": len(text),
        "augmented_chars": len(augmented),
        "source_occurrences": occurrences,
        "label_preserved": occurrences == 1,
        "noise_version": NOISE_VERSION,
    }


def validate_augmentation(row: dict[str, Any], condition: str) -> str:
    audit = audit_record(row, condition)
    if not audit["label_preserved"]:
        raise ValueError("augmentation did not preserve source text exactly once")
    return perturb(str(row["text"]), condition, str(row["id"]))
