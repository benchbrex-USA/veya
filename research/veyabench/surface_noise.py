"""Deterministic surface-noise perturbations (S4) for blind benchmarking."""
from __future__ import annotations

import hashlib
from typing import Any


NOISE_VERSION = "blind-surface-noise-v1"
CONDITIONS = (
    "clean",
    "typo_injection",
    "ocr_substitution",
    "punctuation_strip",
    "abbreviation",
)

_OCR_SUBS = (
    ("l", "1"),
    ("O", "0"),
    ("rn", "m"),
    ("I", "l"),
)

_ABBREVIATIONS = {
    "please": "pls",
    "information": "info",
    "because": "bc",
    "you": "u",
    "your": "ur",
    "are": "r",
    "about": "abt",
}

_PUNCTUATION = ".,!?;:'\"()-"


def _stable_index(row_id: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(
        (NOISE_VERSION + "\0" + namespace + "\0" + row_id).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % size


def _typo_injection(clean: str, row_id: str) -> tuple[str, int]:
    if len(clean) < 4:
        return clean, 0
    n = 1 + _stable_index(row_id, "typo-count", 2)
    lo, hi = 1, len(clean) - 2  # avoid first/last char
    span = hi - lo
    if span <= 0:
        return clean, 0
    modifications = 0
    chars = list(clean)
    for k in range(n):
        idx = lo + _stable_index(row_id, "typo-pos-%d" % k, span)
        if idx + 1 >= len(chars) or chars[idx] == chars[idx + 1]:
            continue
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        modifications += 1
    return "".join(chars), modifications


def _ocr_substitution(clean: str, row_id: str) -> tuple[str, int]:
    positions = [
        i
        for i in range(len(clean))
        if clean[i : i + 2] == "rn"
        or clean[i] in "lO"
        or (clean[i] == "I")
    ]
    if not positions:
        return clean, 0
    n = min(1 + _stable_index(row_id, "ocr-count", 2), len(positions))
    modifications = 0
    chars = list(clean)
    for k in range(n):
        pos = positions[_stable_index(row_id, "ocr-pos-%d" % k, len(positions))]
        if chars[pos] is None:
            continue
        if chars[pos : pos + 2] == ["r", "n"]:
            chars[pos : pos + 2] = ["m", None]
            modifications += 1
        elif chars[pos] == "l":
            chars[pos] = "1"
            modifications += 1
        elif chars[pos] == "O":
            chars[pos] = "0"
            modifications += 1
        elif chars[pos] == "I":
            chars[pos] = "l"
            modifications += 1
    return "".join(c for c in chars if c is not None), modifications


def _punctuation_strip(clean: str) -> str:
    return "".join(ch for ch in clean if ch not in _PUNCTUATION)


def _abbreviation(clean: str) -> str:
    words = clean.split()
    out = []
    for word in words:
        key = word.strip(_PUNCTUATION)
        repl = _ABBREVIATIONS.get(key.lower())
        if repl:
            out.append(word.replace(key, repl, 1))
        else:
            out.append(word)
    return " ".join(out)


def perturb(text: str, condition: str, row_id: str) -> str:
    """Apply a deterministic surface-noise perturbation to the source text."""
    if condition not in CONDITIONS:
        raise ValueError("unknown surface-noise condition %r" % condition)
    clean = str(text).strip()
    if not clean:
        raise ValueError("source text must not be empty")
    if condition == "clean":
        return clean
    if condition == "typo_injection":
        result, _ = _typo_injection(clean, row_id)
        return result
    if condition == "ocr_substitution":
        result, _ = _ocr_substitution(clean, row_id)
        return result
    if condition == "punctuation_strip":
        return _punctuation_strip(clean)
    if condition == "abbreviation":
        return _abbreviation(clean)
    raise AssertionError("condition dispatch is incomplete")


def modifications_applied(text: str, condition: str, row_id: str) -> int:
    """Count how many discrete edits the given condition applied."""
    clean = str(text).strip()
    if condition == "typo_injection":
        return _typo_injection(clean, row_id)[1]
    if condition == "ocr_substitution":
        return _ocr_substitution(clean, row_id)[1]
    if condition == "punctuation_strip":
        return sum(1 for ch in clean if ch in _PUNCTUATION)
    if condition == "abbreviation":
        original = clean.split()
        modified = perturb(clean, condition, row_id).split()
        return sum(1 for a, b in zip(original, modified) if a != b)
    return 0


def audit_record(row: dict[str, Any], condition: str) -> dict[str, Any]:
    """Build an audit entry for a surface-noise augmentation."""
    text = str(row["text"]).strip()
    augmented = perturb(text, condition, str(row["id"]))
    mods = modifications_applied(text, condition, str(row["id"]))
    return {
        "row_id": str(row["id"]),
        "condition": condition,
        "source_chars": len(text),
        "augmented_chars": len(augmented),
        "noise_version": NOISE_VERSION,
        "modifications_applied": mods,
    }
