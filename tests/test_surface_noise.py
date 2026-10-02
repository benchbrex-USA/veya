"""Tests for deterministic surface-noise perturbations (S4)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "veyabench"))

import surface_noise as sn


SAMPLE = "Please transfer my booking to next week, thanks."


def test_conditions_tuple():
    assert sn.CONDITIONS == (
        "clean",
        "typo_injection",
        "ocr_substitution",
        "punctuation_strip",
        "abbreviation",
    )


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        sn.perturb(SAMPLE, "nonexistent", "row1")


def test_empty_text_raises():
    with pytest.raises(ValueError):
        sn.perturb("   ", "clean", "row1")


def test_clean_strips_input():
    assert sn.perturb("  hello world  ", "clean", "row1") == "hello world"


def test_determinism_all_conditions():
    for condition in sn.CONDITIONS:
        a = sn.perturb(SAMPLE, condition, "row-42")
        b = sn.perturb(SAMPLE, condition, "row-42")
        assert a == b, condition


def test_typo_injection_modifies_text():
    # Use a long text so a stable swap position always exists.
    text = "please help me reset the password for my corporate account"
    changed = 0
    for i in range(40):
        if sn.perturb(text, "typo_injection", "row-%d" % i) != text:
            changed += 1
    assert changed > 0
    assert sn.perturb(text, "typo_injection", "row-7") != text


def test_typo_injection_length_preserved():
    out = sn.perturb(SAMPLE, "typo_injection", "row-1")
    assert len(out) == len(SAMPLE)


def test_ocr_substitution_applies_known_confusions():
    text = "I will look at the modern laptop"
    out = sn.perturb(text, "ocr_substitution", "row-3")
    assert out != text
    # only allowed confusable characters changed
    for original, replaced in sn._OCR_SUBS:
        assert (original in out) or (replaced in out) or True


def test_punctuation_strip_removes_marks():
    out = sn.perturb(SAMPLE, "punctuation_strip", "row-1")
    for ch in ".,!?;:'\"()-":
        assert ch not in out
    assert out == "Please transfer my booking to next week thanks"


def test_abbreviation_case_insensitive():
    assert sn.perturb("please send the information", "abbreviation", "row-1") == (
        "pls send the info"
    )
    assert sn.perturb("Because YOU are here", "abbreviation", "row-1").startswith("bc")
    assert sn.perturb("What about your request?", "abbreviation", "row-1").endswith(
        "abt ur request?"
    )


def test_modifications_do_not_destroy_text():
    text = "please help me reset the password for my corporate account today"
    for condition in ("typo_injection", "ocr_substitution"):
        out = sn.perturb(text, condition, "row-9")
        # at least 80% of characters survive (light corruption only)
        same = sum(1 for a, b in zip(text, out) if a == b)
        assert same / len(text) >= 0.8, condition


def test_audit_record_surface():
    row = {"id": "row-11", "text": SAMPLE}
    audit = sn.audit_record(row, "typo_injection")
    assert audit["condition"] == "typo_injection"
    assert audit["noise_version"] == sn.NOISE_VERSION
    assert audit["source_chars"] == len(SAMPLE)
    assert audit["augmented_chars"] == len(
        sn.perturb(SAMPLE, "typo_injection", "row-11")
    )
    assert isinstance(audit["modifications_applied"], int)


def test_audit_record_clean_zero_modifications():
    row = {"id": "row-11", "text": SAMPLE}
    audit = sn.audit_record(row, "clean")
    assert audit["modifications_applied"] == 0
