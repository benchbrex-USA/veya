"""Tests for deterministic template-paraphrase perturbations (S5)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "veyabench"))

import paraphrase_noise as pn


SAMPLE = "my flight refund has not arrived"


def test_conditions_tuple():
    assert pn.CONDITIONS == (
        "clean",
        "polite_prefix",
        "formal_rewrite",
        "casual_rewrite",
        "question_form",
    )


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        pn.perturb(SAMPLE, "nonexistent", "row1")


def test_empty_text_raises():
    with pytest.raises(ValueError):
        pn.perturb("  ", "clean", "row1")


def test_clean_strips_input():
    assert pn.perturb("  hello  ", "clean", "row1") == "hello"


def test_determinism_all_conditions():
    for condition in pn.CONDITIONS:
        a = pn.perturb(SAMPLE, condition, "row-42")
        b = pn.perturb(SAMPLE, condition, "row-42")
        assert a == b, condition


def test_source_preserved_verbatim_all_conditions():
    for condition in pn.CONDITIONS:
        out = pn.perturb(SAMPLE, condition, "row-42")
        assert SAMPLE in out, condition
        assert out.count(SAMPLE) == 1, condition


def test_each_condition_differs_from_clean():
    clean = pn.perturb(SAMPLE, "clean", "row-42")
    for condition in pn.CONDITIONS[1:]:
        assert pn.perturb(SAMPLE, condition, "row-42") != clean, condition


def test_polite_prefix_uses_known_intros():
    for i in range(20):
        out = pn.perturb(SAMPLE, "polite_prefix", "row-%d" % i)
        assert any(out.startswith(p) for p in pn._POLITE_PREFIXES), out


def test_formal_rewrite_framing():
    out = pn.perturb(SAMPLE, "formal_rewrite", "row-1")
    assert out == (
        "The following matter requires attention: "
        + SAMPLE
        + ". Please advise accordingly."
    )


def test_casual_rewrite_uses_known_frames():
    for i in range(20):
        out = pn.perturb(SAMPLE, "casual_rewrite", "row-%d" % i)
        assert out.startswith(("hey so basically ", "yo i got this thing - ")), out


def test_question_form_ends_with_question_mark():
    for i in range(20):
        out = pn.perturb(SAMPLE, "question_form", "row-%d" % i)
        assert out.endswith("?"), out
        assert SAMPLE in out


def test_audit_record_verifies_single_occurrence():
    row = {"id": "row-11", "text": SAMPLE}
    for condition in pn.CONDITIONS:
        audit = pn.audit_record(row, condition)
        assert audit["label_preserved"] is True, condition
        assert audit["source_occurrences"] == 1, condition
        assert audit["noise_version"] == pn.NOISE_VERSION


def test_validate_augmentation_returns_string():
    row = {"id": "row-11", "text": SAMPLE}
    for condition in pn.CONDITIONS:
        out = pn.validate_augmentation(row, condition)
        assert isinstance(out, str) and SAMPLE in out, condition
