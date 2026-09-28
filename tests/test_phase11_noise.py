import pytest

from research.veyabench.phase11_noise import (
    CONDITIONS,
    NOISE_VERSION,
    TRAINABLE_CONDITIONS,
    audit_record,
    perturb,
    training_conditions,
    validate_augmentation,
)


ROW = {"id": "row-7", "text": "please reverse the duplicate charge", "candidate_id": "refund"}


def test_every_noise_condition_is_deterministic_and_label_preserving():
    for condition in CONDITIONS:
        first = perturb(ROW["text"], condition, ROW["id"])
        second = perturb(ROW["text"], condition, ROW["id"])
        assert first == second
        assert first.count(ROW["text"]) == 1
        audit = audit_record(ROW, condition)
        assert audit["label_preserved"] is True
        assert audit["candidate_id"] == "refund"
        assert audit["noise_version"] == NOISE_VERSION


def test_training_curriculum_is_bounded_and_uses_allowed_conditions():
    selected = training_conditions(ROW["id"])
    assert selected[0] == "clean"
    assert len(selected) == 3
    assert set(selected[1:]).issubset(TRAINABLE_CONDITIONS)
    assert training_conditions(ROW["id"]) == selected


def test_noise_has_no_candidate_or_label_input():
    changed_label = {**ROW, "candidate_id": "different-secret-label"}
    for condition in CONDITIONS:
        assert validate_augmentation(ROW, condition) == validate_augmentation(
            changed_label, condition
        )


def test_noise_rejects_empty_or_unknown_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        perturb(" ", "clean", "row")
    with pytest.raises(ValueError, match="unknown Phase-11"):
        perturb("signal", "label_hint", "row")
