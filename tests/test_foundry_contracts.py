import pytest

from veya.foundry import (
    VeyaDistribution,
    VeyaQuestion,
    VeyaQuestionType,
    VeyaRequest,
)


def test_distribution_normalizes_mapping():
    d = VeyaDistribution.from_mapping({"a": 2.0, "b": 1.0})
    assert d.selected == "a"
    assert d.as_mapping()["a"] == pytest.approx(2 / 3)


def test_choice_requires_options():
    with pytest.raises(ValueError):
        VeyaQuestion(
            question_id="q",
            question_type=VeyaQuestionType.CHOICE,
            instruction="pick",
        )


def test_boolean_uses_fixed_labels():
    q = VeyaQuestion(
        question_id="q",
        question_type=VeyaQuestionType.BOOLEAN,
        instruction="is it true",
    )
    assert q.normalized_options() == ("false", "true")


def test_request_requires_unique_ids():
    q = VeyaQuestion(
        question_id="q",
        question_type=VeyaQuestionType.BOOLEAN,
        instruction="is it true",
    )
    with pytest.raises(ValueError):
        VeyaRequest(state="x", questions=(q, q))
