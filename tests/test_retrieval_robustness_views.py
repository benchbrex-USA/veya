"""Perturbations keep the label outside the transform and views stay deterministic."""

import numpy as np

from research.veyabench.retrieval_robustness import (
    adversarial_suffix,
    apply_perturbation,
    choose_wrong_index,
    classify_outcome,
    fusion_recommendation,
    query_views,
)


def test_perturbations_preserve_the_source_text_and_are_stable():
    text = "Which countries are represented?"
    for kind in ("clean", "short_suffix", "long_suffix", "prefix", "json_wrapper"):
        once = apply_perturbation(text, kind)
        twice = apply_perturbation(text, kind)
        assert once == twice
        if kind == "json_wrapper":
            assert query_views(once)["json_text"] == text
        elif kind == "clean":
            assert once == text
        else:
            assert text in once
            assert once != text


def test_adversarial_suffix_names_a_different_intent_and_fusion_is_the_max():
    gold = "card arrival"
    wrong = "cash withdrawal"
    suffix = adversarial_suffix(wrong)
    assert gold not in suffix
    assert wrong in suffix
    ranked = [0, 2, 1]
    naturalized = ["card arrival", "cash withdrawal", "wire transfer"]
    assert choose_wrong_index(ranked, 0, naturalized) == 2
    assert choose_wrong_index(ranked, 0, naturalized) != 0
    scores = [
        np.asarray([[0.1, 0.4], [0.2, 0.3]]),
        np.asarray([[0.5, 0.0], [0.0, 0.9]]),
    ]
    fused = np.max(np.stack(scores, axis=0), axis=0)
    assert fused.tolist() == [[0.5, 0.4], [0.2, 0.9]]
    captured = classify_outcome(
        baseline_correct=True,
        fusion_correct=False,
        baseline_hit=True,
        fusion_hit=True,
        adversarial=True,
        prediction="clinc150:cash_withdrawal",
        injected="clinc150:cash_withdrawal",
    )
    assert captured["primary"] == "BAD_FLIP"
    assert captured["adversarial_capture"] is True
    assert fusion_recommendation(
        {
            "short_e2e_delta": 0.02,
            "long_e2e_delta": 0.03,
            "clean_e2e_delta": 0.0,
            "long_recall8_delta_vs_phase10_full": 0.2,
            "fusion_harm_rate": 0.04,
            "production_harm_rate": 0.03,
            "fusion_wrong_top1": 0.05,
            "production_wrong_top1": 0.04,
        }
    ) == "CANDIDATE"
    assert fusion_recommendation(
        {
            "short_e2e_delta": -0.01,
            "long_e2e_delta": -0.02,
            "clean_e2e_delta": -0.02,
            "long_recall8_delta_vs_phase10_full": -0.1,
            "fusion_harm_rate": 0.20,
            "production_harm_rate": 0.05,
            "fusion_wrong_top1": 0.30,
            "production_wrong_top1": 0.05,
        }
    ) == "REJECT"


def test_suffix_head_is_the_observed_first_line_not_a_side_channel():
    text = "I need a refund"
    noisy = apply_perturbation(text, "long_suffix")
    views = query_views(noisy)
    assert views["head"] == text
    assert "Unrelated operations note" in views["tail"]
    assert "refund" not in views["tail"]
