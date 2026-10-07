"""Phase 13 sealed counterfactual bank, metrics, and pairwise loss."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.veyabench.phase13_counterfactual import (  # noqa: E402
    CANDIDATES,
    GATES,
    HIGH_CONFIDENCE_FLOOR,
    INSTRUCTION,
    TRANSCRIPT,
    build_all,
    dumps_bank,
    evaluate,
    index_of,
    normalize_state,
    pairwise_hinge_loss,
    verify_seal,
    write_banks,
)


ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "data", "phase13")


@pytest.fixture(scope="module")
def banks():
    return build_all()


def test_preregistered_gates_stay_fixed():
    assert GATES == {
        "overall": 0.98,
        "negation": 0.97,
        "resolved_vs_active": 0.97,
        "paraphrase": 0.98,
        "prompt_injection": 0.98,
        "high_confidence_error_rate": 0.005,
    }
    assert HIGH_CONFIDENCE_FLOOR == 0.90
    assert INSTRUCTION == "Which team should take the next unresolved action?"
    assert set(CANDIDATES) == {"billing", "technical", "sales", "none"}


def test_bank_is_deterministic_and_in_the_requested_band(banks):
    again = build_all()
    assert dumps_bank(banks["sealed_eval"]) == dumps_bank(again["sealed_eval"])
    assert dumps_bank(banks["train"]) == dumps_bank(again["train"])
    assert dumps_bank(banks["calibration"]) == dumps_bank(again["calibration"])
    assert 3000 <= len(banks["sealed_eval"]) <= 10000
    assert banks["train"] and banks["calibration"]
    assert len(banks["train"]) < len(banks["sealed_eval"])


def test_splits_are_disjoint_and_anchors_do_not_cross(banks):
    surfaces = {
        split: {normalize_state(row["state"]) for row in rows}
        for split, rows in banks.items()
    }
    assert not surfaces["sealed_eval"] & surfaces["train"]
    assert not surfaces["sealed_eval"] & surfaces["calibration"]
    assert not surfaces["train"] & surfaces["calibration"]
    from research.veyabench.phase13_counterfactual import CALIBRATION_GOALS, EVAL_GOALS, TRAIN_GOALS

    blobs = {
        split: "\n".join(row["state"].casefold() for row in rows)
        for split, rows in banks.items()
    }
    anchors = {
        "sealed_eval": [goal["anchor"] for goal in EVAL_GOALS],
        "train": [goal["anchor"] for goal in TRAIN_GOALS],
        "calibration": [goal["anchor"] for goal in CALIBRATION_GOALS],
    }
    for split, blob in blobs.items():
        for other, other_anchors in anchors.items():
            if other == split:
                continue
            for anchor in other_anchors:
                assert anchor.casefold() not in blob


def test_every_goal_sentence_keeps_its_anchor_and_label(banks):
    from research.veyabench.phase13_counterfactual import EVAL_GOALS

    bare = {
        (row["counterfactual_id"], row["paraphrase_index"]): row
        for row in banks["sealed_eval"]
        if row["variant"] == "normal"
    }
    for goal in EVAL_GOALS:
        for index, sentence in enumerate(goal["sentences"]):
            row = bare[(goal["goal_id"], index)]
            assert row["state"] == sentence
            assert row["target"] == goal["label"]
            assert goal["anchor"].casefold() in sentence.casefold()
            assert row["target"] in {item[0] for item in row["criteria"]}


def test_missing_injection_and_resolved_rows_keep_their_contracts(banks):
    sealed = banks["sealed_eval"]
    missing = [row for row in sealed if "missing_correct_candidate" in row["tags"]]
    assert missing
    for row in missing:
        assert row["target"] == "none"
        assert row["dropped_candidate"] not in {item[0] for item in row["criteria"]}
        assert row["criteria_key"] != "billing|technical|sales|none"
    injections = [row for row in sealed if row["variant"] == "injection"]
    assert injections
    for row in injections:
        assert row["injected_team"] != row["target"]
        assert row["injected_team"] in {item[0] for item in row["criteria"]}
    resolved = [row for row in sealed if "resolved_vs_active" in row["tags"]]
    assert len(resolved) > 1000
    for row in resolved:
        assert row["resolved_domains"]
        assert row["target"] not in row["resolved_domains"]
        assert row["hard_negative"] in row["resolved_domains"]


def test_transcript_and_family_are_regression_rows(banks):
    sealed = banks["sealed_eval"]
    transcript = next(row for row in sealed if row["state"] == TRANSCRIPT)
    assert transcript["target"] == "technical"
    assert "regression" in transcript["tags"]
    assert transcript["resolved_domains"] == ["billing"]
    assert transcript["hard_negative"] == "billing"
    family = [
        row
        for row in sealed
        if "minimal:invoice_status" in row["minimal_pair_ids"]
    ]
    assert {row["target"] for row in family} == {"billing", "technical", "sales"}
    api_regression = [
        row
        for row in sealed
        if row["counterfactual_id"] == "api_401" and "regression" in row["tags"]
    ]
    assert 10 <= len(api_regression) <= 30
    assert all(row["instructions"] == INSTRUCTION for row in sealed)
    assert "not evidence" not in INSTRUCTION


def test_hard_negative_index_is_a_rival(banks):
    for row in banks["sealed_eval"][:50]:
        assert index_of(row, row["hard_negative"]) != index_of(row, row["target"])


def test_metrics_on_toy_predictions():
    rows = [
        {
            "id": "a",
            "target": "technical",
            "slice": "normal",
            "variant": "normal",
            "tags": ["normal", "paraphrase", "minimal_pair"],
            "counterfactual_id": "api_401",
            "paraphrase_index": 0,
            "criteria_key": "full",
            "minimal_pair_ids": ["contrast"],
        },
        {
            "id": "b",
            "target": "technical",
            "slice": "resolved_vs_active",
            "variant": "resolved",
            "tags": ["resolved_vs_active", "counterfactual"],
            "counterfactual_id": "api_401",
            "paraphrase_index": 0,
            "criteria_key": "full",
            "minimal_pair_ids": ["contrast"],
        },
        {
            "id": "c",
            "target": "billing",
            "slice": "normal",
            "variant": "normal",
            "tags": ["normal", "paraphrase", "minimal_pair"],
            "counterfactual_id": "unpaid_invoice",
            "paraphrase_index": 0,
            "criteria_key": "full",
            "minimal_pair_ids": ["contrast"],
        },
        {
            "id": "d",
            "target": "none",
            "slice": "missing_correct_candidate",
            "variant": "missing",
            "tags": ["missing_correct_candidate"],
            "counterfactual_id": "api_401",
            "paraphrase_index": 0,
            "criteria_key": "dropped",
            "minimal_pair_ids": [],
        },
    ]
    probs = {"billing": 0.97, "technical": 0.02, "sales": 0.01, "none": 0.0}

    def prediction(answer, confidence=0.9):
        return {
            "answer": answer,
            "probabilities": dict(probs),
            "confidence": confidence,
            "deferred": False,
        }

    swung = {
        "a": prediction("technical", 0.4),
        "b": prediction("billing", 0.95),
        "c": prediction("billing", 0.95),
        "d": prediction("billing", 0.95),
    }
    swung["a"]["probabilities"] = {"billing": 0.1, "technical": 0.8, "sales": 0.05, "none": 0.05}
    report = evaluate(rows, swung)
    assert report["counterfactual"]["consistency"] == 0.0
    assert report["minimal_pairs"]["collapse_rate"] == 0.0
    assert report["overall_accuracy"] == 0.5
    assert report["gates"]["passed"] is False
    assert report["temperature_fit"] is False
    assert report["high_confidence_error_rate_max_probability"]["errors"] == 2

    perfect = {}
    for row in rows:
        perfect[row["id"]] = prediction(row["target"], 0.2)
        perfect[row["id"]]["probabilities"] = {
            "billing": 0.1,
            "technical": 0.1,
            "sales": 0.1,
            "none": 0.1,
        }
        perfect[row["id"]]["probabilities"][row["target"]] = 0.7
    # One family cannot show slice gates passing on four rows of mixed slices.
    # A separate perfect bank covers the pre-registered thresholds.
    perfect_report = evaluate(rows, perfect, gates={})
    assert perfect_report["overall_accuracy"] == 1.0
    assert perfect_report["counterfactual"]["consistency"] == 1.0
    assert perfect_report["brier"] is not None
    assert perfect_report["ece_max_probability"] is not None


def test_perfect_predictions_meet_preregistered_gates():
    slices = {
        "normal": ("paraphrase", "normal"),
        "negation": ("negation",),
        "resolved_vs_active": ("resolved_vs_active",),
        "prompt_injection": ("prompt_injection",),
        "misleading_keywords": ("misleading_keywords",),
    }
    rows = []
    predictions = {}
    number = 0
    for slice_name, tags in slices.items():
        for _ in range(4):
            number += 1
            row_id = f"r{number}"
            rows.append(
                {
                    "id": row_id,
                    "target": "technical",
                    "slice": slice_name,
                    "variant": "normal" if slice_name == "normal" else slice_name,
                    "tags": list(tags),
                    "counterfactual_id": "g",
                    "paraphrase_index": 0,
                    "criteria_key": "full",
                    "minimal_pair_ids": [],
                }
            )
            predictions[row_id] = {
                "answer": "technical",
                "probabilities": {"technical": 0.97, "billing": 0.03},
                "confidence": 0.8,
                "deferred": False,
            }
    report = evaluate(rows, predictions)
    assert report["gates"]["passed"] is True
    assert report["high_confidence_error_rate_max_probability"]["rate"] == 0.0


def test_pairwise_hinge_adds_margin_only_when_the_rival_is_close():
    torch = pytest.importorskip("torch")
    easy = torch.tensor([0.0, 4.0, 1.0], dtype=torch.float64)
    loss = pairwise_hinge_loss(easy, 1, 0, margin=0.5)
    expected = -torch.log_softmax(easy, dim=-1)[1]
    assert torch.allclose(loss, expected)

    contested = torch.tensor([3.0, 0.2, 0.0], dtype=torch.float64, requires_grad=True)
    contested_loss = pairwise_hinge_loss(contested, 1, 0, margin=0.5)
    hinge = 0.5 - 0.2 + 3.0
    assert contested_loss.item() > hinge
    contested_loss.backward()
    assert contested.grad[1].item() < 0
    assert contested.grad[0].item() > 0
    with pytest.raises(ValueError):
        pairwise_hinge_loss(easy, 1, 1)


def test_written_seal_matches_the_generator(tmp_path):
    document = write_banks(tmp_path)
    verified = verify_seal(tmp_path)
    assert verified["sha256"] == document["sha256"]
    assert verified["train_prohibited"] is True
    assert verified["temperature_fit_prohibited"] is True
    assert verified["provenance"]["not_a_human_rater_panel"] is True
    assert verified["provenance"]["human_rater_panel"] is False
    assert verified["abstention"]["fitted_margin_threshold"] is None
    tampered = tmp_path / "sealed_eval.jsonl"
    payload = bytearray(tampered.read_bytes())
    payload[-2:] = b"x\n"
    tampered.write_bytes(payload)
    with pytest.raises(ValueError, match="hash"):
        verify_seal(tmp_path)


def test_runner_does_not_fit_a_temperature():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "research",
        "veyabench",
        "run_phase13_baseline.py",
    )
    source = open(path, encoding="utf-8").read()
    assert "fit_temperature" not in source


def test_committed_banks_match_the_generator_when_present(banks):
    if not os.path.exists(os.path.join(ROOT, "seal.json")):
        pytest.skip("phase 13 banks have not been written")
    import json

    verify_seal(__import__("pathlib").Path(ROOT))
    on_disk = open(os.path.join(ROOT, "sealed_eval.jsonl"), "rb").read()
    assert on_disk == dumps_bank(banks["sealed_eval"])
    seal = json.loads(open(os.path.join(ROOT, "seal.json"), encoding="utf-8").read())
    assert seal["gates"] == GATES
    assert copy.deepcopy(seal["provenance"]["not_a_human_rater_panel"]) is True
