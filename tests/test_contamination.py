import json

import numpy as np
import pytest

from research.veyabench.contamination import (
    REPORT_FORMAT,
    contamination_report,
    exact_overlap,
    fuzzy_match,
    ngram_overlap,
    semantic_neighbors,
)


def test_exact_overlap_flags_normalized_duplicates_only():
    blind = ["Book a table", "  book a table  ", "what time is it", "spell restaurant"]
    training = ["BOOK A TABLE", "make a reservation", "what is the time"]
    assert exact_overlap(blind, training) == {"book a table"}
    assert exact_overlap([], ["anything"]) == set()
    assert exact_overlap(["anything"], []) == set()


def test_ngram_overlap_computes_character_jaccard():
    blind = ["abcabc"]
    training = ["abcabd"]
    flagged = ngram_overlap(blind, training, n=3, threshold=0.75)
    assert [(pair["blind_index"], pair["training_index"]) for pair in flagged] == [
        (0, 0)
    ]
    assert flagged[0]["similarity"] == pytest.approx(0.75)
    assert ngram_overlap(blind, training, n=3, threshold=0.8) == []


def test_ngram_overlap_flags_near_duplicate_utterances():
    blind = ["book a flight to boston", "spell the word restaurant"]
    training = ["book a flight to boston ma", "turn on the kitchen light"]
    flagged = ngram_overlap(blind, training, n=5, threshold=0.8)
    assert [(pair["blind_index"], pair["training_index"]) for pair in flagged] == [
        (0, 0)
    ]
    assert flagged[0]["similarity"] == pytest.approx(19 / 22)


def test_fuzzy_match_flags_close_strings():
    blind = ["book a flight to paris", "spell the word restaurant"]
    training = ["book a flight to paris.", "make a reservation for two"]
    flagged = fuzzy_match(blind, training, threshold=0.9)
    assert [(pair["blind_index"], pair["training_index"]) for pair in flagged] == [
        (0, 0)
    ]
    assert flagged[0]["ratio"] == pytest.approx(44 / 45)


def test_fuzzy_match_threshold_controls_sensitivity():
    blind = ["book a flight to paris"]
    training = ["book a flight to paris tomorrow"]
    assert fuzzy_match(blind, training, threshold=0.9) == []
    flagged = fuzzy_match(blind, training, threshold=0.8)
    assert [(pair["blind_index"], pair["training_index"]) for pair in flagged] == [
        (0, 0)
    ]
    assert flagged[0]["ratio"] == pytest.approx(44 / 53)


def test_semantic_neighbors_flags_near_duplicate_vectors():
    blind = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    training = np.asarray([[0.9, 0.1], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    flagged = semantic_neighbors(blind, training, threshold=0.92)
    assert [(pair["blind_index"], pair["training_index"]) for pair in flagged] == [
        (0, 0),
        (0, 1),
    ]
    assert flagged[0]["cosine"] == pytest.approx(0.9 / np.sqrt(0.82))
    assert flagged[1]["cosine"] == pytest.approx(1.0)


def test_semantic_neighbors_handles_zero_vectors_and_rejects_bad_shapes():
    assert semantic_neighbors([[0.0, 0.0]], [[0.0, 0.0]], threshold=0.5) == []
    with pytest.raises(ValueError, match="two-dimensional"):
        semantic_neighbors([1.0, 0.0], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimensions do not match"):
        semantic_neighbors([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


def test_invalid_n_and_threshold_are_rejected():
    with pytest.raises(ValueError, match="n-gram size"):
        ngram_overlap(["abc"], ["abc"], n=0)
    with pytest.raises(ValueError, match="must lie in"):
        ngram_overlap(["abc"], ["abc"], threshold=1.5)
    with pytest.raises(ValueError, match="must lie in"):
        fuzzy_match(["abc"], ["abc"], threshold=-0.1)
    with pytest.raises(ValueError, match="must lie in"):
        semantic_neighbors([[1.0]], [[1.0]], threshold=1.1)


def test_contamination_report_summarizes_all_screens():
    blind_texts = ["book a table", "what time is it"]
    training_texts = ["BOOK A TABLE", "make a reservation"]
    blind_vectors = [[1.0, 0.0], [0.0, 1.0]]
    training_vectors = [[1.0, 0.0], [1.0, 1.0]]
    report = contamination_report(
        blind_texts, training_texts, blind_vectors, training_vectors
    )
    assert report["format"] == REPORT_FORMAT == "veya-contamination-report-v1"
    assert report["blind_n"] == 2
    assert report["training_n"] == 2
    assert report["checks"]["exact"] == {"count": 1, "indices": [0]}
    ngram = report["checks"]["ngram"]
    assert ngram["count"] == 1
    assert ngram["threshold"] == 0.8
    assert [
        (pair["blind_index"], pair["training_index"]) for pair in ngram["flagged"]
    ] == [(0, 0)]
    fuzzy = report["checks"]["fuzzy"]
    assert fuzzy["count"] == 1
    assert fuzzy["threshold"] == 0.9
    assert [
        (pair["blind_index"], pair["training_index"]) for pair in fuzzy["flagged"]
    ] == [(0, 0)]
    semantic = report["checks"]["semantic"]
    assert semantic["count"] == 1
    assert semantic["threshold"] == 0.92
    assert [
        (pair["blind_index"], pair["training_index"]) for pair in semantic["flagged"]
    ] == [(0, 0)]
    assert report["total_flagged"] == [0]
    assert report["veya_controlled_overlap"] == "1 cases flagged; see checks"
    assert report["base_model_pretraining"] == (
        "no detected evidence of base-model pretraining exposure"
    )
    json.dumps(report, sort_keys=True)


def test_contamination_report_unions_blind_indices_across_checks():
    blind_texts = ["Book A Table", "unrelated utterance"]
    training_texts = ["book a table", "unrelated training text"]
    blind_vectors = [[1.0, 0.0], [0.6, 0.8]]
    training_vectors = [[0.0, 1.0], [0.6, 0.8]]
    report = contamination_report(
        blind_texts, training_texts, blind_vectors, training_vectors
    )
    assert report["checks"]["exact"]["indices"] == [0]
    assert report["checks"]["ngram"]["count"] == 1
    assert report["checks"]["fuzzy"]["count"] == 1
    assert report["checks"]["semantic"]["count"] == 1
    assert report["checks"]["semantic"]["flagged"][0]["blind_index"] == 1
    assert report["total_flagged"] == [0, 1]
    assert report["veya_controlled_overlap"] == "2 cases flagged; see checks"


def test_contamination_report_verified_disjoint_without_vectors():
    blind = ["spell the word restaurant", "turn on the kitchen light"]
    training = ["book a flight to boston", "make a reservation for two"]
    report = contamination_report(blind, training)
    assert report["checks"]["semantic"] is None
    assert report["checks"]["exact"] == {"count": 0, "indices": []}
    assert report["checks"]["ngram"] == {"count": 0, "threshold": 0.8, "flagged": []}
    assert report["checks"]["fuzzy"] == {"count": 0, "threshold": 0.9, "flagged": []}
    assert report["total_flagged"] == []
    assert report["veya_controlled_overlap"] == "verified disjoint"
    assert report["base_model_pretraining"] == (
        "no detected evidence of base-model pretraining exposure"
    )


def test_contamination_report_handles_empty_inputs():
    report = contamination_report([], [])
    assert report["blind_n"] == 0
    assert report["training_n"] == 0
    assert report["total_flagged"] == []
    assert report["veya_controlled_overlap"] == "verified disjoint"
    assert report["checks"]["semantic"] is None


def test_contamination_report_rejects_unpaired_or_misaligned_vectors():
    with pytest.raises(ValueError, match="both blind and training vectors"):
        contamination_report(["a"], ["b"], blind_vectors=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="align with blind texts"):
        contamination_report(["a", "b"], ["c"], [[1.0, 0.0]], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="align with training texts"):
        contamination_report(
            ["a"], ["b", "c", "d"], [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]
        )
