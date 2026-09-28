import copy

import numpy as np
import pytest

from research.veyabench.phase11_protocol import (
    HARD_NEGATIVE_VERSION,
    mine_hard_negatives_with_lineage,
    token_length_bucket,
    tokenization_audit,
    validate_hard_negative_lineage,
)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False):
        del truncation
        ids = list(range(len(text)))
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [-1, *ids, -2]
            offsets = [(0, 0), *offsets, (0, 0)]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_tokenization_audit_detects_visible_partial_and_dropped_signal():
    tokenizer = CharacterTokenizer()
    visible = tokenization_audit(tokenizer, "signal", "signal noise", max_length=8, view_count=1)
    dropped = tokenization_audit(tokenizer, "signal", "x" * 20 + "signal", max_length=8, view_count=4)
    assert visible["positive_signal_position"] == "fully_retained"
    assert dropped["positive_signal_position"] == "truncated"
    assert dropped["tokens_dropped"] > 0
    assert dropped["view_count"] == 4


def test_hard_negative_lineage_is_train_only_filtered_and_tamper_evident():
    rows = [{"id": "q1", "candidate_id": "s:a", "schema": "s"}]
    candidates = [
        {"candidate_id": "s:a", "raw": "alpha", "schema": "s"},
        {"candidate_id": "s:a2", "raw": "alpha", "schema": "s"},
        {"candidate_id": "s:b", "raw": "beta", "schema": "s"},
    ]
    scores = np.asarray([[1.0, 0.99, 0.8]], dtype=np.float32)
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    selected, lineage = mine_hard_negatives_with_lineage(
        scores, rows, candidates, vectors, retriever_id="rb1", limit=1
    )
    assert selected == {"q1": [2]}
    assert lineage["hard_negative_version"] == HARD_NEGATIVE_VERSION
    assert lineage["rejected"]["normalized_duplicate"] == 1
    validate_hard_negative_lineage(lineage)
    tampered = copy.deepcopy(lineage)
    tampered["records"][0]["rank"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_hard_negative_lineage(tampered)


@pytest.mark.parametrize(
    ("length", "expected"),
    [(0, "0-32"), (32, "0-32"), (33, "33-64"), (128, "65-128"), (257, "257+")],
)
def test_token_length_buckets(length, expected):
    assert token_length_bucket(length) == expected
