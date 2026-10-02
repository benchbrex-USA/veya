"""Contamination screens for blind Veya evaluations against controlled training partitions."""

from __future__ import annotations

import difflib
from typing import Any

import numpy as np

REPORT_FORMAT = "veya-contamination-report-v1"


def _normalize_text(text: str) -> str:
    """Strip and casefold so surface variation cannot mask a duplicate."""
    return str(text).strip().casefold()


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity thresholds must lie in [0, 1]")


def _character_ngrams(text: str, n: int) -> set[str]:
    if n < 1:
        raise ValueError("n-gram size must be positive")
    return {text[start : start + n] for start in range(len(text) - n + 1)}


def _normalized_ngram_sets(texts: list[str], n: int) -> list[set[str]]:
    return [_character_ngrams(_normalize_text(text), n) for text in texts]


def _jaccard(left: set[str], right: set[str]) -> float:
    intersection = len(left & right)
    union_size = len(left) + len(right) - intersection
    if union_size == 0:
        return 0.0
    return intersection / union_size


def exact_overlap(blind_texts: list[str], training_texts: list[str]) -> set[str]:
    """Return the normalized blind texts that also appear in the training texts.

    Both sides are compared after ``.strip().casefold()``, mirroring the inline
    screen in ``run_commercial_blind_hwu64.py``: the result is the intersection
    of the two normalized sets, so whitespace or casing cannot hide a duplicate.
    """
    exposed = {_normalize_text(text) for text in training_texts}
    return {
        normalized
        for normalized in (_normalize_text(text) for text in blind_texts)
        if normalized in exposed
    }


def ngram_overlap(
    blind_texts: list[str],
    training_texts: list[str],
    n: int = 5,
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Flag blind/training pairs whose character n-gram Jaccard similarity reaches ``threshold``.

    Similarity is the Jaccard index of the ``n``-character gram sets taken from
    the normalized texts; a pair at exactly the threshold is flagged. Texts
    shorter than ``n`` characters contribute no grams and never match.
    """
    if n < 1:
        raise ValueError("n-gram size must be positive")
    _validate_threshold(threshold)
    blind_grams = _normalized_ngram_sets(blind_texts, n)
    training_grams = _normalized_ngram_sets(training_texts, n)
    flagged: list[dict[str, Any]] = []
    for blind_index, blind_set in enumerate(blind_grams):
        for training_index, training_set in enumerate(training_grams):
            similarity = _jaccard(blind_set, training_set)
            if similarity >= threshold:
                flagged.append(
                    {
                        "blind_index": blind_index,
                        "training_index": training_index,
                        "similarity": similarity,
                    }
                )
    return flagged


def fuzzy_match(
    blind_texts: list[str],
    training_texts: list[str],
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Flag blind/training pairs whose ``difflib.SequenceMatcher`` ratio reaches ``threshold``.

    Ratios are computed on the normalized texts; a pair at exactly the
    threshold is flagged.
    """
    _validate_threshold(threshold)
    blind_normalized = [_normalize_text(text) for text in blind_texts]
    training_normalized = [_normalize_text(text) for text in training_texts]
    flagged: list[dict[str, Any]] = []
    for blind_index, blind_text in enumerate(blind_normalized):
        for training_index, training_text in enumerate(training_normalized):
            ratio = difflib.SequenceMatcher(None, blind_text, training_text).ratio()
            if ratio >= threshold:
                flagged.append(
                    {
                        "blind_index": blind_index,
                        "training_index": training_index,
                        "ratio": ratio,
                    }
                )
    return flagged


def semantic_neighbors(
    blind_vectors: np.ndarray,
    training_vectors: np.ndarray,
    threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """Flag blind/training embedding pairs whose cosine similarity reaches ``threshold``.

    Rows are L2-normalized (zero rows stay zero) and compared in one matrix
    multiply, mirroring the base-embedding screen in ``phase11_protocol.py``.
    """
    _validate_threshold(threshold)
    blind = np.asarray(blind_vectors, dtype=np.float32)
    training = np.asarray(training_vectors, dtype=np.float32)
    if blind.ndim != 2 or training.ndim != 2:
        raise ValueError("embedding vectors must be two-dimensional")
    if blind.shape[1] != training.shape[1]:
        raise ValueError("blind and training embedding dimensions do not match")
    blind_normalized = blind / np.maximum(
        np.linalg.norm(blind, axis=1, keepdims=True), 1e-12
    )
    training_normalized = training / np.maximum(
        np.linalg.norm(training, axis=1, keepdims=True), 1e-12
    )
    similarities = blind_normalized @ training_normalized.T
    flagged: list[dict[str, Any]] = []
    for blind_index, training_index in np.argwhere(similarities >= threshold):
        flagged.append(
            {
                "blind_index": int(blind_index),
                "training_index": int(training_index),
                "cosine": float(similarities[blind_index, training_index]),
            }
        )
    return flagged


def contamination_report(
    blind_texts: list[str],
    training_texts: list[str],
    blind_vectors: np.ndarray | None = None,
    training_vectors: np.ndarray | None = None,
    ngram_n: int = 5,
    ngram_threshold: float = 0.8,
    fuzzy_threshold: float = 0.9,
    semantic_threshold: float = 0.92,
) -> dict[str, Any]:
    """Run every applicable contamination screen and summarize the flagged evidence.

    ``total_flagged`` is the sorted union of blind indices flagged by any
    check, so ``veya_controlled_overlap`` states how many blind cases need a
    manual look even when several screens agree on the same row. The
    ``base_model_pretraining`` field records the standing attestation: these
    screens compare only Veya-controlled partitions and cannot observe
    pretraining corpora. Provide both embedding matrices to enable the
    semantic screen; the check is ``None`` when neither is supplied.
    """
    if (blind_vectors is None) != (training_vectors is None):
        raise ValueError("semantic screening requires both blind and training vectors")
    exact = exact_overlap(blind_texts, training_texts)
    exact_indices = [
        index
        for index, text in enumerate(blind_texts)
        if _normalize_text(text) in exact
    ]
    ngram_flagged = ngram_overlap(
        blind_texts, training_texts, n=ngram_n, threshold=ngram_threshold
    )
    fuzzy_flagged = fuzzy_match(blind_texts, training_texts, threshold=fuzzy_threshold)
    semantic_check: dict[str, Any] | None = None
    if blind_vectors is not None and training_vectors is not None:
        blind = np.asarray(blind_vectors, dtype=np.float32)
        training = np.asarray(training_vectors, dtype=np.float32)
        if blind.ndim != 2 or blind.shape[0] != len(blind_texts):
            raise ValueError(
                "blind vectors must be two-dimensional and align with blind texts"
            )
        if training.ndim != 2 or training.shape[0] != len(training_texts):
            raise ValueError(
                "training vectors must be two-dimensional and align with training texts"
            )
        semantic_flagged = semantic_neighbors(
            blind, training, threshold=semantic_threshold
        )
        semantic_check = {
            "count": len(semantic_flagged),
            "threshold": semantic_threshold,
            "flagged": semantic_flagged,
        }
    flagged_blind = set(exact_indices)
    for pair in ngram_flagged + fuzzy_flagged:
        flagged_blind.add(int(pair["blind_index"]))
    if semantic_check is not None:
        for pair in semantic_check["flagged"]:
            flagged_blind.add(int(pair["blind_index"]))
    total_flagged = sorted(flagged_blind)
    controlled_overlap = (
        "verified disjoint"
        if not total_flagged
        else f"{len(total_flagged)} cases flagged; see checks"
    )
    return {
        "format": REPORT_FORMAT,
        "blind_n": len(blind_texts),
        "training_n": len(training_texts),
        "checks": {
            "exact": {"count": len(exact_indices), "indices": exact_indices},
            "ngram": {
                "count": len(ngram_flagged),
                "threshold": ngram_threshold,
                "flagged": ngram_flagged,
            },
            "fuzzy": {
                "count": len(fuzzy_flagged),
                "threshold": fuzzy_threshold,
                "flagged": fuzzy_flagged,
            },
            "semantic": semantic_check,
        },
        "total_flagged": total_flagged,
        "veya_controlled_overlap": controlled_overlap,
        "base_model_pretraining": "no detected evidence of base-model pretraining exposure",
    }
