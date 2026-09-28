"""Auditable Phase-11 truncation and hard-negative protocol helpers."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from veya.experimental.domain_retriever import naturalize_label


HARD_NEGATIVE_VERSION = "phase11-second-refresh-v1"
LENGTH_BUCKETS = ((0, 32), (33, 64), (65, 128), (129, 256), (257, None))


def tokenization_audit(
    tokenizer: Any,
    source: str,
    augmented: str,
    *,
    max_length: int,
    view_count: int,
) -> dict[str, Any]:
    """Report actual tokenizer retention and the useful source span position."""
    if max_length < 4 or view_count < 1:
        raise ValueError("invalid audit bounds")
    source = str(source).strip()
    augmented = str(augmented).strip()
    start = augmented.find(source)
    if start < 0:
        escaped_source = json.dumps(source, ensure_ascii=False)[1:-1]
        start = augmented.find(escaped_source)
        if start >= 0:
            source = escaped_source
    if not source or start < 0:
        raise ValueError("augmented query must contain the clean source")
    encoded = tokenizer(
        augmented,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    source_end = start + len(source)
    positive_indices: list[int] = []
    if offsets is not None:
        for index, pair in enumerate(offsets):
            left, right = int(pair[0]), int(pair[1])
            if right > start and left < source_end:
                positive_indices.append(index)
    if not positive_indices:
        prefix = tokenizer(augmented[:start], add_special_tokens=False)["input_ids"]
        signal = tokenizer(source, add_special_tokens=False)["input_ids"]
        positive_indices = list(range(1 + len(prefix), 1 + len(prefix) + len(signal)))
    token_length = len(input_ids)
    positive_start = min(positive_indices)
    positive_end = max(positive_indices)
    retained_positive = sum(index < max_length for index in positive_indices)
    if positive_start < max_length and positive_end < max_length:
        position = "fully_retained"
    elif positive_start < max_length:
        position = "partially_retained"
    else:
        position = "truncated"
    return {
        "raw_character_length": len(augmented),
        "token_length": token_length,
        "truncated": token_length > max_length,
        "tokens_retained": min(token_length, max_length),
        "tokens_dropped": max(0, token_length - max_length),
        "positive_signal_start_token": positive_start,
        "positive_signal_end_token": positive_end,
        "positive_signal_tokens": len(positive_indices),
        "positive_signal_tokens_retained": retained_positive,
        "positive_signal_position": position,
        "view_count": view_count,
    }


def mine_hard_negatives_with_lineage(
    scores: np.ndarray,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_base_vectors: np.ndarray,
    *,
    retriever_id: str,
    limit: int = 8,
    semantic_similarity_threshold: float = 0.92,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Mine one TRAIN-only refresh with explicit false-negative screening."""
    matrix = np.asarray(scores, dtype=np.float32)
    vectors = np.asarray(candidate_base_vectors, dtype=np.float32)
    if matrix.shape != (len(rows), len(candidates)):
        raise ValueError("score matrix shape does not match rows and candidates")
    if vectors.ndim != 2 or vectors.shape[0] != len(candidates):
        raise ValueError("candidate vectors do not match candidates")
    if limit < 1:
        raise ValueError("hard-negative limit must be positive")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.maximum(norms, 1e-12)
    candidate_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    normalized_labels = [naturalize_label(str(row["raw"])) for row in candidates]
    order = np.argsort(-matrix, axis=1, kind="stable")
    selected: dict[str, list[int]] = {}
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        truth = candidate_index[str(row["candidate_id"])]
        chosen: list[int] = []
        for rank, raw_index in enumerate(order[row_index], start=1):
            index = int(raw_index)
            if index == truth:
                rejected["gold"] += 1
                continue
            if normalized_labels[index] == normalized_labels[truth]:
                rejected["normalized_duplicate"] += 1
                continue
            semantic_similarity = float(normalized_vectors[index] @ normalized_vectors[truth])
            if semantic_similarity >= semantic_similarity_threshold:
                rejected["base_embedding_near_duplicate"] += 1
                continue
            chosen.append(index)
            records.append(
                {
                    "row_id": str(row["id"]),
                    "negative_candidate_id": str(candidates[index]["candidate_id"]),
                    "negative_source": "allowed_train_pool",
                    "retriever_version": retriever_id,
                    "rank": rank,
                    "similarity": float(matrix[row_index, index]),
                    "truth_candidate_id": str(row["candidate_id"]),
                    "truth_schema": str(row["schema"]),
                    "negative_schema": str(candidates[index]["schema"]),
                    "false_negative_screening": {
                        "normalized_duplicate": False,
                        "base_embedding_similarity": semantic_similarity,
                        "threshold": semantic_similarity_threshold,
                        "accepted": True,
                    },
                }
            )
            if len(chosen) == limit:
                break
        selected[str(row["id"])] = chosen
    lineage = {
        "format": "veya-phase11-hard-negative-lineage-v1",
        "hard_negative_version": HARD_NEGATIVE_VERSION,
        "source_partition": "TRAIN only",
        "retriever_id": retriever_id,
        "rows": len(rows),
        "accepted": len(records),
        "accepted_per_row": limit,
        "semantic_similarity_threshold": semantic_similarity_threshold,
        "rejected": dict(sorted(rejected.items())),
        "records_sha256": _records_digest(records),
        "records": records,
        "manual_audit_sample": records[: min(32, len(records))],
    }
    return selected, lineage


def _records_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        value = "\0".join(
            (
                str(record["row_id"]),
                str(record["negative_candidate_id"]),
                str(record["rank"]),
                format(float(record["similarity"]), ".9g"),
            )
        )
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_hard_negative_lineage(lineage: dict[str, Any]) -> None:
    if lineage.get("hard_negative_version") != HARD_NEGATIVE_VERSION:
        raise ValueError("unexpected hard-negative version")
    if lineage.get("source_partition") != "TRAIN only":
        raise ValueError("hard negatives are not TRAIN-only")
    records = lineage.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("hard-negative lineage has no records")
    if _records_digest(records) != lineage.get("records_sha256"):
        raise ValueError("hard-negative lineage digest mismatch")
    for record in records:
        screen = record.get("false_negative_screening", {})
        if record.get("negative_source") != "allowed_train_pool" or not screen.get("accepted"):
            raise ValueError("hard-negative lineage contains an unaudited record")


def token_length_bucket(token_length: int) -> str:
    if token_length < 0:
        raise ValueError("token length must not be negative")
    for low, high in LENGTH_BUCKETS:
        if high is None or token_length <= high:
            return f"{low}+" if high is None else f"{low}-{high}"
    raise AssertionError("length buckets are incomplete")
