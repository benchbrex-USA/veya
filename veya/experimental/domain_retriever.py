"""Experimental Phase-10 domain-general candidate retrieval.

This module is deliberately outside the audited release path.  It reuses the
already-loaded Veya encoder, then applies small query/candidate projections and
exact cosine ranking.  Candidate strings remain untrusted data passed only to
the tokenizer; they never become instructions.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from veya.common import serialize_state
from veya.shortlist_artifact import file_sha256


FORMAT = "veya-domain-retriever-v1"
MODE_CURRENT = "current"
MODE_PHASE10 = "phase10_domain_general"
MODE_PHASE10_MULTIVIEW = "phase10_domain_general_multiview"
MODE_PHASE11 = "phase11_noise_robust"
QUERY_FUSIONS = frozenset({"first", "max", "mean", "rrf"})
PHASE11_VIEW_POLICIES = frozenset(
    {"single", "first_last", "first_middle_last", "four_view", "adaptive"}
)


def build_query_views(
    state: Any,
    *,
    max_views: int = 4,
    max_chars: int = 4096,
) -> list[str]:
    """Build bounded semantic views without interpreting state as instructions.

    A single-line string remains byte-for-byte equivalent after surrounding
    whitespace is removed. Multiline text exposes individual lines before the
    full serialized state, allowing an experimental ranker to avoid diluting a
    useful leading segment with trailing operational metadata. Structured
    states expose bounded scalar leaf values plus their complete serialization.
    """
    if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views < 1:
        raise ValueError("max_views must be a positive integer")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 32:
        raise ValueError("max_chars must be an integer of at least 32")

    full = serialize_state(state).strip()
    if not full:
        raise ValueError("state must not be empty")

    raw_views: list[str] = []
    if isinstance(state, str):
        raw_views.extend(line.strip() for line in state.splitlines() if line.strip())
    elif isinstance(state, dict):
        preferred = ("text", "message", "body", "description", "subject", "title", "query")
        ordered_keys = sorted(
            state,
            key=lambda key: (
                preferred.index(str(key).lower())
                if str(key).lower() in preferred
                else len(preferred),
                str(key),
            ),
        )
        for key in ordered_keys:
            value = state[key]
            if isinstance(value, str) and value.strip():
                raw_views.append(f"{key}: {value.strip()}")
            elif isinstance(value, (int, float, bool)) or value is None:
                raw_views.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    elif isinstance(state, list):
        for index, value in enumerate(state):
            if isinstance(value, str) and value.strip():
                raw_views.append(f"item {index}: {value.strip()}")

    if raw_views:
        raw_views = raw_views[: max(0, max_views - 1)]
    raw_views.append(full)
    views: list[str] = []
    seen: set[str] = set()
    for value in raw_views:
        bounded = value.strip()[:max_chars]
        if bounded and bounded not in seen:
            seen.add(bounded)
            views.append(bounded)
        if len(views) == max_views:
            break
    return views


def _generic_semantic_fields(state: Any, prefix: str = "") -> list[str]:
    metadata_names = {
        "id",
        "uuid",
        "timestamp",
        "created_at",
        "updated_at",
        "request_id",
        "trace_id",
        "hash",
        "version",
        "attempt",
        "display_theme",
    }
    values: list[str] = []
    if isinstance(state, dict):
        for key in sorted(state, key=str):
            normalized = str(key).strip().casefold()
            if normalized in metadata_names or normalized.endswith("_id"):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            values.extend(_generic_semantic_fields(state[key], path))
    elif isinstance(state, list):
        for index, value in enumerate(state):
            path = f"{prefix}[{index}]" if prefix else f"item[{index}]"
            values.extend(_generic_semantic_fields(value, path))
    elif isinstance(state, str) and state.strip():
        values.append(f"{prefix}: {state.strip()}" if prefix else state.strip())
    elif isinstance(state, (int, float, bool)) and prefix:
        values.append(f"{prefix}: {state}")
    return values


def build_phase11_query_views(
    state: Any,
    *,
    policy: str = "adaptive",
    max_views: int = 4,
    max_chars: int = 4096,
    window_chars: int = 512,
    max_state_chars: int = 32768,
) -> list[str]:
    """Build label-blind bounded windows for Phase-11 retrieval."""
    if policy not in PHASE11_VIEW_POLICIES:
        raise ValueError("unknown Phase-11 query-view policy %r" % policy)
    for name, numeric_value, minimum in (
        ("max_views", max_views, 1),
        ("max_chars", max_chars, 32),
        ("window_chars", window_chars, 32),
        ("max_state_chars", max_state_chars, 32),
    ):
        if (
            isinstance(numeric_value, bool)
            or not isinstance(numeric_value, int)
            or numeric_value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if max_views > 4:
        raise ValueError("Phase-11 query views are capped at 4")
    full = serialize_state(state).strip()
    if not full:
        raise ValueError("state must not be empty")
    if len(full) > max_state_chars:
        raise ValueError("state exceeds the Phase-11 character bound")

    head = full[:window_chars]
    tail = full[-window_chars:]
    middle_start = max(0, (len(full) - window_chars) // 2)
    middle = full[middle_start : middle_start + window_chars]
    span = max(0, len(full) - window_chars)
    one_third = full[span // 3 : span // 3 + window_chars]
    two_thirds = full[(2 * span) // 3 : (2 * span) // 3 + window_chars]
    structured = _generic_semantic_fields(state) if isinstance(state, (dict, list)) else []

    if policy == "single":
        raw = [full]
    elif policy == "first_last":
        raw = [head, tail]
    elif policy == "first_middle_last":
        raw = [head, middle, tail]
    elif policy == "four_view":
        raw = [head, middle, tail, full]
    elif structured:
        raw = structured[:max_views]
    elif len(full) <= window_chars:
        raw = [full]
    elif len(full) <= window_chars * 2:
        raw = [head, tail]
    elif len(full) <= window_chars * 3:
        raw = [head, middle, tail]
    else:
        raw = [head, one_third, two_thirds, tail]

    views: list[str] = []
    seen: set[str] = set()
    for raw_value in raw:
        bounded = str(raw_value).strip()[:max_chars]
        if bounded and bounded not in seen:
            seen.add(bounded)
            views.append(bounded)
        if len(views) == max_views:
            break
    if not views:
        raise ValueError("query-view policy produced no usable state")
    return views


def fuse_query_scores(view_scores: np.ndarray, fusion: str) -> np.ndarray:
    """Fuse a ``views x candidates`` score matrix deterministically."""
    scores = np.asarray(view_scores, dtype=np.float32)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("view_scores must be a non-empty matrix")
    if not np.isfinite(scores).all():
        raise ValueError("view_scores must be finite")
    if fusion not in QUERY_FUSIONS:
        raise ValueError("unknown query fusion %r" % fusion)
    if fusion == "first":
        return scores[0]
    if fusion == "max":
        return scores.max(axis=0)
    if fusion == "mean":
        return scores.mean(axis=0)
    ranks = np.empty_like(scores, dtype=np.int64)
    for row_index, row in enumerate(scores):
        order = np.argsort(-row, kind="mergesort")
        ranks[row_index, order] = np.arange(1, len(order) + 1)
    return np.sum(1.0 / (60.0 + ranks), axis=0)


def naturalize_label(value: str) -> str:
    """Remove machine-label formatting without inventing any semantics."""
    if not isinstance(value, str):
        raise TypeError("candidate labels must be strings")
    text = unicodedata.normalize("NFKC", value).strip()
    out: list[str] = []
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if char in "_-./":
            out.append(" ")
        elif char.isupper() and previous and previous.islower():
            out.extend((" ", char.lower()))
        elif char.isupper() and previous.isupper() and following.islower():
            out.extend((" ", char.lower()))
        else:
            out.append(char.lower())
    return " ".join("".join(out).split())


def text_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class ProjectionPair(nn.Module):
    """Shared-backbone query/candidate projections for cosine retrieval."""

    def __init__(self, input_dim: int, output_dim: int, projection: str = "dual") -> None:
        super().__init__()
        if projection not in ("shared", "dual"):
            raise ValueError("projection must be 'shared' or 'dual'")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = projection
        self.query = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.candidate = self.query if projection == "shared" else nn.Linear(
            self.input_dim, self.output_dim, bias=False
        )

    @staticmethod
    def _unit(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(value, p=2, dim=-1, eps=1e-12)

    def encode_query(self, value: torch.Tensor) -> torch.Tensor:
        return self._unit(self.query(value))

    def encode_candidate(self, value: torch.Tensor) -> torch.Tensor:
        return self._unit(self.candidate(value))

    def scores(self, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.encode_query(queries) @ self.encode_candidate(candidates).T


class DomainRetriever:
    """Loaded retriever with deterministic exact ranking and versioned caching."""

    def __init__(
        self,
        projection: ProjectionPair,
        metadata: dict,
        embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> None:
        self.projection = projection.cpu().eval()
        self.metadata = dict(metadata)
        self.retriever_id = str(metadata["retriever_id"])
        self.embed_fn = embed_fn
        self._candidate_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def set_embed_fn(self, embed_fn: Callable[[Sequence[str]], np.ndarray]) -> None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        self.embed_fn = embed_fn

    @staticmethod
    def _base_vectors(raw: object, rows: int) -> np.ndarray:
        if hasattr(raw, "detach"):
            raw = raw.detach().float().cpu().numpy()
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != rows:
            raise ValueError("embedding function returned shape %r for %d rows" % (array.shape, rows))
        if not np.isfinite(array).all():
            raise ValueError("embedding function returned non-finite values")
        return array

    def project_queries(self, values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            return self.projection.encode_query(tensor).cpu().numpy()

    def project_candidates(self, values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            return self.projection.encode_candidate(tensor).cpu().numpy()

    def candidate_vectors(
        self,
        candidates: Sequence[tuple[str, str]],
        schema_version: str,
    ) -> np.ndarray:
        if self.embed_fn is None:
            raise RuntimeError("retriever has no embedding function")
        labels = [label for label, _text in candidates]
        texts = [text for _label, text in candidates]
        content_hash = text_sha256([part for pair in candidates for part in pair])
        key = (self.retriever_id, str(schema_version), content_hash)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if len(set(labels)) != len(labels):
            raise ValueError("candidate labels must be unique")
        base = self._base_vectors(self.embed_fn(texts), len(texts))
        projected = self.project_candidates(base)
        self._candidate_cache[key] = projected
        self.cache_misses += 1
        return projected

    def rank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        schema_version: str,
        k: int = 8,
    ) -> tuple[list[str], np.ndarray]:
        if self.embed_fn is None:
            raise RuntimeError("retriever has no embedding function")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not candidates:
            raise ValueError("candidates must not be empty")
        query_base = self._base_vectors(self.embed_fn([str(query)]), 1)
        query_vector = self.project_queries(query_base)[0]
        candidate_vectors = self.candidate_vectors(candidates, schema_version)
        scores = candidate_vectors @ query_vector
        order = np.argsort(-scores, kind="mergesort")[: min(k, len(candidates))]
        return [candidates[int(index)][0] for index in order], scores[order]

    def rank_views(
        self,
        views: Sequence[str],
        candidates: Sequence[tuple[str, str]],
        schema_version: str,
        k: int = 8,
        fusion: str = "rrf",
    ) -> tuple[list[str], np.ndarray]:
        """Rank from bounded query views using deterministic score fusion."""
        if self.embed_fn is None:
            raise RuntimeError("retriever has no embedding function")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not candidates:
            raise ValueError("candidates must not be empty")
        if fusion not in QUERY_FUSIONS:
            raise ValueError("unknown query fusion %r" % fusion)
        normalized = [str(view).strip() for view in views]
        if not normalized or any(not view for view in normalized):
            raise ValueError("query views must not be empty")
        if len(normalized) > 8 or any(len(view) > 8192 for view in normalized):
            raise ValueError("query views exceed runtime bounds")

        query_base = self._base_vectors(self.embed_fn(normalized), len(normalized))
        query_vectors = self.project_queries(query_base)
        candidate_vectors = self.candidate_vectors(candidates, schema_version)
        view_scores = query_vectors @ candidate_vectors.T
        scores = fuse_query_scores(view_scores, fusion)
        order = np.argsort(-scores, kind="mergesort")[: min(k, len(candidates))]
        return [candidates[int(index)][0] for index in order], scores[order]


def save_domain_retriever(path: str, projection: ProjectionPair, metadata: dict) -> dict:
    required = {
        "retriever_id",
        "backbone",
        "checkpoint_hash",
        "training_datasets",
        "held_out_schemas",
        "candidate_representation",
        "query_representation",
        "objective",
        "k",
        "git_commit",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError("retriever metadata is missing %s" % ", ".join(missing))
    blob = {
        "format": FORMAT,
        **metadata,
        "architecture": "frozen_shared_backbone_biencoder",
        "projection": projection.projection,
        "input_dim": projection.input_dim,
        "output_dim": projection.output_dim,
        "similarity": "cosine",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state_dict": projection.state_dict(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(blob, path)
    sidecar = {key: value for key, value in blob.items() if key != "state_dict"}
    sidecar["checkpoint_hash"] = str(sidecar["checkpoint_hash"])
    sidecar["file_sha256"] = file_sha256(path)
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sidecar


def load_domain_retriever(
    path: str,
    checkpoint_hash: str,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> DomainRetriever:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT:
        raise ValueError("unsupported retriever format %r" % blob.get("format"))
    if blob.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("retriever belongs to a different base checkpoint")
    if int(blob.get("k", -1)) != 8:
        raise ValueError("Phase-10 retriever must use K=8")
    projection = ProjectionPair(
        int(blob["input_dim"]), int(blob["output_dim"]), str(blob["projection"])
    )
    projection.load_state_dict(blob["state_dict"])
    metadata = {key: value for key, value in blob.items() if key != "state_dict"}
    return DomainRetriever(projection, metadata, embed_fn=embed_fn)


def configure_experimental_retriever(
    bundle: Any,
    config: dict,
    artifact_path: str | None = None,
) -> Any:
    """Select the audited retriever or opt one bundle instance into Phase 10."""
    mode = config.get("retriever", MODE_CURRENT)
    if mode == MODE_CURRENT:
        return bundle
    if mode not in (MODE_PHASE10, MODE_PHASE10_MULTIVIEW, MODE_PHASE11):
        raise ValueError("unknown retriever mode %r" % mode)
    if not artifact_path:
        raise ValueError("phase10_domain_general requires an artifact path")
    from veya.shortlist import embed_fn_from_agent

    embed_fn = embed_fn_from_agent(
        bundle.agent,
        max_length=int(bundle.router["embed_max_length"]),
        batch_size=16,
    )
    retriever = load_domain_retriever(
        artifact_path,
        checkpoint_hash=bundle.checkpoint_hash,
        embed_fn=embed_fn,
    )
    schema_version = str(config.get("candidate_schema_version", "runtime-v1"))

    def shortlist(
        _bundle: Any,
        state: Any,
        criteria_items: Sequence[tuple[str, object]],
        k: int,
    ):
        candidates = []
        for label, value in criteria_items:
            if retriever.metadata.get("candidate_representation") == "naturalized_label":
                text = naturalize_label(label)
            else:
                text = value if isinstance(value, str) and value.strip() else naturalize_label(label)
            candidates.append((label, str(text)))
        if mode == MODE_PHASE10_MULTIVIEW:
            max_views = int(config.get("query_max_views", 4))
            max_chars = int(config.get("query_view_max_chars", 4096))
            fusion = str(config.get("query_fusion", "rrf"))
            views = build_query_views(
                state,
                max_views=max_views,
                max_chars=max_chars,
            )
            labels, _scores = retriever.rank_views(
                views,
                candidates,
                schema_version=schema_version,
                k=k,
                fusion=fusion,
            )
        elif mode == MODE_PHASE11:
            max_views = int(config.get("query_max_views", 4))
            max_chars = int(config.get("query_view_max_chars", 4096))
            window_chars = int(config.get("query_window_chars", 512))
            max_state_chars = int(config.get("query_max_state_chars", 32768))
            fusion = str(config.get("query_fusion", "max"))
            policy = str(config.get("query_view_policy", "adaptive"))
            views = build_phase11_query_views(
                state,
                policy=policy,
                max_views=max_views,
                max_chars=max_chars,
                window_chars=window_chars,
                max_state_chars=max_state_chars,
            )
            labels, _scores = retriever.rank_views(
                views,
                candidates,
                schema_version=schema_version,
                k=k,
                fusion=fusion,
            )
        else:
            views = [serialize_state(state)]
            fusion = "single"
            labels, _scores = retriever.rank(
                views[0], candidates, schema_version=schema_version, k=k
            )
        return labels, {
            "state_embed": None,
            "candidate_embed": None,
            "shortlist_score": None,
            "experimental_retriever": retriever.retriever_id,
            "experimental_query_views": len(views),
            "experimental_query_fusion": fusion,
        }

    bundle.shortlist = types.MethodType(shortlist, bundle)
    bundle.experimental_retriever = retriever
    bundle.experimental_retriever_mode = mode
    return bundle
