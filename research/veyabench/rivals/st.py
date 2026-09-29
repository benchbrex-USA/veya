"""Sentence-transformer nearest-centroid rival. Research-only; light fallback (no 1.6 GB download)."""
from __future__ import annotations

import importlib.util

import numpy as np

from .base import Rival, probs_from_scores, render_state


class StCentroidRival(Rival):
    name = "st-centroid"
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    TEMP = 0.1

    def __init__(self, device=None):
        self._device = device
        self._model = None

    def available(self) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "sentence-transformers not installed"

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_ID, device=self._device)
        return self._model

    def _options_probs(self, state, qdef, options):
        model = self._ensure()
        premise = render_state(state) + "\n\n" + str(qdef.get("instructions", ""))
        emb = model.encode([premise] + list(options), convert_to_numpy=True,
                           normalize_embeddings=True)
        sims = emb[1:] @ emb[0]                     # cosine (embeddings normalized)
        return probs_from_scores(np.asarray(sims, float) / self.TEMP)
