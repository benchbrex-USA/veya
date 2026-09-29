"""BART-MNLI zero-shot rival. Research-only; primary runnable rival."""
from __future__ import annotations

import importlib.util

from .base import Rival, render_state

HYPOTHESIS = {
    "choice": "This text is about {}.",
    "noul": "{}",
    "score": "{}",
}


class NliRival(Rival):
    name = "nli-bart"
    MODEL_ID = "facebook/bart-large-mnli"

    def __init__(self, device=None):
        self._device = device          # None: transformers picks; -1 cpu; 0 first gpu
        self._pipe = None

    def available(self) -> bool:
        return importlib.util.find_spec("transformers") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "transformers not installed"

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("zero-shot-classification", model=self.MODEL_ID,
                                  device=self._device)
        return self._pipe

    def _options_probs(self, state, qdef, options):
        pipe = self._ensure()
        premise = render_state(state) + "\n\n" + str(qdef.get("instructions", ""))
        template = HYPOTHESIS.get(qdef["type"], "{}")
        res = pipe(premise, candidate_labels=list(options),
                   hypothesis_template=template, multi_label=False)
        # res["labels"] is sorted by score; re-index to `options` order (correctness-critical).
        by_label = dict(zip(res["labels"], res["scores"]))
        return [float(by_label.get(o, 0.0)) for o in options]
