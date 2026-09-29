"""Dormant Jev / TypeSafe System One rival. Research-only.

Written against the documented sync API (docs.typesafe.ai/sdk/python):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    with TypeSafeClient() as client:
        resp = client.system_one(state={...}, questions={"q": Choice|Noul|Score(...)})
    resp.answers["q"].probabilities  # choice/score: dict keyed by option/level
    resp.answers["q"].noul           # noul: float P(yes)

Activates ONLY when TYPESAFE_API_KEY is set AND typesafe_sdk imports. Otherwise it reports
itself unavailable and the driver records it as "not measured" - never a fabricated number.
"""
from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

from .base import Rival, option_labels


def _state_dict(state):
    """TypeSafe state must be a dict of named fields."""
    if isinstance(state, dict):
        return {str(k): (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v))
                for k, v in state.items()}
    return {"text": str(state)}


class JevRival(Rival):
    name = "jev-typesafe"

    def __init__(self, max_cases=None, sleep_s=0.0):
        self.max_cases = int(os.environ.get("JEV_MAX_CASES", max_cases or 200))
        self.sleep_s = float(sleep_s)
        self._ctx = None
        self._client = None
        self.errors = 0

    def available(self) -> bool:
        return bool(os.environ.get("TYPESAFE_API_KEY")) and \
            importlib.util.find_spec("typesafe_sdk") is not None

    def unavailable_reason(self) -> str:
        if not os.environ.get("TYPESAFE_API_KEY"):
            return "TYPESAFE_API_KEY not set - Jev not measured here"
        if importlib.util.find_spec("typesafe_sdk") is None:
            return "typesafe_sdk not installed - Jev not measured here"
        return ""

    def _ensure(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            self._ctx = TypeSafeClient()
            self._client = self._ctx.__enter__()
        return self._client

    def score(self, cases, gold):
        cases = list(cases)[: self.max_cases]
        gold = list(gold)[: self.max_cases]
        return super().score(cases, gold)

    def _options_probs(self, state, qdef, options):
        from typesafe_sdk import Choice, Noul, Score
        client = self._ensure()
        t = qdef["type"]
        ins = str(qdef.get("instructions", ""))
        labels = option_labels(qdef)
        st = _state_dict(state)
        try:
            if t == "noul":
                resp = client.system_one(state=st, questions={"q": Noul(instructions=ins)})
                p = float(resp.answers["q"].noul)
                out = np.array([1.0 - p, p])
            elif t == "score":
                resp = client.system_one(
                    state=st, questions={"q": Score(instructions=ins, criteria=list(labels))})
                probs = getattr(resp.answers["q"], "probabilities", {}) or {}
                out = np.array([float(probs.get(lbl, 0.0)) for lbl in labels])
            else:
                crit = qdef.get("criteria") or {}
                criteria = {k: (v if v else None) for k, v in crit.items()} \
                    if isinstance(crit, dict) else {str(c): None for c in crit}
                resp = client.system_one(
                    state=st, questions={"q": Choice(instructions=ins, criteria=criteria)})
                probs = resp.answers["q"].probabilities or {}
                out = np.array([float(probs.get(lbl, 0.0)) for lbl in labels])
        finally:
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return out

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            finally:
                self._ctx = None
                self._client = None
