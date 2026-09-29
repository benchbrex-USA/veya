"""Shared adapter interface + probability mapping for the rival head-to-head benchmark.

Research-only: nothing here is imported by the `veya` package. Every rival reduces its raw
output to the same rows = [(gold_idx, probs_vector|None), ...] that bench_local.metrics()
scores, so Veya and rivals are compared by construction on byte-identical cases.

Gold-index order (the single source of truth, matches bench_apps / build_typed_decisions):
  choice -> list(criteria.keys())   noul -> [false, true]   score -> level list
"""
from __future__ import annotations

import time

import numpy as np

STATE_CHAR_BUDGET = 4000


def render_state(state) -> str:
    """Flatten a case state (dict of named text fields, or a str) into readable text."""
    if isinstance(state, str):
        return state[:STATE_CHAR_BUDGET]
    if not isinstance(state, dict):
        return str(state)[:STATE_CHAR_BUDGET]
    text = "\n".join("%s: %s" % (k, v) for k, v in state.items())
    return text[:STATE_CHAR_BUDGET]


def first_question(questions):
    """bench_apps suites carry exactly one question per case; return it (or None)."""
    for _qid, qdef in (questions or {}).items():
        return qdef
    return None


def option_labels(qdef) -> list:
    """Bare labels/levels in gold-index order."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            return [str(c) for c in crit]
        return [str(k) for k in (crit or {}).keys()]
    if t == "noul":
        return ["false", "true"]
    if t == "score":
        if isinstance(crit, list):
            return [str(c) for c in crit]
        return [str(k) for k in (crit or {}).keys()]
    return []


def option_texts(qdef) -> list:
    """Descriptive option strings (for NLI candidates / ST embeddings), gold-index order."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            return [str(c) for c in crit]
        return [("%s - %s" % (k, v) if v else str(k)) for k, v in (crit or {}).items()]
    if t == "noul":
        if isinstance(crit, dict) and ("true" in crit or "false" in crit):
            return [str(crit.get("false", "false")), str(crit.get("true", "true"))]
        return ["false", "true"]
    if t == "score":
        if isinstance(crit, list):
            return [str(c) for c in crit]
        return [("%s - %s" % (k, v) if v else str(k)) for k, v in (crit or {}).items()]
    return []


def probs_from_scores(scores) -> np.ndarray:
    """Numerically stable softmax over raw scores."""
    z = np.asarray(scores, dtype=float)
    if z.size == 0:
        return z
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def to_rows(per_question, gold) -> list:
    """Zip per-case probability vectors (or None) with gold indices, in case order."""
    return [(gi, pv) for gi, pv in zip(gold, per_question)]


class Rival:
    """Adapter base. Subclasses implement `_options_probs` and, optionally, availability."""

    name = "rival"

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def _options_probs(self, state, qdef, options):
        """Return a score/probability per entry of `options` (gold-index order)."""
        raise NotImplementedError

    def score(self, cases, gold):
        """Return (rows, seconds); rows align to gold, bad cases -> None (dropped by metrics)."""
        t0 = time.time()
        per_q = []
        for state, questions in cases:
            qdef = first_question(questions)
            try:
                options = option_texts(qdef) if qdef else []
                labels = option_labels(qdef) if qdef else []
                if len(options) < 2 or len(options) != len(labels):
                    per_q.append(None)
                    continue
                p = np.asarray(self._options_probs(state, qdef, options), dtype=float)
                if p.shape != (len(options),) or p.sum() <= 0 or not np.isfinite(p).all():
                    per_q.append(None)
                    continue
                per_q.append(p / p.sum())
            except Exception:
                per_q.append(None)
        return to_rows(per_q, gold), time.time() - t0
