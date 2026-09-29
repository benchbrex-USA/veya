# Veya Rival Head-to-Head Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled, measured head-to-head benchmark that runs rival decision models on byte-identical `bench_apps` cases through the same `metrics()` pipeline as Veya's three checkpoints, replacing the repo's pasted/indicative Jev numbers.

**Architecture:** A `Rival` adapter interface in `research/veyabench/rivals/` reduces every rival's raw output to the same `rows = [(gold_idx, probs_vector|None), ...]` that `bench_local.metrics()` already scores. Two runnable local rivals (BART-MNLI zero-shot, sentence-transformer nearest-centroid) plus a dormant Jev/TypeSafe adapter. A driver reuses `bench_apps.build()` suites and `bench_local`'s engine, runs Veya×3 + rivals, and writes `research/results/rival_benchmark.json` + a combined table. A separate plot script and a `BENCHMARKS.md` section present the result.

**Tech Stack:** Python 3.10+, numpy, `transformers` (zero-shot pipeline), `sentence-transformers`, optional `datasets` (to build suites) and `typesafe_sdk` (dormant Jev). matplotlib for the plot. pytest for tests.

**Spec:** `docs/superpowers/specs/2026-09-29-veya-rival-head-to-head-benchmark-design.md`

## Global Constraints

- **Research-only.** All new code lives under `research/` (plus one test file and one `BENCHMARKS.md` edit). Nothing here is imported by the `veya` package. Never touch any `veya/*.py` file or `artifacts/release/manifest.json` — they are hash-pinned by the release manifest.
- **Reuse, do not fork.** Score Veya through `bench_local.score_cases` / `softmax_t` / `temp_for` / `metrics`. Build suites through `bench_apps.build()` and iterate whatever it registers — do not hardcode the suite list.
- **Case contract.** A case is `(state, questions)`; every `bench_apps` suite has exactly **one** question per case, so `gold[ci]` is a single option index. `criteria` is a dict `{label: description|None}` for choice, an optional `{true, false}` dict for noul, and a list of level strings for score.
- **Gold-index order is the single source of truth.** For choice, order = `list(criteria.keys())`; for noul, order = `[false, true]`; for score, order = the level list. Every adapter MUST emit probabilities in this exact order.
- **Backend env pins.** Entry points set `USE_TF=0`, `USE_TORCH=1`, `TOKENIZERS_PARALLELISM=false` before importing torch/transformers.
- **Fixed seed.** `SEED = 13`, matching `bench_apps`/`bench_local`.
- **Honest degradation.** Missing `datasets`, a missing Veya checkpoint, an unavailable rival, or a per-case error is recorded as a skip/`None` — never a fabricated number and never a silent crash.
- **Tunables (verbatim from spec):** `STATE_CHAR_BUDGET = 4000`, `ST_TEMP = 0.1`, `BENCH_N = 400`, `JEV_MAX_CASES = 200`.

## File Structure

- `research/veyabench/rivals/__init__.py` — package marker; re-exports the rival classes + `REGISTRY`.
- `research/veyabench/rivals/base.py` — `Rival` base class + shared mapping (`render_state`, `option_texts`, `option_labels`, `first_question`, `probs_from_scores`, `to_rows`).
- `research/veyabench/rivals/nli.py` — `NliRival` (BART-MNLI zero-shot).
- `research/veyabench/rivals/st.py` — `StCentroidRival` (MiniLM nearest-centroid).
- `research/veyabench/rivals/jev.py` — `JevRival` (dormant TypeSafe adapter).
- `research/scripts/bench_rival.py` — driver.
- `research/scripts/make_rival_plot.py` — dedicated plot (`assets/veya_vs_rival.png`).
- `tests/test_rival_benchmark.py` — unit tests (no large downloads, no network).
- `BENCHMARKS.md` — new *Measured head-to-head* section (edited last).

---

### Task 1: Shared `Rival` interface + probability mapping (`base.py`)

**Files:**
- Create: `research/veyabench/rivals/__init__.py`
- Create: `research/veyabench/rivals/base.py`
- Test: `tests/test_rival_benchmark.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `render_state(state) -> str`
  - `option_texts(qdef) -> list[str]` (descriptive strings for embedding/NLI candidates)
  - `option_labels(qdef) -> list[str]` (bare labels/levels, same order as `option_texts`)
  - `first_question(questions) -> dict | None`
  - `probs_from_scores(scores) -> np.ndarray` (stable softmax)
  - `to_rows(per_question, gold) -> list[tuple[int, np.ndarray | None]]`
  - `class Rival` with `name: str`, `available() -> bool`, `unavailable_reason() -> str`, `score(cases, gold) -> (rows, seconds)`, and overridable `_options_probs(state, qdef, options) -> sequence[float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rival_benchmark.py`:

```python
"""Unit tests for the rival head-to-head benchmark. No large model downloads, no network."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "research"))
sys.path.insert(0, os.path.join(REPO, "research", "scripts"))

from veyabench.rivals.base import (  # noqa: E402
    Rival, first_question, option_labels, option_texts, probs_from_scores,
    render_state, to_rows,
)


CHOICE_Q = {"type": "choice", "instructions": "Pick the topic of `article`.",
            "criteria": {"world": "international news", "sports": None, "business": "economy"}}
NOUL_Q = {"type": "noul", "instructions": "Is `email` spam?",
          "criteria": {"true": "unsolicited bulk", "false": "legitimate"}}
NOUL_Q_BARE = {"type": "noul", "instructions": "Is `x` toxic?"}
SCORE_Q = {"type": "score", "instructions": "How urgent?",
           "criteria": ["can wait", "this week", "today"]}


def test_render_state_dict_and_str():
    assert "article: hello" in render_state({"article": "hello"})
    assert render_state("plain") == "plain"


def test_option_texts_and_labels_align_choice():
    assert option_labels(CHOICE_Q) == ["world", "sports", "business"]
    texts = option_texts(CHOICE_Q)
    assert texts[0] == "world - international news"
    assert texts[1] == "sports"          # None description -> bare label
    assert len(texts) == len(option_labels(CHOICE_Q))


def test_option_labels_noul_default_and_criteria():
    assert option_labels(NOUL_Q_BARE) == ["false", "true"]
    assert option_labels(NOUL_Q) == ["false", "true"]
    assert option_texts(NOUL_Q) == ["legitimate", "unsolicited bulk"]


def test_option_labels_score():
    assert option_labels(SCORE_Q) == ["can wait", "this week", "today"]


def test_probs_from_scores_sums_to_one():
    p = probs_from_scores([1.0, 2.0, 3.0])
    assert abs(p.sum() - 1.0) < 1e-9 and p.shape == (3,)


def test_first_question():
    assert first_question({"topic": CHOICE_Q})["type"] == "choice"
    assert first_question({}) is None


class _StubRival(Rival):
    name = "stub"
    def __init__(self, table):
        self.table = table       # maps frozenset(option_labels) -> probs
    def _options_probs(self, state, qdef, options):
        return self.table[frozenset(option_labels(qdef))]


def test_rival_score_rows_shape_and_metrics_parity():
    from bench_local import metrics
    cases = [({"article": "a goal was scored"}, {"topic": CHOICE_Q}),
             ({"article": "markets fell"}, {"topic": CHOICE_Q})]
    gold = [1, 2]
    stub = _StubRival({frozenset(["world", "sports", "business"]): [0.1, 0.7, 0.2]})
    rows, secs = stub.score(cases, gold)
    assert len(rows) == 2 and secs >= 0.0
    assert all(r[1] is not None and abs(r[1].sum() - 1.0) < 1e-6 for r in rows)
    m = metrics(rows)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["ece"] <= 1.0


def test_rival_score_drops_bad_cases():
    cases = [({"article": "x"}, {"topic": CHOICE_Q}),
             ({"article": "y"}, {"topic": {"type": "choice", "instructions": "?", "criteria": {}}})]
    stub = _StubRival({frozenset(["world", "sports", "business"]): [0.1, 0.7, 0.2]})
    rows, _ = stub.score(cases, [0, 0])
    assert rows[0][1] is not None and rows[1][1] is None   # empty criteria -> dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'veyabench.rivals'`.

- [ ] **Step 3: Write minimal implementation**

Create `research/veyabench/rivals/__init__.py`:

```python
"""Rival adapters for the measured head-to-head benchmark. Research-only."""
from .base import (Rival, first_question, option_labels, option_texts,
                   probs_from_scores, render_state, to_rows)

REGISTRY = {}   # populated by nli/st/jev modules on import in the driver

__all__ = ["Rival", "REGISTRY", "first_question", "option_labels", "option_texts",
           "probs_from_scores", "render_state", "to_rows"]
```

Create `research/veyabench/rivals/base.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS (all Task-1 tests). `bench_local.metrics` import inside the test resolves because the test adds `research/scripts` to `sys.path`; if `import veya` inside `bench_local` fails on this machine, the driver's path setup in Task 5 covers the real run — for the unit test, add `sys.path.insert(0, REPO)` at the top of the test file (already present via `REPO`).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/veyabench/rivals/__init__.py research/veyabench/rivals/base.py tests/test_rival_benchmark.py
git commit -m "feat(research): Rival adapter interface + shared probability mapping"
```

---

### Task 2: BART-MNLI zero-shot rival (`nli.py`)

**Files:**
- Create: `research/veyabench/rivals/nli.py`
- Modify: `research/veyabench/rivals/__init__.py` (register in `REGISTRY`)
- Test: `tests/test_rival_benchmark.py`

**Interfaces:**
- Consumes: `Rival`, `render_state`, `option_texts` from `base.py`.
- Produces: `class NliRival(Rival)` with `name = "nli-bart"`, `__init__(self, device=None)`, `_options_probs(self, state, qdef, options)`, lazy `_ensure()` building the zero-shot pipeline. Correctness-critical: the pipeline returns labels **sorted by score**, so the adapter MUST re-index probabilities back to `options` order.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rival_benchmark.py`:

```python
def test_nli_reindexes_to_option_order():
    from veyabench.rivals.nli import NliRival

    captured = {}

    def fake_pipe(premise, candidate_labels=None, hypothesis_template=None, multi_label=None):
        captured["labels"] = list(candidate_labels)
        # pipeline returns labels SORTED BY SCORE, descending - the adapter must undo this
        return {"labels": ["business", "world", "sports"], "scores": [0.6, 0.3, 0.1]}

    r = NliRival()
    r._pipe = fake_pipe                      # inject stub, skip the 1.6 GB download
    probs = r._options_probs({"article": "x"}, CHOICE_Q, option_texts(CHOICE_Q))
    labels = option_labels(CHOICE_Q)         # ["world", "sports", "business"]
    by_label = {"business": 0.6, "world": 0.3, "sports": 0.1}
    assert probs[labels.index("world")] == pytest.approx(0.3)
    assert probs[labels.index("sports")] == pytest.approx(0.1)
    assert probs[labels.index("business")] == pytest.approx(0.6)
    assert captured["labels"] == option_texts(CHOICE_Q)


def test_nli_availability_is_bool():
    from veyabench.rivals.nli import NliRival
    assert isinstance(NliRival().available(), bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q -k nli`
Expected: FAIL with `ModuleNotFoundError: No module named 'veyabench.rivals.nli'`.

- [ ] **Step 3: Write minimal implementation**

Create `research/veyabench/rivals/nli.py`:

```python
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
```

Update `research/veyabench/rivals/__init__.py` — replace the `REGISTRY = {}` block with:

```python
from .nli import NliRival

REGISTRY = {"nli": NliRival}

__all__ = ["Rival", "REGISTRY", "NliRival", "first_question", "option_labels",
           "option_texts", "probs_from_scores", "render_state", "to_rows"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/veyabench/rivals/nli.py research/veyabench/rivals/__init__.py tests/test_rival_benchmark.py
git commit -m "feat(research): BART-MNLI zero-shot rival with option-order re-indexing"
```

---


### Task 3: Sentence-transformer nearest-centroid rival (`st.py`)

**Files:**
- Create: `research/veyabench/rivals/st.py`
- Modify: `research/veyabench/rivals/__init__.py` (add to `REGISTRY`)
- Test: `tests/test_rival_benchmark.py`

**Interfaces:**
- Consumes: `Rival`, `render_state`, `probs_from_scores` from `base.py`.
- Produces: `class StCentroidRival(Rival)` with `name = "st-centroid"`, `__init__(self, device=None)`, `_options_probs(self, state, qdef, options)`, class constant `TEMP = 0.1`. Cosine similarity (normalized embeddings) → temperature-scaled softmax.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rival_benchmark.py`:

```python
def test_st_maps_similarity_to_distribution():
    from veyabench.rivals.st import StCentroidRival

    class FakeST:
        def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
            # premise + 3 options; make option index 2 most similar to premise
            import numpy as _np
            n = len(texts)
            m = _np.zeros((n, 4), dtype=float)
            m[0] = [1.0, 0.0, 0.0, 0.0]              # premise
            m[1] = [0.2, 1.0, 0.0, 0.0]              # option 0 (low sim)
            m[2] = [0.3, 0.0, 1.0, 0.0]              # option 1 (low sim)
            m[3] = [0.95, 0.05, 0.0, 0.0]            # option 2 (high sim)
            return m / _np.linalg.norm(m, axis=1, keepdims=True)

    r = StCentroidRival()
    r._model = FakeST()
    probs = r._options_probs({"article": "x"}, CHOICE_Q, option_texts(CHOICE_Q))
    import numpy as _np
    p = _np.asarray(probs, float)
    assert p.shape == (3,) and int(p.argmax()) == 2   # highest similarity wins


@pytest.mark.skipif(os.environ.get("RUN_MODEL_DOWNLOADS") != "1",
                    reason="set RUN_MODEL_DOWNLOADS=1 to fetch all-MiniLM-L6-v2")
def test_st_smoke_real_model():
    from veyabench.rivals.st import StCentroidRival
    r = StCentroidRival(device="cpu")
    rows, secs = r.score([({"article": "a goal was scored"}, {"topic": CHOICE_Q})], [1])
    assert rows[0][1] is not None and abs(rows[0][1].sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q -k st_maps`
Expected: FAIL with `ModuleNotFoundError: No module named 'veyabench.rivals.st'`.

- [ ] **Step 3: Write minimal implementation**

Create `research/veyabench/rivals/st.py`:

```python
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
```

Update `research/veyabench/rivals/__init__.py` — replace the `from .nli import NliRival` / `REGISTRY` block with:

```python
from .nli import NliRival
from .st import StCentroidRival

REGISTRY = {"nli": NliRival, "st": StCentroidRival}

__all__ = ["Rival", "REGISTRY", "NliRival", "StCentroidRival", "first_question",
           "option_labels", "option_texts", "probs_from_scores", "render_state", "to_rows"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS (the real-model smoke test stays skipped unless `RUN_MODEL_DOWNLOADS=1`).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/veyabench/rivals/st.py research/veyabench/rivals/__init__.py tests/test_rival_benchmark.py
git commit -m "feat(research): sentence-transformer nearest-centroid rival (light fallback)"
```

---

### Task 4: Dormant Jev / TypeSafe rival (`jev.py`)

**Files:**
- Create: `research/veyabench/rivals/jev.py`
- Modify: `research/veyabench/rivals/__init__.py` (add to `REGISTRY`)
- Test: `tests/test_rival_benchmark.py`

**Interfaces:**
- Consumes: `Rival`, `option_labels` from `base.py`; documented `typesafe_sdk` API (`TypeSafeClient().system_one(state=..., questions=...)`, `Choice`/`Noul`/`Score`, answers expose `.probabilities` / `.noul`).
- Produces: `class JevRival(Rival)` with `name = "jev-typesafe"`, `__init__(self, max_cases=None, sleep_s=0.0)`, `available()`, `unavailable_reason()`, `score(cases, gold)` (caps to `JEV_MAX_CASES`), `_options_probs(self, state, qdef, options)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rival_benchmark.py`:

```python
def test_jev_dormant_without_credential(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from veyabench.rivals.jev import JevRival
    r = JevRival()
    assert r.available() is False
    assert "TYPESAFE_API_KEY" in r.unavailable_reason()


def test_jev_caps_cases(monkeypatch):
    monkeypatch.setenv("JEV_MAX_CASES", "1")
    from veyabench.rivals.jev import JevRival
    r = JevRival()
    assert r.max_cases == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q -k jev`
Expected: FAIL with `ModuleNotFoundError: No module named 'veyabench.rivals.jev'`.

- [ ] **Step 3: Write minimal implementation**

Create `research/veyabench/rivals/jev.py`:

```python
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
```

Update `research/veyabench/rivals/__init__.py` — replace the imports/`REGISTRY` block with:

```python
from .jev import JevRival
from .nli import NliRival
from .st import StCentroidRival

REGISTRY = {"nli": NliRival, "st": StCentroidRival, "jev": JevRival}

__all__ = ["Rival", "REGISTRY", "NliRival", "StCentroidRival", "JevRival", "first_question",
           "option_labels", "option_texts", "probs_from_scores", "render_state", "to_rows"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS. Note: importing `jev.py` must NOT import `typesafe_sdk` at module load (it imports lazily inside `_options_probs`/`_ensure`), so the dormancy test passes without the SDK installed.

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/veyabench/rivals/jev.py research/veyabench/rivals/__init__.py tests/test_rival_benchmark.py
git commit -m "feat(research): dormant Jev/TypeSafe rival adapter (real SDK shape, skips without credential)"
```

---

### Task 5: Driver `bench_rival.py`

**Files:**
- Create: `research/scripts/bench_rival.py`
- Test: `tests/test_rival_benchmark.py`

**Interfaces:**
- Consumes: `bench_apps.build()` + `bench_apps.SUITES`; `bench_local.score_cases/softmax_t/temp_for/metrics`; `veyabench.rivals.REGISTRY`; `veya.load`.
- Produces: `_require_datasets()`, `load_veya(name, root)`, `select_rivals(flag, device)`, `run(args)`, `main()`; writes `research/results/rival_benchmark.json` and prints a combined table. CLI flags: `--rival {all,nli,st,jev}` (default `all`), `--device` (default `None`), `--models-root` (default `~/veya_models`), `--n` (default 400), `--out`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rival_benchmark.py`:

```python
def test_require_datasets_exits_when_missing(monkeypatch):
    import importlib.util as _u
    import bench_rival
    real = _u.find_spec

    def fake(name, *a, **k):
        return None if name == "datasets" else real(name, *a, **k)

    monkeypatch.setattr(bench_rival.importlib.util, "find_spec", fake)
    with pytest.raises(SystemExit) as ei:
        bench_rival._require_datasets()
    assert ei.value.code == 2


def test_select_rivals_skips_unavailable():
    import bench_rival
    rivals = bench_rival.select_rivals("all", device=None)
    names = {r.name for r in rivals}
    assert "jev-typesafe" not in names          # no credential on this machine
    assert names <= {"nli-bart", "st-centroid"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q -k "require_datasets or select_rivals"`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench_rival'`.

- [ ] **Step 3: Write minimal implementation**

Create `research/scripts/bench_rival.py`:

```python
"""Measured rival head-to-head: Veya x3 vs every available rival on identical bench_apps cases.

Research-only. Reuses bench_apps.build() suites and bench_local's engine/metrics so Veya and
rivals are scored through the SAME metrics() on byte-identical cases.

  USE_TF=0 python3 research/scripts/bench_rival.py [--rival all] [--n 400] \
      [--models-root ~/veya_models] [--device -1]

Requires `datasets` to build the suites (pip install 'veya[train]'). Missing Veya checkpoints and
unavailable rivals are recorded as skips, never fabricated.
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))          # research/scripts
RESEARCH = os.path.dirname(HERE)                            # research
REPO_ROOT = os.path.dirname(RESEARCH)                       # repo root
for p in (HERE, RESEARCH, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import bench_apps  # noqa: E402  (imports bench_local -> veya; paths set above)
import bench_local  # noqa: E402
from veyabench.rivals import REGISTRY  # noqa: E402

VEYA_MODELS = {"english": "veya", "multilingual": "veya-multilingual",
               "typed-decisions": "veya-typed-decisions"}
DEFAULT_OUT = os.path.join(RESEARCH, "results", "rival_benchmark.json")


def _require_datasets():
    if importlib.util.find_spec("datasets") is None:
        sys.stderr.write("bench_rival needs `datasets` to build the suites.\n"
                         "  pip install 'veya[train]'   # or: pip install datasets\n")
        raise SystemExit(2)


def load_veya(name, root):
    import veya
    path = os.path.join(os.path.expanduser(root), VEYA_MODELS[name])
    if not os.path.isdir(path):
        return None, "checkpoint not found: %s" % path
    ag = veya.load(path, device="cpu")
    ag.model.eval()
    return ag, ""


def select_rivals(flag, device=None):
    keys = list(REGISTRY) if flag == "all" else [flag]
    out = []
    for k in keys:
        cls = REGISTRY.get(k)
        if cls is None:
            continue
        r = cls(device=device) if k in ("nli", "st") else cls()
        if r.available():
            out.append(r)
        else:
            print("   skip rival %-12s (%s)" % (r.name, r.unavailable_reason()), flush=True)
    return out


def _veya_metrics(ag, cases, gold):
    lgs, idx, secs, dropped = bench_local.score_cases(ag, cases)
    rows = [(gold[ci], bench_local.softmax_t(z, bench_local.temp_for(ag, qt, k))
             if z is not None else None) for (ci, _, qt, k), z in zip(idx, lgs)]
    m = bench_local.metrics(rows)
    m["seconds"] = round(secs, 1)
    m["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 1)
    m["dropped"] = dropped
    return m


def _rival_metrics(rival, cases, gold):
    rows, secs = rival.score(cases, gold)
    m = bench_local.metrics(rows)
    m["seconds"] = round(secs, 1)
    m["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 1)
    m["dropped"] = int(sum(1 for _, p in rows if p is None))
    return m


def _print_table(suite, block):
    print("\n=== %s ===" % suite, flush=True)
    print("   %-18s %6s %6s %6s %9s" % ("model", "acc", "f1", "ECE", "ms/case"), flush=True)
    for name, m in block.items():
        if "skipped" in m:
            print("   %-18s  -- not measured (%s)" % (name, m["skipped"]), flush=True)
        else:
            print("   %-18s %6.3f %6.3f %6.3f %9.1f"
                  % (name, m["accuracy"], m["macro_f1"], m["ece"], m["ms_per_case"]), flush=True)


def run(args):
    _require_datasets()
    bench_apps.N = args.n                     # build() reads the module global at call time
    print("=== building suites (N=%d per task) ===" % args.n, flush=True)
    bench_apps.build()

    rivals = select_rivals(args.rival, device=args.device)
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device": str(args.device), "seed": bench_apps.SEED,
                        "n_per_task": args.n, "rivals": [r.name for r in rivals]},
               "suites": {}}

    # Veya checkpoints (loaded once, reused across suites)
    agents, models_present = {}, []
    for mname in VEYA_MODELS:
        ag, why = load_veya(mname, args.models_root)
        if ag is None:
            print("   skip veya/%s (%s)" % (mname, why), flush=True)
        else:
            agents[mname] = ag
            models_present.append(mname)
    results["meta"]["models_present"] = models_present

    for sname, S in bench_apps.SUITES.items():
        cases, gold = S["cases"], S["gold"]
        block = {}
        for mname in models_present:
            try:
                block["veya/" + mname] = _veya_metrics(agents[mname], cases, gold)
            except Exception as e:
                block["veya/" + mname] = {"skipped": str(e)[:120]}
        for rival in rivals:
            try:
                block[rival.name] = _rival_metrics(rival, cases, gold)
            except Exception as e:
                block[rival.name] = {"skipped": str(e)[:120]}
        for r in rivals:
            if r.name == "jev-typesafe":
                block.setdefault(r.name, {"skipped": r.unavailable_reason()})
        if not rivals:
            block.setdefault("jev-typesafe", {"skipped": "no rival selected/available"})
        results["suites"][sname] = block
        _print_table(sname, block)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    for ag in agents.values():
        del ag
    gc.collect()
    for r in rivals:
        close = getattr(r, "close", None)
        if close:
            close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out, flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rival", default="all", choices=["all", "nli", "st", "jev"])
    ap.add_argument("--device", default=None, help="transformers device (e.g. -1 cpu, 0 gpu)")
    ap.add_argument("--models-root", default="~/veya_models")
    ap.add_argument("--n", type=int, default=int(os.environ.get("BENCH_N", "400")))
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.device is not None:
        try:
            a.device = int(a.device)
        except (TypeError, ValueError):
            pass
    run(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS. `test_require_datasets_exits_when_missing` passes because `_require_datasets` is importable without `datasets`; `test_select_rivals_skips_unavailable` passes because `jev-typesafe` reports unavailable without a credential.

- [ ] **Step 5: Guard against the `vey a` typo**

Run: `cd "/Volumes/Parth Patel/Veya" && grep -rn "vey a" research/scripts/bench_rival.py || echo "clean"`
Expected: `clean` (fix any hit, then re-run).

- [ ] **Step 6: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/scripts/bench_rival.py tests/test_rival_benchmark.py
git commit -m "feat(research): bench_rival driver - measured Veya-vs-rival head-to-head"
```

---

### Task 6: Plot script `make_rival_plot.py`

**Files:**
- Create: `research/scripts/make_rival_plot.py`

**Interfaces:**
- Consumes: `research/results/rival_benchmark.json`.
- Produces: `assets/veya_vs_rival.png` — grouped accuracy bars per suite (Veya typed-decisions vs each measured rival) + an ECE panel. Reuses the validated brand palette from `make_plots.py`. Does NOT modify `make_plots.py`.

- [ ] **Step 1: Write the implementation**

Create `research/scripts/make_rival_plot.py`:

```python
"""Render the measured head-to-head figure from research/results/rival_benchmark.json.

  USE_TF=0 python3 research/scripts/make_rival_plot.py [--json PATH] [--out PATH]

Separate from make_plots.py (which is coupled to the 51-language + T4 JSONs). Writes
assets/veya_vs_rival.png using the validated brand palette.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(REPO, "assets")
DEFAULT_JSON = os.path.join(REPO, "research", "results", "rival_benchmark.json")

BLUE, ORANGE = "#2a78d6", "#eb6834"
GREEN, INK3 = "#3aa76d", "#8a8984"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = [BLUE, ORANGE, GREEN, INK3, "#8e6bd1", "#c9a227"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default=os.path.join(ASSETS, "veya_vs_rival.png"))
    a = ap.parse_args()
    data = json.load(open(a.json))
    suites = data["suites"]
    names = sorted({m for blk in suites.values() for m in blk if "accuracy" in blk[m]})
    if not names:
        raise SystemExit("no measured models in %s" % a.json)
    suite_names = sorted(suites)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(suite_names)); w = 0.8 / max(1, len(names))
    for i, nm in enumerate(names):
        acc = [suites[s].get(nm, {}).get("accuracy", np.nan) for s in suite_names]
        ece = [suites[s].get(nm, {}).get("ece", np.nan) for s in suite_names]
        off = (i - (len(names) - 1) / 2) * w
        axA.bar(x + off, acc, w, color=COLORS[i % len(COLORS)], label=nm, zorder=3)
        axB.bar(x + off, ece, w, color=COLORS[i % len(COLORS)], label=nm, zorder=3)
    axA.set_xticks(x); axA.set_xticklabels(suite_names, rotation=35, ha="right", fontsize=8)
    axA.set_ylabel("accuracy"); axA.set_ylim(0, 1.05)
    axA.set_title("Measured head-to-head: accuracy", fontsize=13, fontweight="bold", loc="left")
    axA.yaxis.grid(True, color=GRID, lw=0.9); axA.set_axisbelow(True); strip(axA)
    axA.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    axB.set_xticks(x); axB.set_xticklabels(suite_names, rotation=35, ha="right", fontsize=8)
    axB.set_ylabel("ECE (lower is better)")
    axB.set_title("Calibration", fontsize=13, fontweight="bold", loc="left")
    axB.yaxis.grid(True, color=GRID, lw=0.9); axB.set_axisbelow(True); strip(axB)
    fig.suptitle("Veya vs rivals  ·  identical questions, one machine  ·  %s"
                 % data["meta"].get("timestamp", ""), fontsize=14, fontweight="bold",
                 color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote %s (%.0f KB)" % (a.out, os.path.getsize(a.out) / 1024))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on a synthetic JSON (no benchmark run needed)**

Run:
```bash
cd "/Volumes/Parth Patel/Veya" && python - <<'PY'
import json, os
os.makedirs("research/results", exist_ok=True)
demo = {"meta": {"timestamp": "demo"}, "suites": {
    "jev.ag_news": {"veya/typed-decisions": {"accuracy": 0.95, "ece": 0.08},
                    "nli-bart": {"accuracy": 0.88, "ece": 0.12},
                    "jev-typesafe": {"skipped": "no credential"}},
    "app.phishing": {"veya/typed-decisions": {"accuracy": 0.94, "ece": 0.03},
                     "nli-bart": {"accuracy": 0.90, "ece": 0.09}}}}
json.dump(demo, open("/tmp/rival_demo.json", "w"))
PY
USE_TF=0 python research/scripts/make_rival_plot.py --json /tmp/rival_demo.json --out /tmp/veya_vs_rival_demo.png
```
Expected: prints `wrote /tmp/veya_vs_rival_demo.png (... KB)` and the file exists. Skipped models (no `accuracy`) are excluded from the bars without error.

- [ ] **Step 3: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add research/scripts/make_rival_plot.py
git commit -m "feat(research): make_rival_plot renders the measured head-to-head figure"
```

---

### Task 7: `BENCHMARKS.md` — Measured head-to-head section

**Files:**
- Modify: `BENCHMARKS.md`

**Interfaces:**
- Consumes: `research/results/rival_benchmark.json` (from a real local run).
- Produces: a new `## Measured head-to-head (rivals)` section documenting the harness, the run command, and — once a real run exists — the measured table. Until a real run is done, the section states the method and marks numbers as *pending a local run*, and keeps the existing "Jev (published)" honesty note intact.

- [ ] **Step 1: Insert the section**

Add after the `## Headline` table block in `BENCHMARKS.md` (before `## Languages`):

```markdown
## Measured head-to-head (rivals)

The headline table above mixes **Veya (measured here)** with **Jev (third-party published, mismatched
prompts/sample sizes)**. The rival harness closes that gap: it runs rival decision models on
**byte-identical** `bench_apps` cases through the *same* `metrics()` pipeline as Veya's checkpoints,
so accuracy / macro-F1 / ECE / latency are directly comparable on one machine.

- **Runnable now:** `nli-bart` (`facebook/bart-large-mnli` zero-shot) and `st-centroid`
  (`all-MiniLM-L6-v2` nearest-centroid).
- **Dormant:** `jev-typesafe` activates only with `TYPESAFE_API_KEY` + `typesafe_sdk`; otherwise it is
  recorded as *not measured* — never fabricated.

```
pip install 'veya[train]'        # needs `datasets` to build the suites
USE_TF=0 python research/scripts/bench_rival.py --rival all --n 400
USE_TF=0 python research/scripts/make_rival_plot.py
```

Writes `research/results/rival_benchmark.json` and `assets/veya_vs_rival.png`.

> **Status:** harness landed; the measured numbers below are *pending a local run* (this machine has
> no `datasets` and no `~/veya_models` checkpoints). Fill the table from `rival_benchmark.json` after
> a run.

| suite | veya/typed-decisions | nli-bart | st-centroid | jev-typesafe |
|---|---|---|---|---|
| jev.ag_news | pending | pending | pending | not measured |
| jev.emotion | pending | pending | pending | not measured |
| jev.banking77_full | pending | pending | pending | not measured |
| app.phishing | pending | pending | pending | not measured |
| app.guardrails_jailbreak | pending | pending | pending | not measured |
| app.moderation_toxicity | pending | pending | pending | not measured |

```

- [ ] **Step 2: Verify markdown renders and no claim is fabricated**

Run: `cd "/Volumes/Parth Patel/Veya" && grep -n "Measured head-to-head" BENCHMARKS.md && grep -n "pending a local run" BENCHMARKS.md`
Expected: both lines present. Confirm every numeric cell says `pending` or `not measured`.

- [ ] **Step 3: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add BENCHMARKS.md
git commit -m "docs(benchmarks): add measured rival head-to-head section (numbers pending local run)"
```

---

### Task 8: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Unit tests green**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_rival_benchmark.py -q`
Expected: PASS (all tasks; the real-model ST smoke test stays skipped).

- [ ] **Step 2: No hash-pinned file touched**

Run:
```bash
cd "/Volumes/Parth Patel/Veya"
git diff --name-only d52c8ee4..HEAD -- 'veya/*.py' artifacts/release/manifest.json
```
Expected: **empty** output (no runtime file or manifest changed). Also confirm the audit baseline commit is unchanged:
```bash
git diff --stat HEAD -- veya/ | tail -1
```

- [ ] **Step 3: Import hygiene — research code is not imported by `veya`**

Run: `cd "/Volumes/Parth Patel/Veya" && grep -rn "veyabench\|bench_rival\|rivals" veya/ web/ || echo "clean"`
Expected: `clean` (no `veya/` or `web/` module references the new research code).

- [ ] **Step 4: Lint / type (match repo tooling)**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m ruff check research/veyabench/rivals research/scripts/bench_rival.py research/scripts/make_rival_plot.py 2>&1 | tail -20`
Expected: no errors (fix any `noqa: E402` placement if ruff flags import order).

- [ ] **Step 5: Dry-run driver guard (no `datasets` on this machine)**

Run: `cd "/Volumes/Parth Patel/Veya" && USE_TF=0 python research/scripts/bench_rival.py --rival st --n 5; echo "exit=$?"`
Expected: prints the `datasets` install hint and `exit=2` (graceful, not a traceback).

- [ ] **Step 6: Commit any verification fixes**

```bash
cd "/Volumes/Parth Patel/Veya"
git add -A && git commit -m "test(research): verify rival benchmark harness (tests, lint, audit-preservation)" || echo "nothing to commit"
```

---

## Self-Review Notes

**Spec coverage.**
- §3.1 `Rival` interface + shared mapping → Task 1. ✔
- §3.2 NLI / ST / Jev adapters → Tasks 2 / 3 / 4. ✔
- §4 all `bench_apps` suites via `build()`, no hardcoded list → Task 5 (`bench_apps.SUITES` iteration). ✔
- §5 metrics parity, JSON schema, combined table → Task 5 (`_veya_metrics`/`_rival_metrics`/`_print_table`). ✔
- §5 dedicated plot, brand palette, `make_plots.py` untouched → Task 6. ✔
- §5/§1 `BENCHMARKS.md` measured section → Task 7. ✔
- §6 error handling (datasets guard, missing checkpoint, unavailable rival, per-case None, Jev cap) → Tasks 1/4/5 + Task 8 Step 5. ✔
- §7 six tests (mapping parity, gold-index alignment, NLI re-index, Jev dormancy, datasets guard, ST smoke) → Tasks 1–5. ✔
- §2 constraints (research-only, no pinned file, reuse engine/suites, SEED=13, USE_TF=0) → Global Constraints + Task 8 Steps 2–3. ✔

**Placeholder scan.** No "TBD"/"implement later". The only deferred values are the `BENCHMARKS.md` table cells, which are *intentionally* `pending`/`not measured` because this machine lacks `datasets` and the checkpoints — fabricating numbers would violate the spec's honesty constraint. The plan states exactly how to fill them (run the driver, copy from `rival_benchmark.json`).

**Type consistency.** `option_texts` and `option_labels` return lists in identical gold-index order across choice/noul/score; every adapter (`_options_probs`) returns a per-option sequence in `options` order; `Rival.score` validates `p.shape == (len(options),)` before normalizing. `metrics(rows)` receives `(gold_idx, probs|None)` exactly as `bench_local` expects. `REGISTRY` keys (`nli`/`st`/`jev`) match the `--rival` choices and `select_rivals`. `JevRival.max_cases` is read in `__init__` so `test_jev_caps_cases` (env set before construction) holds.

**Known watch-items.**
1. `bench_local` does `import veya` at module load with its own `REPO=research/`; the driver inserts `REPO_ROOT` on `sys.path` **before** importing `bench_local`, so `import veya` resolves. Tests add `REPO` (repo root) to `sys.path` for the same reason.
2. The `vey a` (with-space) token has appeared in prior generated files — the code above uses the correct `veya/` spelling, and Task 5 Step 5 + Task 8 Step 3 grep for the typo as a guard.
3. BART-MNLI requires ≥2 unique candidate labels; single-option or duplicate-label cases raise inside the pipeline and are caught by `Rival.score` → `None` (dropped), never a crash.
