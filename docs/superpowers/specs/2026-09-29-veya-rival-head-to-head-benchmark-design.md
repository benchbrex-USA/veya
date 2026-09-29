# Veya Rival Head-to-Head Benchmark — Design

**Date:** 2026-09-29
**Status:** Proposed (awaiting user review)
**Track:** Research / benchmarking (`research/` only — nothing here is imported by the `veya` package)

## 1. Summary

Today every Jev number in this repo is a **pasted constant**, never measured. `research/README.md`
and `BENCHMARKS.md` both say so plainly: *"There is no TypeSafe API credential in this project, so
Jev was never run here … Treat those as indicative, not a controlled head-to-head."*

This design closes that gap with a **controlled, measured head-to-head benchmark**: rival decision
models answer **byte-identical** `(state, questions)` cases — the same cases, the same fixed seed,
the same `metrics()` pipeline — as Veya's three checkpoints. It ships:

- a **`Rival` adapter interface** that maps any rival's raw output into the exact probability-vector
  contract Veya's `metrics()` already consumes, across all three primitives (choice / noul / score);
- **two runnable local rivals** — BART-MNLI zero-shot (primary) and a sentence-transformer
  nearest-centroid (light fallback), flag-selectable;
- a **dormant Jev / TypeSafe adapter**, written against the real `typesafe_sdk` API, that activates
  only when `TYPESAFE_API_KEY` is set *and* `typesafe_sdk` imports — otherwise it skips with an
  explicit "no credential — Jev not measured here" message (preserving the repo's honesty);
- a **driver** (`bench_rival.py`) that runs Veya×3 plus every available rival over all 9
  `bench_apps` suites and writes `research/results/rival_benchmark.json` + one combined table;
- a **dedicated plot script** (`make_rival_plot.py`) and a new *Measured head-to-head* section in
  `BENCHMARKS.md`.

The result: the comparison table stops being "Veya (measured) vs Jev (published, mismatched prompts)"
and becomes "Veya vs NLI vs ST [vs Jev], identical questions, one machine, measured latency."

## 2. Binding constraints (verified against the repo)

1. **Research-only.** All new code lives under `research/`. `research/README.md` line 3–4: *"nothing
   here is imported by the `veya` package."* This must stay true.
2. **No hash-pinned file may be touched.** The release manifest hash-pins every `veya/*.py`. This
   work adds only `research/veyabench/rivals/*.py`, `research/scripts/bench_rival.py`,
   `research/scripts/make_rival_plot.py`, tests, and edits `BENCHMARKS.md` — none are pinned.
3. **Reuse the existing engine, do not fork it.** `bench_local.py` already provides
   `load(name)`, `score_cases(agent, cases)`, `softmax_t`, `temp_for`, `metrics(rows)`,
   `ece_score`, `macro_f1`. Rivals must be scored through the *same* `metrics()` so numbers are
   comparable by construction.
4. **Reuse the existing suites, do not rebuild them.** `bench_apps.build()` populates a module-level
   `SUITES` dict via `register(name, cases, gold, **meta)`. The driver imports and calls `build()`
   once, then feeds `SUITES[name]["cases"]`/`["gold"]` to both Veya and every rival.
5. **Case contract.** A case is `(state, questions)` where `state` is a dict of named text fields and
   `questions` is `{qid: {"type", "instructions", "criteria"}}`. `criteria` is a dict
   `{label: description_or_None}` for choice, an optional `{true, false}` dict for noul, and a list
   of level descriptions for score. `gold` is the index of the correct option (matching the order of
   `list(criteria.keys())` for choice / `[false, true]` for noul / level list for score).
6. **Backend env pins.** Every entry point sets `USE_TF=0`, `USE_TORCH=1`,
   `TOKENIZERS_PARALLELISM=false` before importing torch/transformers (the abseil deadlock noted in
   `research/README.md` line 17–18).
7. **sys.path bootstrap.** Scripts compute `REPO = dirname(dirname(abspath(__file__)))` and insert
   `REPO/veya` (and `REPO/notebooks` where needed) so they run from `research/scripts/`.
8. **Fixed seed.** `SEED = 13` (matches `bench_apps`/`bench_local`) so suite construction and any
   sampling are byte-identical across models and across runs.

### Environment prerequisites (verified on this machine)

| dependency | status | needed by |
|---|---|---|
| `transformers` | ✅ installed | BART-MNLI rival |
| `sentence-transformers` | ✅ installed | ST nearest-centroid rival |
| `datasets` | ❌ **not installed** | `bench_apps.build()` — required to construct any suite |
| `typesafe_sdk` | ❌ not installed | Jev adapter (dormant by design) |
| Veya checkpoints at `~/veya_models/{veya,veya-multilingual,veya-typed-decisions}` | machine-dependent | Veya side of the head-to-head |

Consequence: the driver must **degrade gracefully**. If `datasets` is missing it exits with a clear
install hint (`pip install 'veya[train]'` or `pip install datasets`). If a Veya checkpoint is missing
it skips that checkpoint (per-model try/except, the pattern already used in `bench_apps.main`). If no
rival is available it still runs Veya and records the empty rival set. The benchmark is useful with
any subset present; it never crashes silently.

## 3. Architecture — the `Rival` interface

New package `research/veyabench/rivals/`. The keystone is a single shared probability-mapping layer
so **every** rival — and Veya — is reduced to the same `rows = [(gold_idx, probs_vector), …]` that
`metrics()` scores. This is what makes the comparison *controlled* rather than indicative.

### 3.1 `research/veyabench/rivals/base.py`

```python
class Rival(Protocol):
    name: str                      # e.g. "nli-bart", "st-centroid", "jev-typesafe"
    def available(self) -> bool: ...           # deps + (for Jev) credential present
    def unavailable_reason(self) -> str: ...   # human-readable skip reason
    def score(self, cases, gold) -> tuple[list, float]:
        """Return (rows, seconds). rows = [(gold_idx, probs_vector|None), ...] in case order.
        probs_vector is a numpy array over the option/level axis, summing to ~1."""
```

Shared helpers (used by every adapter so mappings are identical):

- `render_state(state) -> str` — flatten the state dict into readable text (`key: value` lines),
  truncating long fields to a fixed budget (`STATE_CHAR_BUDGET = 4000`).
- `option_texts(qdef) -> list[str]` — for choice: `label — description` (or bare `label` when the
  description is `None`); for noul: `["false", "true"]` overridden by `criteria` descriptions when
  present; for score: the level strings. **Order defines the gold index**, exactly as
  `bench_apps`/`build_typed_decisions` already assume.
- `probs_from_scores(scores) -> np.ndarray` — softmax over raw rival scores to a distribution.
- `to_rows(per_question_probs, gold) -> rows` — assemble the `(gold_idx, probs_vector)` rows.

The gold index convention (must match the suites exactly):
- **choice** → `list(criteria.keys()).index(gold_label)` (already how `bench_apps` builds `gold`).
- **noul** → `1` if true else `0`; probs vector is `[1 - p_true, p_true]`.
- **score** → level index; probs vector over levels.

### 3.2 Rival adapters

| file | `name` | model | output → probs |
|---|---|---|---|
| `nli.py` | `nli-bart` | `facebook/bart-large-mnli` zero-shot pipeline | premise = `render_state` + instruction; candidate_labels = `option_texts`; normalize softmax over entailment scores |
| `st.py` | `st-centroid` | `sentence-transformers/all-MiniLM-L6-v2` | embed premise and each option text; cosine similarity → temperature-scaled softmax |
| `jev.py` | `jev-typesafe` | TypeSafe System One (`typesafe_sdk`) | real `system_one` call; read `.probabilities` (choice), `.noul` (→`[1-p,p]`), per-level probs (score) |

**NLI (primary, runnable now).** `transformers.pipeline("zero-shot-classification", model=…, device=…)`.
For each question: build `premise = render_state(state) + "\n\n" + instructions`, `candidate_labels =
option_texts(qdef)`, `hypothesis_template` chosen per primitive (choice: *"This text is about {}."*;
noul/score: a neutral *"{}"*). Take the pipeline's returned `scores`, re-softmax them into a
distribution aligned to `option_texts` order (the pipeline sorts by score, so we must re-index by
label — a correctness-critical detail). Batch across cases where the pipeline allows.

**ST nearest-centroid (light fallback, runnable now).** Encode premise and each option text with
`SentenceTransformer(all-MiniLM-L6-v2)`; cosine-similarity → softmax with a fixed temperature
(`ST_TEMP = 0.1`) → distribution over `option_texts` order. Cheaper and dependency-light; weaker on
nuanced judgments, which is exactly why it is the fallback.

**Jev / TypeSafe (dormant).** Real adapter against the documented sync API:
```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
with TypeSafeClient() as client:                 # reads TYPESAFE_API_KEY from env
    resp = client.system_one(state=state, questions=questions)
# choice: resp.answers[qid].probabilities -> [probs[k] for k in option_keys]
# noul:   p = resp.answers[qid].noul       -> [1 - p, p]
# score:  resp.answers[qid].probabilities  -> per-level distribution
```
`available()` returns `bool(os.environ.get("TYPESAFE_API_KEY")) and importlib.util.find_spec("typesafe_sdk")`.
When unavailable, the driver records `{"skipped": unavailable_reason()}` for Jev and the printed
table shows the row as *not measured* — never a fabricated number. Rate-limit awareness: a small
per-request sleep and a `JEV_MAX_CASES` cap (default 200) keep an eventual live run bounded.

## 4. Task set — every `bench_apps` suite

The driver imports `bench_apps`, calls `build()` once, and runs the head-to-head on every registered
suite (each already carries `gold` and `meta` with `jev=` / `in_training=` flags). `build()` currently
registers these ten names across the six workflow themes plus the three Jev-comparable sets:

`jev.ag_news`, `jev.emotion`, `jev.banking77_full`, `app.support_triage`, `app.email_spam`,
`app.phishing`, `app.guardrails_jailbreak`, `app.moderation_toxicity`, `app.rag_relevance`,
`app.model_routing_domain`. The driver iterates whatever `build()` registered (it does not hardcode
the list), so it tracks the source automatically as suites are added or renamed.

`BENCH_N` (default 400, env-overridable exactly as in `bench_apps`) caps cases per suite. Every model
— Veya×3 and each rival — sees the *same* `cases` list for a given suite in a given run.

## 5. Metrics parity & output

For each `(suite, model)` the driver records the full `metrics(rows)` dict — `n`, `accuracy`,
`macro_f1`, `ece`, `brier`, `nll`, `mean_confidence`, `acc_at_50_coverage` — plus `seconds`,
`ms_per_case`, and the suite's `meta` (`jev`, `in_training`, `n_labels`). Veya rows additionally
carry temperature calibration via `softmax_t(z, temp_for(agent, qt, k))` exactly as `bench_apps.main`
does; rivals produce already-normalized probabilities (calibration is a property we *measure*, not
apply, for rivals).

**Output JSON** → `research/results/rival_benchmark.json`:
```
{ "meta": {timestamp, device, seed, n_per_task, veya_version, models_present, rivals_present},
  "suites": { <suite>: { <model_name>: {<metrics…>} | {"skipped": reason} } } }
```

**Combined table** printed to stdout: one block per suite, rows = models, columns = accuracy /
macro_f1 / ECE / ms_per_case, with a `Δ vs best-rival` column and an explicit *not measured* marker
for any skipped model.

**Plot** → new `research/scripts/make_rival_plot.py` (a *separate* script; `make_plots.py` stays
untouched because it is coupled to the 51-language + T4 JSONs). Reads `rival_benchmark.json`, writes
`assets/veya_vs_rival.png`: grouped bars per suite (accuracy) for Veya-typed-decisions vs each
measured rival, plus a calibration (ECE) panel, reusing the validated brand palette
(`BLUE #2a78d6`, `ORANGE #eb6834`, ink/surface tokens) from `make_plots.py`.

## 6. Error handling

- Missing `datasets` → driver prints the install hint and `SystemExit(2)` before building suites.
- Missing Veya checkpoint → per-model try/except records `{"skipped": "checkpoint not found: <path>"}`
  and continues (mirrors `bench_apps.main`'s per-suite try/except).
- Rival `available()` False → recorded as `{"skipped": reason}`; never crashes the run.
- A single case failing inside a rival (e.g. empty option set, pipeline error) → that case's
  `probs_vector` is `None` and it is dropped by `metrics()` (which already filters `r[1] is None`),
  with a per-suite `dropped` count recorded. No partial-state corruption across cases.
- Jev live-run network/HTTP error → caught per request, recorded, and the run continues; a nonzero
  error count is surfaced in `meta`.
- Every adapter is deterministic given the seed; no adapter mutates the shared `cases`/`gold`.

## 7. Testing strategy

Tests live in `tests/test_rival_benchmark.py` and run **without** downloading BART-MNLI or needing
`datasets`/network, via stubs:

1. **Probability-mapping parity (unit).** For a tiny synthetic suite (2 choice, 1 noul, 1 score),
   a `StubRival` returning fixed distributions produces `rows` whose shape and gold indices match
   `build_typed_decisions`' convention; `metrics(rows)` returns finite accuracy/ECE in `[0,1]`.
2. **`option_texts` / gold-index alignment.** Assert `option_texts` order equals
   `list(criteria.keys())` for choice, `[false, true]`-equivalent for noul, and the level list for
   score — the single most correctness-critical invariant.
3. **NLI re-indexing.** With a monkeypatched pipeline that returns label→score in *sorted* order,
   assert the adapter re-aligns probabilities to `option_texts` order (guards the sort bug).
4. **Jev dormancy.** With no `TYPESAFE_API_KEY`, `JevRival().available() is False` and
   `unavailable_reason()` names the credential; the driver records it as skipped, not fabricated.
5. **Graceful degradation.** With `datasets` import blocked, the driver exits with the install hint
   (exit code 2) rather than raising an opaque `ImportError`.
6. **ST rival smoke (optional, network-gated).** Marked `@pytest.mark.skipif` unless
   `RUN_MODEL_DOWNLOADS=1`; encodes 3 sentences and asserts a normalized distribution.

The heavy end-to-end run (real datasets + real models) is a documented manual command, not part of
`pytest`, consistent with how `bench_local`/`bench_apps` are run today.

## 8. Out of scope

- No change to any `veya/*.py` runtime file, the manifest, or the audited `web/` gateway.
- No retraining, no new checkpoints, no calibration refit of rivals (calibration is measured as-is).
- No live Jev run in CI (no credential); the adapter is built and tested only in its dormant path.
- No rewrite of `make_plots.py` or the existing `bench_*` scripts — the new driver/plot are additive.
- No ordinal-Score research path; served Score is categorical (per `BENCHMARKS.md` limits) and the
  benchmark scores it as a categorical distribution over levels.

## 9. Sequencing (previews the implementation plan)

1. `rivals/base.py` — `Rival` protocol, `render_state`, `option_texts`, `probs_from_scores`,
   `to_rows` (+ unit tests 1–2).
2. `rivals/nli.py` — BART-MNLI adapter with correct label re-indexing (+ test 3).
3. `rivals/st.py` — sentence-transformer nearest-centroid adapter (+ smoke test 6).
4. `rivals/jev.py` — dormant TypeSafe adapter against the real SDK shape (+ test 4).
5. `scripts/bench_rival.py` — driver: `datasets` guard, `build()` reuse, Veya×3 + rivals, JSON + table
   (+ test 5).
6. `scripts/make_rival_plot.py` — dedicated plot.
7. `BENCHMARKS.md` — new *Measured head-to-head* section (populated after a real local run).
8. Full verification: `pytest tests/test_rival_benchmark.py`, then the documented manual
   end-to-end command.

## 10. Open risks

- **`datasets` absent on this machine.** A real local run needs `pip install 'veya[train]'`. The
  harness and all tests work without it; only the live numbers require it.
- **Veya checkpoints at `~/veya_models` may be absent.** Driver supports `--models-root` override and
  skips missing checkpoints, so the rival-only side still produces a measured table.
- **BART-MNLI is ~1.6 GB.** First NLI run downloads it; `--rival st` gives a light path with no large
  download. Both are flag-selectable via `--rival {nli,st,jev,all}`.
- **Latency comparability.** Veya latency is CPU here vs a published T4 32.8 ms; the table reports
  `ms_per_case` *on the same machine as the run* and labels the device in `meta`, so no cross-device
  claim is implied.
