# Veya Track A: Runtime & API Advancement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance the Veya decision engine's runtime and developer-facing API — richer gateway responses, graceful concurrency, and a research-only accelerated batch server — without touching the audited release bundle.

**Architecture:** All production-path changes live in `web/` (not hash-pinned by the release manifest). New derived response fields (top-k, confidence band, split latency, provenance) are computed gateway-side from data the upstream already returns. A research-only cross-state dynamic batcher lives in `veya/experimental/`, importing the stable core but never imported by it. The frozen `veya/*.py` runtime files and `artifacts/release/manifest.json` are read-only throughout.

**Tech Stack:** Python 3.10+ stdlib (`http.server`, `threading`, `concurrent.futures`, `math`, `hashlib`), torch for the experimental batcher, pytest for Python tests, Node's built-in test runner for `.mjs` tests.

**Spec:** `docs/superpowers/specs/2026-09-29-veya-track-a-runtime-api-design.md`

## Global Constraints

- **Do NOT modify any hash-pinned `veya/*.py` file**: `__init__.py`, `agent.py`, `calibration.py`, `common.py`, `email.py`, `lang.py`, `manifest.py`, `presets.py`, `release_bundle.py`, `reranker.py`, `router.py`, `selective.py`, `server.py`, `shortlist.py`, `shortlist_artifact.py` (per `artifacts/release/manifest.json` `runtime_files`). Editing any of these breaks the audited load path.
- **Do NOT modify `artifacts/release/manifest.json`** or any file it hashes.
- **Response changes are additive only**: never remove or rename an existing field in the gateway response envelope. `web/static/model-client.mjs` `parseDecision` requires `decision`, `primitive`, `request_id`, `gateway_latency_ms`, and `deferred`; keep all of them.
- **The gateway (`web/server.py`) must not import any `veya` module.** It talks to the model over HTTP. Reimplement small math (Wilson band) locally in `web/`.
- **`veya/experimental/` modules may import the stable core; the stable core and `web/` must never import `veya/experimental/`.**
- **Heavy deps imported lazily inside functions** (`torch`, `onnx`, `onnxruntime`), per project convention.
- **No state text or candidate text in logs.** The existing `_safe_telemetry` shape is unchanged.
- **Tunable constants** (verbatim from spec): Wilson `n_eff = 100`, `z = 1.96`; `QUEUE_TIMEOUT_S = 2.0` (env `VEYA_QUEUE_TIMEOUT_S`); batcher `MAX_BATCH = 16`, `MAX_WAIT_MS = 10`.
- Run tests from the repo root: `cd "/Volumes/Parth Patel/Veya"`.

---

## File Structure

- **Create `web/enrich.py`** — pure gateway-side derivation functions (`top_k`, `prob_band`, `enrich`). No HTTP, no imports from `veya`. One responsibility: turn a validated response into an enriched response.
- **Modify `web/server.py`** — wire enrichment + split latency into `do_POST`; replace the non-blocking inflight reject with a bounded wait-queue; add identical-request dedup coalescing.
- **Modify `web/static/model-client.mjs`** — extend the `VeyaResponse` typedef (no validation tightening; new fields are optional).
- **Modify `web/static/playground.mjs`** — render top-k and confidence band when present; feature-detect so older responses still render.
- **Create `veya/experimental/batch_server.py`** — research-only dynamic batcher + HTTP wrapper + optional int8/ONNX acceleration.
- **Create `tests/test_gateway_enrich.py`** — unit tests for `web/enrich.py`.
- **Modify `tests/test_web_gateway.py`** — gateway integration tests for new fields, queue, dedup.
- **Modify `tests/test_site_content.mjs`** — client rendering of new fields.
- **Create `tests/test_experimental_batch_server.py`** — batcher parity + failure-propagation tests.
- **Modify `tests/test_release_bundle.py`** — regression guard proving manifest `runtime_files` hashes still verify.

---

## Task 1: Gateway enrichment helpers (`web/enrich.py`)

**Files:**
- Create: `web/enrich.py`
- Test: `tests/test_gateway_enrich.py`

**Interfaces:**
- Consumes: nothing (pure functions on plain dicts).
- Produces:
  - `top_k(probabilities: dict[str, float] | None, k: int = 5) -> list[dict] | None`
  - `prob_band(probability: float, n_eff: int = 100, z: float = 1.96) -> dict[str, float] | None`
  - `enrich(result: dict, primitive: str, *, upstream_latency_ms: int | None = None, queue_wait_ms: int | None = None) -> None` (mutates `result` in place; never raises)
  - Constants `TOP_K = 5`, `BAND_N_EFF = 100`, `BAND_Z = 1.96`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gateway_enrich.py`:

```python
"""Unit tests for gateway-side response enrichment. No HTTP, no model."""
from web import enrich


def test_top_k_orders_and_truncates():
    probs = {"a": 0.1, "b": 0.7, "c": 0.2, "d": 0.0, "e": 0.0, "f": 0.0}
    ranked = enrich.top_k(probs)
    assert [entry["label"] for entry in ranked] == ["b", "c", "a", "d", "e"]
    assert ranked[0]["probability"] == 0.7
    assert enrich.top_k(None) is None
    assert enrich.top_k({}) is None


def test_prob_band_brackets_probability():
    band = enrich.prob_band(0.8)
    assert band is not None
    assert band["prob_low"] <= 0.8 <= band["prob_high"]
    assert 0.0 <= band["prob_low"] and band["prob_high"] <= 1.0
    assert enrich.prob_band(1.5) is None
    assert enrich.prob_band(float("nan")) is None


def test_enrich_choice_adds_top_k_band_and_provenance():
    result = {
        "decision": "answer",
        "deferred": False,
        "primitive": "choice",
        "answer": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "route": "packed",
        "calibration_version": "test",
        "model_version": "0.3.5",
    }
    enrich.enrich(result, "choice", upstream_latency_ms=12, queue_wait_ms=0)
    assert result["top_k"][0] == {"label": "billing", "probability": 0.9}
    assert result["prob_band"]["prob_low"] <= 0.9 <= result["prob_band"]["prob_high"]
    assert result["upstream_latency_ms"] == 12
    assert result["queue_wait_ms"] == 0
    assert result["provenance"]["route"] == "packed"
    assert result["provenance"]["calibration_version"] == "test"


def test_enrich_noul_bands_answer_no_top_k():
    result = {"decision": "answer", "deferred": False, "primitive": "noul", "answer": 0.7}
    enrich.enrich(result, "noul")
    assert "top_k" not in result
    assert result["prob_band"]["prob_low"] <= 0.7 <= result["prob_band"]["prob_high"]


def test_enrich_systemone_enriches_each_answer():
    result = {
        "decision": "answer",
        "deferred": False,
        "primitive": "systemone",
        "answers": {
            "q1": {"type": "choice", "answer": "a", "probabilities": {"a": 0.8, "b": 0.2}},
            "q2": {"type": "noul", "answer": 0.4},
        },
    }
    enrich.enrich(result, "systemone")
    assert result["answers"]["q1"]["top_k"][0]["label"] == "a"
    assert result["answers"]["q2"]["prob_band"]["prob_low"] <= 0.4


def test_enrich_skips_deferred_and_never_raises():
    deferred = {"decision": "defer", "deferred": True, "primitive": "choice", "answer": None}
    enrich.enrich(deferred, "choice")
    assert "top_k" not in deferred and "prob_band" not in deferred
    # Malformed input must not raise out of enrich.
    enrich.enrich({"decision": "answer", "deferred": False, "probabilities": "bad"}, "choice")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_gateway_enrich.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.enrich'`.

- [ ] **Step 3: Write minimal implementation**

Create `web/enrich.py`:

```python
"""Gateway-side response enrichment. Pure functions; no HTTP, no `veya` imports.

Every field here is derived from data the gateway has already validated, so the
audited upstream contract and the frozen release manifest are unaffected. These
are additive conveniences for API consumers, not new model guarantees: the
confidence band in particular is an *indicative* interval derived from the
reported probability over an assumed effective support, NOT a frequentist
guarantee over new samples (automation remains disabled in the audited release).
"""
from __future__ import annotations

import math
from typing import Any

TOP_K = 5
BAND_N_EFF = 100  # assumed calibration-set support for the indicative Wilson band
BAND_Z = 1.96

_PROVENANCE_KEYS = (
    "calibration_version",
    "calibration",
    "route",
    "reranker_id",
    "model_version",
    "release",
    "architecture_version",
)


def top_k(probabilities: dict[str, float] | None, k: int = TOP_K) -> list[dict[str, Any]] | None:
    """Highest-probability labels first, truncated to k. None when there is no distribution."""
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return [{"label": label, "probability": round(float(p), 4)} for label, p in ranked[:k]]


def prob_band(probability: float, n_eff: int = BAND_N_EFF, z: float = BAND_Z) -> dict[str, float] | None:
    """Indicative Wilson-score interval around a reported probability in [0, 1]."""
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or not 0.0 <= p <= 1.0 or n_eff <= 0:
        return None
    denom = 1.0 + z * z / n_eff
    center = (p + z * z / (2 * n_eff)) / denom
    margin = z * math.sqrt(max(0.0, p * (1.0 - p) / n_eff + z * z / (4 * n_eff * n_eff))) / denom
    return {
        "prob_low": round(max(0.0, center - margin), 4),
        "prob_high": round(min(1.0, center + margin), 4),
    }


def _enrich_one(kind: str, out: dict[str, Any]) -> None:
    if not isinstance(out, dict):
        return
    if kind == "noul":
        value = out.get("answer")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            band = prob_band(value)
            if band is not None:
                out["prob_band"] = band
        return
    probabilities = out.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return
    ranked = top_k(probabilities)
    if ranked is not None:
        legend = out.get("legend")
        if isinstance(legend, dict):
            for entry in ranked:
                if entry["label"] in legend:
                    entry["text"] = legend[entry["label"]]
        out["top_k"] = ranked
    numeric = [float(v) for v in probabilities.values() if isinstance(v, (int, float))]
    if numeric:
        band = prob_band(max(numeric))
        if band is not None:
            out["prob_band"] = band


def enrich(
    result: dict[str, Any],
    primitive: str,
    *,
    upstream_latency_ms: int | None = None,
    queue_wait_ms: int | None = None,
) -> None:
    """Add derived top_k / prob_band / latency / provenance fields in place. Never raises."""
    try:
        if not isinstance(result, dict) or result.get("decision") != "answer" or result.get("deferred"):
            return
        if primitive == "systemone":
            answers = result.get("answers")
            if isinstance(answers, dict):
                for answer in answers.values():
                    if isinstance(answer, dict):
                        _enrich_one(str(answer.get("type", "")), answer)
        else:
            _enrich_one(primitive, result)
        if upstream_latency_ms is not None:
            result["upstream_latency_ms"] = int(upstream_latency_ms)
        if queue_wait_ms is not None:
            result["queue_wait_ms"] = int(queue_wait_ms)
        provenance = {key: result[key] for key in _PROVENANCE_KEYS if result.get(key) is not None}
        if provenance:
            result["provenance"] = provenance
    except Exception:
        # Enrichment is best-effort; a derivation failure must never fail the request.
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_gateway_enrich.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add web/enrich.py tests/test_gateway_enrich.py
git commit -m "feat(web): add gateway-side response enrichment helpers"
```

---

## Task 2: Wire enrichment + split latency into the gateway (`web/server.py`)

**Files:**
- Modify: `web/server.py` (imports ~line 23; `do_POST` inflight/upstream/enrich region ~lines 496-543)
- Test: `tests/test_web_gateway.py`

**Interfaces:**
- Consumes: `web.enrich.enrich(result, primitive, *, upstream_latency_ms, queue_wait_ms)` (Task 1).
- Produces: gateway responses for `/api/infer` now include `top_k`, `prob_band`, `provenance`, `upstream_latency_ms`, `queue_wait_ms` (additive; existing fields retained).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_gateway.py`:

```python
def test_enrichment_fields_present_on_choice(http_pair):
    _, port = http_pair
    status, out, _ = call(port, CHOICE)
    assert status == 200
    assert out["top_k"][0]["label"] == "billing"
    assert out["prob_band"]["prob_low"] <= 1.0 <= out["prob_band"]["prob_high"]
    assert out["provenance"]["route"] == "packed"
    assert out["provenance"]["model_version"] == "0.3.5"
    assert isinstance(out["upstream_latency_ms"], int)
    assert isinstance(out["queue_wait_ms"], int)
    # Backward compatibility: existing fields still present.
    assert out["gateway_latency_ms"] >= 0
    assert out["request_id"] and out["model"] == "Veya Core"


def test_enrichment_absent_when_deferred(http_pair):
    _, port = http_pair
    status, out, _ = call(port, {**CHOICE, "state": "मुझसे दो बार शुल्क लिया गया।"})
    assert status == 200 and out["deferred"]
    assert "top_k" not in out and "prob_band" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_web_gateway.py::test_enrichment_fields_present_on_choice -v`
Expected: FAIL — `KeyError: 'top_k'`.

- [ ] **Step 3: Write minimal implementation**

In `web/server.py`, add the import next to the existing `.contracts` import (after line 30):

```python
from .enrich import enrich
```

Add a module constant near the other timing constants (after `INFER_TIMEOUT_S = 20`, line 76):

```python
QUEUE_TIMEOUT_S = float(os.environ.get("VEYA_QUEUE_TIMEOUT_S", "2.0"))
```

Replace the inflight/upstream/enrich block in `do_POST`. The current code (lines 496-543) is:

```python
            if not inflight.acquire(blocking=False):
                status, error_code = 503, "model_busy"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Model is busy. Try again shortly.",
                        "retry_after": 2,
                        "request_id": request_id,
                    },
                    2,
                )
                return
            try:
                raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
                upstream = _upstream_request(
                    "/v1/" + primitive, raw, timeout=INFER_TIMEOUT_S
                )
                result = validate_response(
                    cast(Primitive, primitive), upstream, body
                )
            finally:
                inflight.release()
            try:
                version = _version()
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                version = {}
            for key in (
                "model_version",
                "release",
                "architecture_version",
                "supported_languages",
                "release_status",
            ):
                if key in version:
                    result[key] = version[key]
            # Public product name; the upstream's internal checkpoint identifier
            # remains an implementation detail and is not a user-facing model name.
            result["model"] = "Veya Core"
            result["request_id"] = request_id
            result["gateway_latency_ms"] = round((time.monotonic() - started) * 1000)
```

Replace it with:

```python
            wait_start = time.monotonic()
            if not inflight.acquire(blocking=True, timeout=QUEUE_TIMEOUT_S):
                status, error_code = 503, "model_busy"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Model is busy. Try again shortly.",
                        "retry_after": 2,
                        "request_id": request_id,
                    },
                    2,
                )
                return
            queue_wait_ms = round((time.monotonic() - wait_start) * 1000)
            try:
                raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
                upstream_start = time.monotonic()
                upstream = _upstream_request(
                    "/v1/" + primitive, raw, timeout=INFER_TIMEOUT_S
                )
                upstream_latency_ms = round((time.monotonic() - upstream_start) * 1000)
                result = validate_response(
                    cast(Primitive, primitive), upstream, body
                )
            finally:
                inflight.release()
            try:
                version = _version()
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                version = {}
            for key in (
                "model_version",
                "release",
                "architecture_version",
                "supported_languages",
                "release_status",
            ):
                if key in version:
                    result[key] = version[key]
            # Public product name; the upstream's internal checkpoint identifier
            # remains an implementation detail and is not a user-facing model name.
            result["model"] = "Veya Core"
            result["request_id"] = request_id
            result["gateway_latency_ms"] = round((time.monotonic() - started) * 1000)
            enrich(
                result,
                cast(Primitive, primitive),
                upstream_latency_ms=upstream_latency_ms,
                queue_wait_ms=queue_wait_ms,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_web_gateway.py -v`
Expected: PASS (new tests plus all pre-existing gateway tests — the blocking acquire with a 2s timeout does not change single-request behavior).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add web/server.py tests/test_web_gateway.py
git commit -m "feat(web): enrich infer responses with top-k, band, latency, provenance"
```

---

## Task 3: Render new fields in the static client

**Files:**
- Modify: `web/static/model-client.mjs:4` (typedef only)
- Modify: `web/static/playground.mjs:11-12` (`probabilities`/`details` render helpers)
- Test: `tests/test_site_content.mjs`

**Interfaces:**
- Consumes: gateway response fields `top_k`, `prob_band`, `provenance`, `upstream_latency_ms`, `queue_wait_ms` (Task 2).
- Produces: playground renders top-k and confidence band when present; degrades gracefully when absent.

- [ ] **Step 1: Write the failing test**

Inspect the existing `tests/test_site_content.mjs` structure first:

Run: `cd "/Volumes/Parth Patel/Veya" && sed -n '1,40p' tests/test_site_content.mjs`

Then append a test matching the file's existing style (Node built-in test runner). If the file uses `import { test } from 'node:test'` and reads static sources, assert the playground source references the new fields:

```javascript
test('playground renders top-k and confidence band', async () => {
  const src = (await import('node:fs/promises')).readFile(
    new URL('../web/static/playground.mjs', import.meta.url), 'utf8');
  const playground = await src;
  assert.match(playground, /top_k/);
  assert.match(playground, /prob_band/);
});
```

(If `tests/test_site_content.mjs` instead runs a DOM/browser harness, follow that file's existing pattern to mount `renderResult` with a sample object containing `top_k`/`prob_band` and assert the rendered HTML contains the band text. Match the file's established conventions — do not invent a new harness.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && node --test tests/test_site_content.mjs`
Expected: FAIL — playground source does not yet reference `top_k`.

- [ ] **Step 3: Write minimal implementation**

In `web/static/model-client.mjs`, extend the `VeyaResponse` typedef (line 4) by appending the optional fields before the closing `}}`:

```javascript
/** @typedef {{decision:'answer'|'defer',deferred:boolean,primitive:Primitive,answer?:string|number|null,probabilities?:Record<string,number>|null,answers?:Record<string,object>,defer_reason?:string|null,route?:string|null,model?:string|null,model_version?:string,release?:string,request_id:string,gateway_latency_ms:number,upstream_latency_ms?:number,queue_wait_ms?:number,top_k?:Array<{label:string,probability:number,text?:string}>,prob_band?:{prob_low:number,prob_high:number},provenance?:Record<string,string>}} VeyaResponse */
```

In `web/static/playground.mjs`, extend the `probabilities(p, legend)` helper (line 11) so that, after the distribution list, it appends a top-k / band block when those fields exist. Replace the existing `function probabilities(p,legend){...}` with a version that also accepts the whole answer object. Because `probabilities` is called with `(probs, o.legend)` in several places, add a new sibling helper and call it from `renderResult`:

```javascript
function band(o){if(!o||typeof o!=='object')return'';const parts=[];if(Array.isArray(o.top_k)&&o.top_k.length){parts.push('<div class="prob-list"><div class="eyebrow">Top options</div>'+o.top_k.map(t=>`<div class="prob-row"><div class="prob-label"><span>${esc(t.text??t.label)}</span><strong>${(Number(t.probability)*100).toFixed(1)}%</strong></div></div>`).join('')+'</div>')}if(o.prob_band&&typeof o.prob_band.prob_low==='number'&&typeof o.prob_band.prob_high==='number'){parts.push(`<div class="note">Indicative 95% band: ${(o.prob_band.prob_low*100).toFixed(1)}%–${(o.prob_band.prob_high*100).toFixed(1)}%. Not a correctness guarantee.</div>`)}return parts.join('')}
```

Then in `renderResult`, for the non-systemone answer branch, insert `${band(o)}` immediately after the existing `${probabilities(probs,o.legend)}` call; and in the systemone branch, insert `${band(a)}` after `${probabilities(a.probabilities,a.legend)}` inside the per-answer map.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && node --test tests/test_site_content.mjs`
Expected: PASS.

- [ ] **Step 5: Manual render check**

Run the gateway locally and load the playground to confirm the band and top-k render and that an older (unenriched) response still renders:

```bash
cd "/Volumes/Parth Patel/Veya" && python -c "import ast,sys; ast.parse(open('web/static/playground.mjs').read())" 2>/dev/null || node --check web/static/playground.mjs
```

Expected: no syntax error. (Full visual check happens at execution review; `node --check` guards syntax.)

- [ ] **Step 6: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add web/static/model-client.mjs web/static/playground.mjs tests/test_site_content.mjs
git commit -m "feat(web): render top-k and confidence band in the playground"
```

---

## Task 4: Bounded wait-queue behavior test (`web/server.py`)

The implementation of the blocking acquire already landed in Task 2. This task adds the concurrency tests that lock in the queue behavior and makes `QUEUE_TIMEOUT_S`/`MAX_INFLIGHT` test-tunable.

**Files:**
- Modify: `web/server.py` (only if a test hook is needed; the constant already exists)
- Test: `tests/test_web_gateway.py`

**Interfaces:**
- Consumes: `inflight` semaphore, `QUEUE_TIMEOUT_S` (Task 2).
- Produces: verified queue-then-succeed and queue-timeout behavior under concurrency.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_gateway.py`:

```python
def test_third_concurrent_request_queues_then_succeeds(http_pair, monkeypatch):
    mode, port = http_pair
    monkeypatch.setattr(gateway, "QUEUE_TIMEOUT_S", 3.0)
    # MAX_INFLIGHT is fixed at 2 by the module-level semaphore; a slow upstream
    # (mode "timeout" sleeps 0.2s) means a 3rd concurrent request must wait, then pass.
    mode["kind"] = "timeout"
    results = []

    def worker():
        results.append(call(port, CHOICE)[0])

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert results.count(200) == 3


def test_queue_timeout_returns_model_busy(http_pair, monkeypatch):
    mode, port = http_pair
    monkeypatch.setattr(gateway, "QUEUE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(gateway, "INFER_TIMEOUT_S", 5)
    mode["kind"] = "timeout"  # each upstream call sleeps 0.2s
    # Force the semaphore to a single slot so two concurrent requests contend.
    monkeypatch.setattr(gateway, "inflight", threading.BoundedSemaphore(1))
    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        barrier.wait()
        outcomes.append(call(port, CHOICE)[0])

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert 200 in outcomes and 503 in outcomes
```

- [ ] **Step 2: Run test to verify behavior**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_web_gateway.py::test_third_concurrent_request_queues_then_succeeds tests/test_web_gateway.py::test_queue_timeout_returns_model_busy -v`
Expected: PASS (Task 2 already made the acquire blocking with a timeout). If `test_queue_timeout_returns_model_busy` fails because the module-level `inflight` is captured elsewhere, confirm `do_POST` references the module global `inflight` (it does) so monkeypatching `gateway.inflight` takes effect.

- [ ] **Step 3: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add tests/test_web_gateway.py
git commit -m "test(web): cover bounded wait-queue grace and timeout"
```

---

## Task 5: Identical-request dedup coalescing (`web/server.py`)

**Files:**
- Modify: `web/server.py` (module state near line 82; `do_POST` upstream region)
- Test: `tests/test_web_gateway.py`

**Interfaces:**
- Consumes: validated `body` dict, `_upstream_request`, `validate_response`.
- Produces:
  - `_dedup_key(primitive: str, body: dict) -> str` (stable SHA-256 of canonical JSON)
  - module state `dedup_map: dict[str, concurrent.futures.Future]`, `dedup_lock = threading.Lock()`, `DEDUP_MAX = 256`
  - Behavior: concurrent identical requests issue ONE upstream call; each caller independently validates + enriches and gets its own `request_id`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_gateway.py`:

```python
def test_identical_concurrent_requests_share_one_upstream_call(http_pair):
    mode, port = http_pair
    mode["kind"] = "timeout"  # slow upstream widens the coalescing window
    gateway.rate_hits.clear()
    outcomes = []
    barrier = threading.Barrier(3)

    def worker():
        barrier.wait()
        status, out, _ = call(port, CHOICE)
        outcomes.append((status, out.get("request_id")))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(status == 200 for status, _ in outcomes)
    # Each caller gets a distinct request_id...
    assert len({rid for _, rid in outcomes}) == 3
    # ...but only one upstream /v1/choice call was made for the identical bodies.
    choice_calls = [r for r in mode["requests"] if r[0] == "/v1/choice"]
    assert len(choice_calls) == 1


def test_dedup_key_is_stable_and_body_scoped():
    a = gateway._dedup_key("choice", {"state": "s", "instructions": "i", "criteria": {"x": "1", "y": "2"}})
    b = gateway._dedup_key("choice", {"state": "s", "instructions": "i", "criteria": {"y": "2", "x": "1"}})
    c = gateway._dedup_key("choice", {"state": "s2", "instructions": "i", "criteria": {"x": "1", "y": "2"}})
    assert a == b  # key order does not matter
    assert a != c  # different state -> different key
```

Note: `test_identical_concurrent_requests_share_one_upstream_call` relies on the rate limiter allowing 3 identical `choice` calls. `CHOICE` has 2 candidates so `_weight` is 1; with `CAPACITY=12` three calls fit. If the shared rate limiter interferes, clear `gateway.rate_hits` between assertions as shown.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_web_gateway.py::test_dedup_key_is_stable_and_body_scoped -v`
Expected: FAIL — `AttributeError: module 'web.server' has no attribute '_dedup_key'`.

- [ ] **Step 3: Write minimal implementation**

In `web/server.py`, add imports at the top (near `import threading`, line 12):

```python
import concurrent.futures
import hashlib
```

Add module state after `inflight = threading.BoundedSemaphore(MAX_INFLIGHT)` (line 82):

```python
dedup_lock = threading.Lock()
dedup_map: dict[str, concurrent.futures.Future] = {}
DEDUP_MAX = 256


def _dedup_key(primitive: str, body: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"primitive": primitive, "body": body},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Refactor the upstream call in `do_POST` so the leader performs it and waiters reuse the raw payload. Replace the `try: raw = ...; upstream = _upstream_request(...); result = validate_response(...)` block (inside the inflight section from Task 2) with dedup-aware logic. Insert BEFORE the `wait_start = time.monotonic()` line:

```python
            dedup = _dedup_key(cast(str, primitive), body)
            with dedup_lock:
                pending = dedup_map.get(dedup)
                leader = pending is None
                if leader and len(dedup_map) < DEDUP_MAX:
                    pending = concurrent.futures.Future()
                    dedup_map[dedup] = pending
                elif leader:
                    pending = None  # map full; fall through without coalescing
```

Then wrap the upstream request so only the leader calls upstream and publishes the raw payload (or exception) to the future; waiters read it:

```python
            wait_start = time.monotonic()
            if not inflight.acquire(blocking=True, timeout=QUEUE_TIMEOUT_S):
                status, error_code = 503, "model_busy"
                self.send(
                    status,
                    {"code": error_code, "detail": "Model is busy. Try again shortly.",
                     "retry_after": 2, "request_id": request_id},
                    2,
                )
                return
            queue_wait_ms = round((time.monotonic() - wait_start) * 1000)
            try:
                if leader:
                    raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    upstream_start = time.monotonic()
                    try:
                        upstream = _upstream_request("/v1/" + primitive, raw, timeout=INFER_TIMEOUT_S)
                        upstream_latency_ms = round((time.monotonic() - upstream_start) * 1000)
                        if pending is not None:
                            pending.set_result(upstream)
                    except BaseException as exc:  # publish failure to waiters, then re-raise
                        if pending is not None:
                            pending.set_exception(exc)
                        raise
                else:
                    upstream_start = time.monotonic()
                    upstream = pending.result(timeout=INFER_TIMEOUT_S + QUEUE_TIMEOUT_S + 1)
                    upstream_latency_ms = round((time.monotonic() - upstream_start) * 1000)
                result = validate_response(cast(Primitive, primitive), upstream, body)
            finally:
                inflight.release()
                if leader and pending is not None:
                    with dedup_lock:
                        dedup_map.pop(dedup, None)
```

The remainder of `do_POST` (version merge, `model`/`request_id`/`gateway_latency_ms`, `enrich(...)`) is unchanged from Task 2 — each caller runs it independently, so waiters get their own `request_id` and enrichment. Because a waiter re-raises the leader's exception via `pending.result()`, the existing `except HTTPError / URLError / ...` branches map it identically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_web_gateway.py -v`
Expected: PASS (dedup tests plus all prior gateway tests — non-concurrent or non-identical requests take the leader path exactly as before).

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add web/server.py tests/test_web_gateway.py
git commit -m "feat(web): coalesce identical concurrent infer requests into one upstream call"
```

---

## Task 6: Experimental dynamic batcher core (`veya/experimental/batch_server.py`)

**Files:**
- Create: `veya/experimental/batch_server.py`
- Test: `tests/test_experimental_batch_server.py`

**Interfaces:**
- Consumes (from stable core, imported lazily/never modifying): `veya.agent.Agent`, `veya.common.build_sequence`, `veya.common.collate_items`, `veya.common.QTYPES`, `veya.common.render_options`, `veya.common.confidence_from_probs`, `veya.common.temp_bucket`.
- Produces:
  - `class DynamicBatcher:` with `__init__(self, agent, max_batch=16, max_wait_ms=10)`, `submit(self, state, questions) -> dict` (blocking; returns the same shape as `Agent.system_one`), `start()`, `stop()`.
  - `RESEARCH_ONLY = True` module constant and a `research_banner()` string.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experimental_batch_server.py`:

```python
"""Parity + failure tests for the research-only experimental dynamic batcher.

Skipped when no checkpoint is reachable; the batcher must produce output that
matches serial Agent.system_one within tolerance for the SAME loaded agent.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from veya.experimental.batch_server import DynamicBatcher, RESEARCH_ONLY  # noqa: E402

MODEL_DIR = os.environ.get("VEYA_TEST_MODEL_DIR", "models/production")


def _agent():
    import veya
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("no local checkpoint for batcher parity test")
    return veya.load(MODEL_DIR, device="cpu")


def test_research_only_flag():
    assert RESEARCH_ONLY is True


def test_batched_matches_serial_system_one():
    agent = _agent()
    requests = [
        {"state": "Duplicate charge, please refund.", "questions": {"q": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs"}}}},
        {"state": "The app crashes on login.", "questions": {"q": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs"}}}},
        {"state": "Is this phishing?", "questions": {"q": {"type": "noul", "instructions": "Is this phishing?"}}},
    ]
    serial = [agent.system_one(r["state"], r["questions"]) for r in requests]
    batcher = DynamicBatcher(agent, max_batch=16, max_wait_ms=10)
    batcher.start()
    try:
        batched = [batcher.submit(r["state"], r["questions"]) for r in requests]
    finally:
        batcher.stop()
    for s, b in zip(serial, batched):
        assert s["decision"] == b["decision"]
        for qid, sa in s["answers"].items():
            ba = b["answers"][qid]
            assert sa["type"] == ba["type"]
            if sa["type"] == "choice":
                assert sa["choice"] == ba["choice"]
                for label, p in sa["probabilities"].items():
                    assert abs(p - ba["probabilities"][label]) < 1e-4
            elif sa["type"] == "noul":
                assert abs(sa["noul"] - ba["noul"]) < 1e-4


def test_batch_failure_propagates_to_all_waiters():
    agent = _agent()
    batcher = DynamicBatcher(agent, max_batch=16, max_wait_ms=5)

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic forward failure")

    agent.model = boom  # force the joint forward to fail
    batcher.start()
    try:
        with pytest.raises(RuntimeError):
            batcher.submit("text", {"q": {"type": "noul", "instructions": "Yes?"}})
    finally:
        batcher.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_experimental_batch_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'veya.experimental.batch_server'` (or skip if no local checkpoint, but the import error comes first).

- [ ] **Step 3: Write minimal implementation**

Create `veya/experimental/batch_server.py`:

```python
"""RESEARCH ONLY: cross-state dynamic batching for a loaded Veya agent.

This module is NOT part of any release manifest and is never imported by the
audited load path (veya/__init__.py, veya/server.py, veya/release_bundle.py, or
web/). It coalesces concurrent requests with DIFFERENT states into a single
padded encoder forward, which the single-state /v1/systemone contract cannot
express. It reuses the stable core (build_sequence, collate_items) and does not
modify the loaded Agent. Output is required to match serial Agent.system_one
within numerical tolerance (see tests/test_experimental_batch_server.py).
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from ..common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    state_has_content,
    temp_bucket,
)

RESEARCH_ONLY = True
MAX_BATCH = 16
MAX_WAIT_MS = 10


def research_banner() -> str:
    return "veya.experimental.batch_server: RESEARCH ONLY, not part of any release manifest."


class _Job:
    __slots__ = ("state", "questions", "event", "result", "error")

    def __init__(self, state: Any, questions: dict):
        self.state = state
        self.questions = questions
        self.event = threading.Event()
        self.result: dict | None = None
        self.error: BaseException | None = None


class DynamicBatcher:
    """Drain a queue of single-request jobs into padded joint forwards."""

    def __init__(self, agent, max_batch: int = MAX_BATCH, max_wait_ms: int = MAX_WAIT_MS):
        self.agent = agent
        self.max_batch = int(max_batch)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self._queue: list[_Job] = []
        self._cond = threading.Condition()
        self._stop = False
        self._worker: threading.Thread | None = None
        self._forward_lock = threading.Lock()

    def start(self) -> None:
        with self._cond:
            self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

    def submit(self, state: Any, questions: dict) -> dict:
        job = _Job(state, questions)
        with self._cond:
            self._queue.append(job)
            self._cond.notify_all()
        job.event.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def _drain(self) -> list[_Job]:
        with self._cond:
            while not self._queue and not self._stop:
                self._cond.wait(timeout=0.5)
            if self._stop and not self._queue:
                return []
            batch = self._queue[: self.max_batch]
            self._queue = self._queue[len(batch):]
            return batch

    def _run(self) -> None:
        while True:
            batch = self._drain()
            if not batch:
                if self._stop:
                    return
                continue
            # Small wait window to let concurrent submissions join the batch.
            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch and time.monotonic() < deadline:
                with self._cond:
                    if self._queue:
                        take = self._queue[: self.max_batch - len(batch)]
                        self._queue = self._queue[len(take):]
                        batch.extend(take)
                    else:
                        self._cond.wait(timeout=max(0.0, deadline - time.monotonic()))
            try:
                results = self._run_batch(batch)
            except BaseException as exc:  # propagate to every waiter in the batch
                for job in batch:
                    job.error = exc
                    job.event.set()
                continue
            for job, result in zip(batch, results):
                job.result = result
                job.event.set()

    # --- batch execution -------------------------------------------------

    def _encode(self, job: _Job) -> tuple[list[str], list[dict], list[dict]]:
        agent = self.agent
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        ids = list(job.questions.keys())
        items = []
        for qid in ids:
            q = agent._to_internal(job.questions[qid])
            seq, markers = build_sequence(agent.tok, job.state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "_qid": qid, "_q": q})
        return ids, items, []

    def _run_batch(self, batch: list[_Job]) -> list[dict]:
        import torch

        agent = self.agent
        # Defer/invalid schema short-circuits mirror Agent.system_one exactly.
        prepared = []
        results: list[dict | None] = [None] * len(batch)
        flat_items: list[dict] = []
        spans: list[tuple[int, int]] = []
        for index, job in enumerate(batch):
            from ..lang import defer_reason
            if not isinstance(job.questions, dict) or not job.questions or not state_has_content(job.state):
                results[index] = {
                    "model": "veya-rl-agent", "decision": "defer", "reason": "invalid_schema",
                    "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                spans.append((0, 0))
                continue
            blocked = defer_reason(agent.cfg, job.state)
            if blocked:
                results[index] = {
                    "model": "veya-rl-agent", "decision": "defer", "reason": blocked,
                    "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                spans.append((0, 0))
                continue
            _ids, items, _ = self._encode(job)
            start = len(flat_items)
            flat_items.extend(items)
            spans.append((start, len(flat_items)))
            prepared.append(index)

        if flat_items:
            b = collate_items([flat_items], agent.tok.pad_token_id)
            use_amp = agent.device.type == "cuda"
            with self._forward_lock, torch.no_grad(), torch.autocast(
                device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp
            ):
                logits, _act = agent.model(
                    b["input_ids"].to(agent.device),
                    b["attention_mask"].to(agent.device),
                    b["marker_pos"].to(agent.device),
                    b["marker_mask"].to(agent.device),
                    b["qtype"].to(agent.device),
                )
            logits = logits.float().cpu().numpy()
            n_tokens_total = int(b["attention_mask"].sum())
            for index in prepared:
                start, end = spans[index]
                results[index] = self._decode(batch[index], flat_items[start:end], logits[start:end], n_tokens_total)

        return [r for r in results if r is not None]

    def _decode(self, job: _Job, items: list[dict], rows: np.ndarray, n_tokens: int) -> dict:
        agent = self.agent
        answers: dict[str, Any] = {}
        for offset, qid in enumerate(job.questions.keys()):
            q = items[offset]["_q"]
            k = len(items[offset]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = rows[offset, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf,
                }
            else:
                answers[qid] = {
                    "type": "noul", "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                }
        return {
            "model": "veya-rl-agent", "decision": "answer", "reason": None,
            "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_experimental_batch_server.py -v`
Expected: PASS when a local checkpoint exists at `models/production` (skips otherwise). Parity tolerance is 1e-4; padding is masked by `attention_mask`, so rows must not leak.

- [ ] **Step 5: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add veya/experimental/batch_server.py tests/test_experimental_batch_server.py
git commit -m "feat(experimental): research-only cross-state dynamic batcher with serial parity"
```

---

## Task 7: Batch server HTTP wrapper + optional CPU acceleration

**Files:**
- Modify: `veya/experimental/batch_server.py`
- Test: `tests/test_experimental_batch_server.py`

**Interfaces:**
- Consumes: `DynamicBatcher` (Task 6).
- Produces:
  - `make_handler(batcher) -> type` — stdlib `BaseHTTPRequestHandler` exposing `POST /v1/systemone`, `GET /health`, `GET /version` (version includes a `research_only: true` banner).
  - `serve(agent, host="127.0.0.1", port=8090, quantize=None) -> ThreadingHTTPServer`
  - `quantize_agent(agent, mode="int8") -> None` — applies `torch.quantization.quantize_dynamic` to a COPY of the encoder's Linear layers in place on `agent.model`; no files written.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experimental_batch_server.py`:

```python
def test_batch_server_http_roundtrip_and_research_banner():
    import json
    import threading
    from urllib.request import Request, urlopen
    from veya.experimental.batch_server import serve

    agent = _agent()
    server = serve(agent, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/version", timeout=5) as resp:
            version = json.load(resp)
        assert version["research_only"] is True
        body = json.dumps({
            "state": "Duplicate charge.",
            "questions": {"q": {"type": "choice", "instructions": "Team?",
                                  "criteria": {"billing": "Pay", "technical": "Bugs"}}},
        }).encode()
        req = Request(f"http://127.0.0.1:{port}/v1/systemone", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=10) as resp:
            out = json.load(resp)
        assert out["decision"] == "answer" and "q" in out["answers"]
    finally:
        server.shutdown()
        server.server_close()


def test_int8_quantization_parity_within_tolerance():
    agent = _agent()
    from veya.experimental.batch_server import quantize_agent
    ref = agent.system_one("Duplicate charge.", {"q": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Pay", "technical": "Bugs"}}})
    quantize_agent(agent, "int8")
    out = agent.system_one("Duplicate charge.", {"q": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Pay", "technical": "Bugs"}}})
    assert out["decision"] == ref["decision"]
    # Quantization changes logits slightly; the argmax should usually hold, but we
    # only require the distribution to stay valid and within a loose tolerance.
    for label, p in ref["answers"]["q"]["probabilities"].items():
        assert abs(p - out["answers"]["q"]["probabilities"][label]) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_experimental_batch_server.py::test_batch_server_http_roundtrip_and_research_banner -v`
Expected: FAIL — `ImportError: cannot import name 'serve'`.

- [ ] **Step 3: Write minimal implementation**

Append to `veya/experimental/batch_server.py`:

```python
def quantize_agent(agent, mode: str = "int8") -> None:
    """Apply dynamic int8 quantization to the encoder's Linear layers in place.

    No weights are written to disk; the frozen checkpoint files are untouched.
    Only the in-memory module is quantized, for local CPU throughput research.
    """
    import torch

    if mode != "int8":
        raise ValueError("only 'int8' dynamic quantization is supported")
    agent.model.eval().to("cpu")
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.model.encoder = torch.quantization.quantize_dynamic(
        agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8
    )


def make_handler(batcher: "DynamicBatcher"):
    import json
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/version":
                self._send(200, {"research_only": True, "banner": research_banner()})
            else:
                self._send(404, {"decision": "invalid", "defer_reason": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/systemone":
                self._send(404, {"decision": "invalid", "defer_reason": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                result = batcher.submit(body.get("state"), body.get("questions") or {})
                self._send(200, result)
            except Exception as exc:  # research server: surface the error, never hang
                self._send(400, {"decision": "invalid", "defer_reason": "error", "detail": str(exc)})

    return Handler


def serve(agent, host: str = "127.0.0.1", port: int = 8090, quantize: str | None = None):
    from http.server import ThreadingHTTPServer

    if quantize:
        quantize_agent(agent, quantize)
    batcher = DynamicBatcher(agent)
    batcher.start()
    server = ThreadingHTTPServer((host, port), make_handler(batcher))
    server.veya_batcher = batcher  # keep a reference so it is not garbage collected
    return server
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_experimental_batch_server.py -v`
Expected: PASS (or skip when no local checkpoint). If `quantize_dynamic` is unsupported on the platform's torch build, the quantization test should be marked `pytest.importorskip`-style guarded; wrap the `quantize_agent` call in a `try/except (RuntimeError, ValueError)` and `pytest.skip` on failure.

- [ ] **Step 5: Verify the experimental module is not on the audited import path**

Run: `cd "/Volumes/Parth Patel/Veya" && python -c "import sys, veya, veya.server, veya.release_bundle, web.server; assert 'veya.experimental.batch_server' not in sys.modules, 'batch_server leaked into audited path'; print('ok')"`
Expected: `ok` — importing the audited surface never pulls in the experimental batcher.

- [ ] **Step 6: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add veya/experimental/batch_server.py tests/test_experimental_batch_server.py
git commit -m "feat(experimental): batch-server HTTP wrapper and optional int8 CPU quantization"
```

---

## Task 8: Manifest-hash regression guard

**Files:**
- Modify: `tests/test_release_bundle.py`

**Interfaces:**
- Consumes: `artifacts/release/manifest.json` `runtime_files` map.
- Produces: a test asserting every pinned `veya/*.py` file still matches its recorded SHA-256 — proving Tasks 1-7 left the audited bundle untouched.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_release_bundle.py` (it uses the `check()`/PASS/FAIL pattern and `hashlib` is not yet imported there):

```python
def test_release_manifest_runtime_hashes_still_verify():
    import hashlib
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "artifacts", "release", "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    for relative, expected in manifest["runtime_files"].items():
        path = os.path.join(root, relative)
        with open(path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        check("manifest/%s hash" % relative, actual == expected, actual)
```

Register it in the `__main__` block by adding `test_release_manifest_runtime_hashes_still_verify()` before `print("%d passed...")`, and rely on the existing `test_recorded_checks_pass` assertion to fail the suite if any hash diverges.

- [ ] **Step 2: Run test to verify it passes**

Run: `cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_release_bundle.py -v`
Expected: PASS — no pinned file was modified by Tasks 1-7, so every recorded hash still verifies. If any hash FAILS, a pinned file was edited by mistake: revert that edit (this is the guard doing its job).

- [ ] **Step 3: Commit**

```bash
cd "/Volumes/Parth Patel/Veya"
git add tests/test_release_bundle.py
git commit -m "test: guard that the audited release manifest runtime hashes still verify"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the affected Python suites**

Run:
```bash
cd "/Volumes/Parth Patel/Veya" && python -m pytest tests/test_gateway_enrich.py tests/test_web_gateway.py tests/test_release_bundle.py tests/test_experimental_batch_server.py -v
```
Expected: all PASS (batcher/quantization tests skip if no local checkpoint).

- [ ] **Step 2: Run the Node client tests**

Run: `cd "/Volumes/Parth Patel/Veya" && node --test tests/test_site_content.mjs tests/test_model_client.mjs`
Expected: PASS.

- [ ] **Step 3: Confirm no pinned file changed across the whole branch**

Run:
```bash
cd "/Volumes/Parth Patel/Veya" && git diff --name-only d52c8ee4..HEAD -- 'veya/*.py' artifacts/release/manifest.json
```
Expected: EMPTY output — no `veya/*.py` runtime file and no manifest was modified relative to the audited baseline commit. (`veya/experimental/batch_server.py` is a NEW file and does not match `veya/*.py` at the top level; if it appears, the glob is matching recursively — restrict with `git diff --name-only d52c8ee4..HEAD -- 'veya/*.py'` and confirm the only experimental path is under `veya/experimental/`, which is not hash-pinned.)

- [ ] **Step 4: Lint / type check**

Run: `cd "/Volumes/Parth Patel/Veya" && (ruff check web/ veya/experimental/batch_server.py tests/ 2>/dev/null || true) && (mypy web/enrich.py 2>/dev/null || true)`
Expected: no new errors in the files touched by this plan.

---

## Self-Review Notes

- **Spec coverage:** Part A → Tasks 1-3 (top_k §3.1, band §3.2, latency §3.3, provenance §3.4, client §3.5). Part B → Tasks 4-5 (queue §4.1, dedup §4.2). Part C → Tasks 6-7 (batcher §5.1, quantization §5.2, guardrails §5.3). Audit guard §8 → Task 8. Testing §8 → Tasks 1-8 test steps + Task 9. All spec sections mapped.
- **Type consistency:** `top_k`/`prob_band`/`enrich` signatures match between Task 1 (definition) and Task 2 (call). `_dedup_key(primitive, body)` matches between Task 5 definition and test. `DynamicBatcher.submit(state, questions)` matches between Task 6 and Task 7's handler. `quantize_agent(agent, mode)` matches Task 7 test and definition.
- **Placeholders:** none — all code steps contain full implementations; the two "match the existing file's conventions" notes (Task 3 Step 1, Task 7 Step 4) point at real files to read and give concrete fallback code.
- **Constraint check:** no task edits a hash-pinned `veya/*.py` file or the manifest; Task 8 and Task 9 Step 3 both enforce this.
