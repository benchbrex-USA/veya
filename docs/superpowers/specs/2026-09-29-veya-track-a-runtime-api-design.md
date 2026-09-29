# Veya Decision Engine — Track A: Runtime & API Advancement

**Date:** 2026-09-29
**Track:** A (runtime/API) of a four-track engine-advancement roadmap
**Hard constraint:** The frozen release manifest and audit MUST stay valid. No hash-pinned
`veya/*.py` runtime file is modified. No retraining. No re-audit.

## 1. Summary

Advance the Veya decision engine's runtime and developer-facing API without touching the
audited release bundle. Work splits into three parts:

- **Part A — Gateway richer responses** (audited production path, `web/`): derive top-k
  options, a statistical confidence band, split latency metrics, and a consolidated
  `provenance` object — all computed gateway-side from data the upstream already returns.
- **Part B — Gateway concurrency grace** (audited production path, `web/`): replace the
  hard `503 model_busy` rejection on the 3rd concurrent request with a bounded wait-queue,
  plus safe dedup coalescing of identical in-flight requests.
- **Part C — Experimental accelerated server** (research-only, `veya/experimental/`): a real
  cross-state dynamic batcher and optional CPU acceleration (int8 dynamic quantization /
  ONNX Runtime), operating on the same frozen weights, explicitly NOT wired into the release
  manifest.

Parts A and B ship on the audited path with zero manifest impact. Part C is a parallel,
clearly-labeled research artifact, mirroring how the project already ships hypothesis and
ordinal scoring as "RESEARCH ONLY / not served."

## 2. Binding constraints (verified against code)

1. **`artifacts/release/manifest.json` hash-pins every `veya/*.py` runtime file** (lines
   27–43), including `server.py`, `release_bundle.py`, `agent.py`, `common.py`,
   `selective.py`, `__init__.py`. `load_release_manifest` (`veya/release_bundle.py:455-456`)
   recomputes each SHA-256 and refuses to load on any mismatch. **These files are read-only
   for this work.**
2. **`web/` is NOT in `runtime_files`.** The gateway (`web/server.py`, `web/contracts.py`)
   and static client may be extended freely without affecting the manifest.
3. **`veya/experimental/` is NOT in `runtime_files`.** New modules there do not affect the
   manifest, provided nothing in the audited load path imports them.
4. **`/v1/systemone` accepts a single `state` + a dict of questions** (`agent.py:292`,
   `release_bundle.py:325`). A batched upstream call can therefore only merge requests that
   share the same state. True cross-state batching requires the model-server forward pass,
   which is hash-pinned — so cross-state batching lives only in Part C.
5. **The upstream serializes inference** on a process-wide `_INFER_LOCK` (`veya/server.py:16`),
   and the gateway caps concurrency at `MAX_INFLIGHT=2` (`web/server.py:79`).
6. **State-embedding caching is unsafe**: manifest `known_limitations` records that
   "SystemOne re-encodes state per question because question-conditioned attention makes
   hidden-state reuse unsafe." No hidden-state reuse across questions is introduced.
7. **Additive response changes only**: the public response envelope may gain fields but must
   not remove or rename existing ones, so the static client and existing gateway tests keep
   passing.

## 3. Part A — Gateway richer responses

**Files:** `web/contracts.py`, `web/server.py`. **Manifest impact:** none.

All new fields are computed in the gateway from the already-validated upstream response. No
new upstream data is requested; the upstream contract is unchanged.

### 3.1 top-k options
- For `choice` and `score` answers (including each `systemone` sub-answer), derive
  `top_k`: a list of `{"label", "probability"}` sorted by probability descending, truncated
  to `k = min(5, n_options)`.
- Source is the validated `probabilities` map already produced by `_answer`
  (`web/contracts.py:219`). Computed after validation so only whitelisted labels are used.
- `noul` is excluded (binary; the band in 3.2 is the meaningful signal).

### 3.2 confidence band
- Add a Wilson-score interval around the reported top probability:
  `prob_low`, `prob_high`, computed at z = 1.96 (95%).
- For `choice`/`score`: band around the top-1 probability. For `noul`: band around P(true).
- Reuse the Wilson math already in `veya/selective.wilson_interval` conceptually, but
  **reimplement locally in `web/`** (the gateway must not import hash-pinned `veya` modules,
  to keep the dependency direction one-way and the manifest untouched). The formula:
  treat the top probability as a rate over an effective sample size `n_eff`, where `n_eff`
  is a documented gateway constant (default 100) representing calibration-set support.
- The band is explicitly labeled in docs/response metadata as an *indicative* interval derived
  from the reported probability, not a frequentist guarantee over new samples. This avoids
  overstating the audited automation guarantees (automation remains DISABLED per manifest).

### 3.3 latency metrics
- Split the existing single `gateway_latency_ms` (`web/server.py:538`) into:
  - `upstream_latency_ms`: measured tightly around the `_upstream_request` call.
  - `queue_wait_ms`: time spent waiting for an inflight slot (Part B) — 0 when not queued.
  - `gateway_latency_ms`: total, retained for backward compatibility.

### 3.4 provenance object
- Consolidate already-whitelisted metadata into a nested `provenance` object:
  `{"calibration_version", "calibration", "route", "reranker_id", "model_version",
  "release", "architecture_version"}`.
- Existing top-level copies of these fields are RETAINED (constraint 7) to avoid breaking
  the static client; `provenance` is additive. A follow-up (out of scope here) can deprecate
  the top-level duplicates once the client reads `provenance`.

### 3.5 static client
- Extend `web/static/model-client.mjs` / playground rendering to display top-k and the
  confidence band when present. Feature-detect (render only if the field exists) so an older
  gateway response still renders.

## 4. Part B — Gateway concurrency grace

**Files:** `web/server.py`. **Manifest impact:** none.

### 4.1 Bounded wait-queue
- Replace `inflight.acquire(blocking=False)` → instant `503 model_busy`
  (`web/server.py:496`) with `inflight.acquire(blocking=True, timeout=QUEUE_TIMEOUT_S)`.
- `QUEUE_TIMEOUT_S` default 2.0s (module constant, env-overridable via
  `VEYA_QUEUE_TIMEOUT_S`). On timeout, retain the existing `503 model_busy` + `Retry-After`.
- `queue_wait_ms` (Part A 3.3) records the blocked duration.
- `MAX_INFLIGHT` stays at 2 (the upstream serializes anyway; raising it only deepens the
  queue without adding throughput).

### 4.2 Identical-request dedup coalescing
- Maintain an in-flight map keyed by a canonical hash of the validated request body
  (`primitive` + `state` + `instructions`/`criteria`/`questions`, canonicalized JSON).
- When an identical request is already in flight, the second caller awaits the first's
  result future instead of issuing a duplicate upstream call; the response is fanned out to
  all waiters (each still gets its own `request_id`).
- Safe because identical validated inputs to a deterministic model yield identical outputs.
  Each waiter independently runs `validate_response` on the shared upstream payload.
- Guarded by a dedicated lock; entries are removed when the originating call settles. Map
  size is bounded (evict on completion; hard cap with fall-through to a normal call).
- Dedup is best-effort: any race that misses simply issues a normal request.

### 4.3 Rate limiting unchanged
- `_charge` / `CAPACITY` / `WINDOW_S` per-IP rate limiting is untouched. Dedup and queueing
  operate after rate-limit acceptance.

## 5. Part C — Experimental accelerated server (research-only)

**Files (new):** `veya/experimental/batch_server.py`, `tests/test_experimental_batch_server.py`.
**Manifest impact:** none (not referenced; not imported by the audited load path).

### 5.1 Dynamic batcher
- A stdlib `ThreadingHTTPServer` exposing the same `/v1/*` surface as `veya/server.py` for
  drop-in local comparison, plus a background batch worker.
- Requests enqueue `(state, question)` items with a `concurrent.futures.Future`. The worker
  drains up to `MAX_BATCH` (default 16) items or until `MAX_WAIT_MS` (default 10) elapses,
  builds ONE padded forward across *different* states via `common.collate_items` (already
  supports variable-length `ids`/`markers` with padding), runs a single `model(...)` call
  under one lock, then scatters per-row logits back to each future.
- **Correctness parity requirement:** batched output for a set of requests MUST match the
  serial `Agent.system_one` output per request within a tight numerical tolerance (padding
  must not leak across rows — attention is masked by `attention_mask`). This parity is the
  primary test.
- Reuses the loaded `Agent`/`ReleaseBundle`; does not modify them.

### 5.2 Optional CPU acceleration (flag-gated)
- `--quantize int8`: `torch.quantization.quantize_dynamic` on the encoder Linear layers,
  applied to a copy of the frozen weights at load. No retraining, no weight files written.
- `--onnx`: export the encoder+head to an in-memory/temp ONNX graph and run via ONNX Runtime
  when available; fall back to torch when the dependency is absent.
- Both are opt-in, both must report accuracy/latency parity vs. the fp32 path within a
  documented tolerance, and both are disabled by default.

### 5.3 Guardrails
- Module docstring and a `RESEARCH ONLY` banner in `/version` output state it is unaudited
  and not part of any release manifest.
- No import of this module from `veya/__init__.py`, `veya/server.py`, `veya/release_bundle.py`,
  or `web/`. Dependency direction stays inward (experimental → stable core).
- Heavy deps (`onnx`, `onnxruntime`) imported lazily inside functions, per project convention.

## 6. Data flow

```
Browser (model-client.mjs)
   | POST /api/infer  {primitive, state, instructions, criteria|questions}
   v
web/server.py Handler.do_POST
   |- rate limit (_charge)               [unchanged]
   |- validate (contracts.validate)      [unchanged]
   |- dedup lookup (Part B 4.2) ---------+--> shared upstream call if identical in-flight
   |- inflight acquire (blocking, Part B 4.1)
   |- _upstream_request /v1/<primitive>  [upstream contract unchanged]
   |- validate_response                  [unchanged whitelist]
   |- ENRICH (Part A): top_k, prob band, split latency, provenance
   v
Response envelope (existing fields + additive fields)
```

Part C is a separate process bound to a different port for local benchmarking only; it is
never placed in front of the audited `web/model_server.py` path.

## 7. Error handling

- Part A enrichment is defensive: any derivation failure (e.g. empty probabilities already
  rejected by `validate_response`) omits the derived field rather than failing the request.
  Enrichment never raises into the response path.
- Part B: queue timeout → existing `503 model_busy` + `Retry-After`. Dedup waiter receives the
  originator's outcome; if the originator errored, each waiter maps the same upstream error
  through the existing handler branches. No new error codes.
- Part C: batch worker wraps the forward in try/except; on failure every queued future in the
  batch is set to the exception so callers get a normal `502`/error rather than hanging.
  A worker crash is logged and the worker restarts; in-flight futures fail rather than stall.

## 8. Testing

- `tests/test_web_gateway.py` (extend): assert `top_k` ordering/truncation; `prob_low <=
  probability <= prob_high`; split latency fields present and non-negative; `provenance`
  object populated; existing fields still present (backward compat).
- `tests/test_web_gateway.py` (extend, Part B): 3rd concurrent request queues then succeeds
  within `QUEUE_TIMEOUT_S`; on timeout returns `503 model_busy`; identical concurrent requests
  produce one upstream call and both receive valid, independently-validated responses with
  distinct `request_id`s.
- `tests/test_experimental_batch_server.py` (new, Part C): batched vs. serial `system_one`
  output parity within tolerance across mixed states/primitives; worker failure propagates to
  all queued futures; quantization/ONNX parity within documented tolerance when enabled.
- `tests/test_release_bundle.py` (extend): assert `artifacts/release/manifest.json`
  `runtime_files` hashes STILL verify after Parts A–C — a regression guard proving the audit
  is untouched.
- `tests/test_site_content.mjs` (extend): playground renders top-k and confidence band when
  present and degrades gracefully when absent.

## 9. Out of scope (deferred to other tracks)

- Any edit to hash-pinned `veya/*.py` files (would require Approach 3 re-audit).
- New decision primitives / model-format changes (Track C of the roadmap).
- Learned calibration, conformal prediction, language-conditioned temperature (Track B).
- Deprecating the duplicated top-level provenance fields (separate client-migration change).
- Async (`asyncio`) rewrite of the gateway; the stdlib threaded model is retained.

## 10. Sequencing

1. Part A (richer responses) — highest value, lowest risk, enables B-track metadata surfacing
   later.
2. Part B (concurrency grace) — independent of A; can land in parallel or after.
3. Part C (experimental server) — independent research artifact; land last.

Each part is independently testable and shippable.
