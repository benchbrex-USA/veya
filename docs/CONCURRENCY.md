# Veya Concurrency Model

This document describes the concurrency and overload model as implemented in
the code (verified against `web/server.py`, `web/model_server.py`, and
`veya/server.py`). It is a description of the shipped system, not a proposal.

## Architecture

Two processes, one HTTP hop between them:

- **Web Gateway** (`web/server.py`): a stdlib `http.server.ThreadingHTTPServer`
  (thread per connection, HTTP/1.1 keep-alive). Serves the website, the
  operational probes (`/healthz`, `/readyz`, `/startupz`), and the single
  inference endpoint `POST /api/infer`.
- **Model Service** (`veya/server.py`, started by `web/model_server.py`): a
  separate process that owns the loaded release bundle. Also a
  `ThreadingHTTPServer` exposing `/health`, `/version`, and `/v1/<primitive>`.

The gateway never loads weights. Every inference is a real HTTP hop to the
model service (`_upstream_request` via `urllib`), so a gateway crash cannot
corrupt model state, and the model process can hold the audited-release
integrity boundary (`load_release_manifest` hash verification) in isolation.

## Request Processing

1. Client → gateway `POST /api/infer`.
2. Gateway validates (size ≤ 32 KB, Content-Type, same-origin, per-IP weighted
   rate limit of `CAPACITY=12` tokens per 60 s window) *before* admitting work.
3. Gateway acquires the `inflight` `threading.BoundedSemaphore(MAX_INFLIGHT=2)`
   with `blocking=False`. If the semaphore is exhausted → immediate
   `503 model_busy` with `Retry-After: 5` (no queueing).
4. Gateway forwards the JSON body to `UPSTREAM/v1/<primitive>` with
   `INFER_TIMEOUT_S` (default 30 s).
5. Model service validates, then runs the forward pass under `_INFER_LOCK`
   (serialized), returns the result.
6. Gateway validates the upstream response (`validate_response`), releases the
   semaphore in a `finally` block, enriches with version metadata, and replies.

## Serialization

- `_INFER_LOCK` (`veya/server.py`, module-level `threading.Lock`): every
  forward path — `agent.system_one`, `bundle.public_choice`,
  `bundle.answer_questions`, hypothesis scoring — executes inside
  `with _INFER_LOCK`.
- **Purpose**: the checkpoint is a single `nn.Module`; concurrent forwards are
  not a supported serving mode. The lock guarantees only one thread touches
  the model at a time, which also protects any question-conditioned state
  re-encoding from interleaving.
- **Scope**: model-service process-wide (all inference endpoints share one
  lock). GET endpoints (`/health`, `/version`) do not take the lock.
- The gateway has its own admission control (`inflight` semaphore) so that
  serialization at the model service never becomes an unbounded wait: at most
  2 requests can be in the upstream hop at once.

## Concurrency Limits

- `MAX_INFLIGHT = 2` (`web/server.py`), enforced by
  `inflight = threading.BoundedSemaphore(MAX_INFLIGHT)`, acquired
  non-blocking just before the upstream inference call and released in
  `finally`.
- Behavior at limit: `503` with `code: "model_busy"`,
  `error_code: "OVERLOADED"`, `retry_after: 5` in the body, and a
  `Retry-After: 5` header.
- **Rationale**: inference is CPU-bound against a single ~2 GB model; more
  concurrent forwards would not add throughput (they queue behind
  `_INFER_LOCK` anyway) and would only add memory pressure and latency
  variance. Refusing fast with `Retry-After` keeps clients well-behaved and
  keeps p99 honest for admitted requests.

## Overload Protection

- **Load shedding** (above): at `MAX_INFLIGHT` the gateway rejects rather than
  queues. Nothing waits on the semaphore; the hop to the model service only
  ever has ≤ 2 concurrent connections.
- **Rate limiting**: per-client-IP sliding 60 s window, capacity 12 weighted
  units (`choice` cost grows with candidate count via `_weight`). Exceeded →
  `429 rate_limited` with `Retry-After`.
- **Graceful shutdown** (`_graceful_shutdown`, G12): on `SIGTERM`/`SIGINT` the
  gateway sets `_SHUTTING_DOWN = True` and arms a hard `os._exit(0)` after
  `SHUTDOWN_DRAIN_S = 30`. While draining:
  - `/healthz` still answers `200` (liveness is process-level).
  - All other routes, including `POST /api/infer`, get
    `503 shutting_down` (`error_code: "SHUTTING_DOWN"`) with
    `Retry-After: 30` — new work is diverted to other replicas immediately.
  - Requests already past admission continue; they have 30 s to finish before
    the hard exit.

## Thread Safety

Gateway shared state (thread-per-request server, so all of this is genuinely
concurrent):

- `rate_hits` — mutable per-IP deques, guarded by `rate_lock`.
- `version_cache` — tuple swap guarded by `version_lock` (cache of
  `/version` metadata, 60 s TTL).
- `_READINESS` / `_READINESS_PROBE` / `_RELEASE_BINDINGS` — guarded by
  `_READINESS_LOCK`; component states latch monotonically, so probes cannot
  flap a settled state.
- `inflight` — `BoundedSemaphore`, safe.
- `_SHUTTING_DOWN` — a bool set by the signal handler; read without a lock,
  which is fine because it only ever transitions `False → True` (GIL-atomic
  flip, one-way).
- Tokenizer / model weights: never touched by the gateway at all. In the model
  service, weights are loaded once at startup and are read-only afterwards;
  all access is serialized by `_INFER_LOCK`.

## Performance Characteristics

Reference: `artifacts/release/runtime_audit.json` (release
`0.3.5-audit-baseline.1`, macOS arm64, CPU):

- Single small packed choice: p50 ≈ 105 ms, p99 ≈ 110 ms, ≈ 9.5 req/s.
- High-n choice (77 candidates): p50 ≈ 163 ms, ≈ 6.1 req/s.
- noul: p50 ≈ 76 ms; score categorical similar.
- Cold start (manifest verify + checkpoint load): ≈ 34 s.
- Memory: ≈ 2030 MB RSS after benchmark.

With `MAX_INFLIGHT=2` and ~105 ms per small request, per-replica peak
throughput is ~2 × 9.5 ≈ 19 req/s in the best case (fully parallel upstream
hops); in practice admitted requests serialize on `_INFER_LOCK`, so sustained
throughput approaches the single-request number while latency stays flat.

## Design Rationale

Serialization is intentional, not an accident of scope:

1. **CPU inference with shared model state.** One checkpoint, one process;
   PyTorch forwards against a single module are not thread-safe in serving
   mode, and question-conditioned attention makes hidden-state reuse unsafe
   (state is re-encoded per question).
2. **Memory.** The model is ~2 GB RSS; concurrency buys no capacity, only
   duplicate activation memory.
3. **Correctness over throughput.** Veya is a decision engine — a garbled or
   interleaved inference is worse than a 503. Locks + load shedding mean every
   200 response was produced by exactly one serialized forward pass.
4. **Honest latency.** Rejecting the 3rd request is cheaper and clearer than
   admitting it and tripling everyone's p99.

## Scaling Strategy

- **Horizontal, not vertical.** Run more gateway+model replica pairs
  (e.g. K8s HPA on CPU utilization or request rate). Each replica serves up to
  `MAX_INFLIGHT=2` concurrent inferences and sheds the rest with
  `Retry-After: 5`, which pairs naturally with client retry/backoff.
- Total system throughput = replicas × per-replica throughput (~9.5–19 req/s
  per replica depending on workload mix).
- Graceful shutdown (`Retry-After` on drain + 30 s drain window) makes rolling
  deploys and scale-in safe: the load balancer's retry lands on another
  replica, in-flight work completes.
- Do not raise `MAX_INFLIGHT` to "fix" throughput — beyond 2, requests would
  queue on `_INFER_LOCK` and only p99 would grow. Scale out instead.
