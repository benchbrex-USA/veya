# Veya Production Scope

What the audited release does and does not support. This document is the
human-readable companion to `artifacts/release/production_scope.json` (format
`veya-production-scope-v1`), which is the machine-readable contract; when the
two disagree, the JSON artifact wins.

- Release: `0.3.5-audit-baseline.1` (version 0.3.5)
- Architecture: `veya-shortlist-k8-packed-rerank-v1`
- Manifest: `artifacts/release/manifest.json` (format `veya-release-manifest-v2`)

## Supported Scope

### Languages

English (`en`) only. The served checkpoint is the English release bundle;
`veya/lang.py` rejects non-Latin script and non-English Latin text, and the
model returns a defer decision instead of guessing.

### Decision primitives

| Primitive | Limits | Route |
|---|---|---|
| `choice` | 2–19 candidates packed; 20–100 candidates via the learned top-8 shortlist (gateway cap 100, `MAX_CHOICE` in `web/contracts.py`) | `packed` / `shortlist_k8_packed_rerank` |
| `noul` | state + instructions | packed, checkpoint temperature only |
| `score` | 2–10 ordered labels, categorical only | packed, checkpoint temperature only |
| `systemone` | 1–8 questions per request | per-question routing |

State is capped at 12 KB (`MAX_STATE_BYTES` in `web/contracts.py`) and
instructions at 500 characters. Requests are capped at 32 KB; upstream
responses at 512 KB.

### Selective automation with deferral

Every production decision passes through the selective policy loaded from
`artifacts/release/selective_policy.json`. Automation is currently disabled
(see below), so the model reports a prediction and defers automation rather
than asserting a risk guarantee. See "Selective Automation Status".

### Deployment model

Single audited English release bundle loaded via
`veya.release_bundle.load_release_manifest` with full SHA-256 verification of
every runtime file, the checkpoint, tokenizer, router, shortlist, calibrations,
and the selective policy. Target-domain deployment is the intended use; a
validated selective policy on prospective target-domain data is required
before any automation is enabled.

### Calibration

Per-cardinality temperature buckets for packed choice plus a frozen
hierarchical route temperature (`1.971127355603397`), pinned and hash-verified
in the release manifest (`models/production/calibration.json` and
`models/production/calibration_shortlist_k8.json`). Noul and categorical score
use checkpoint temperatures only. Ordinal score is research-only.

## Unsupported Scope

| Capability | Status | Reason |
|---|---|---|
| Multilingual | Not supported | The multilingual encoder exists (mmBERT, `models/production/multilingual`) but has no production evaluation; the manifest marks `multilingual_choice/noul/score` RESEARCH ONLY. |
| High-candidate retrieval beyond the audited envelope | Not supported | Candidate counts above the gateway cap of 100 are not accepted (`400 invalid_input`, "Choice requires 2–100 candidates."). Counts of 20–100 do route through the K=8 shortlist, but that projection is Banking77-trained and cross-domain retrieval quality is not assumed. |
| Moderation | Not supported | No content moderation capability in this release. |
| Arbitrary schemas | Not supported | Unseen schemas require target-domain evaluation before production; state is capped at 12 KB and instructions at 500 characters. |
| Zero-shot classification | Not supported | Unrestricted zero-shot is not production-scoped; only the four audited primitives above are served. |
| Autonomous decisions without deferral | Not supported | All production decisions require the selective policy with abstention; manifest `automation.status` is `DISABLED`. |
| Phase-9 hypothesis scoring | RESEARCH ONLY | Development-exposed lineage and unresolved injection sensitivity; not loaded by the manifest (see `research_artifacts_not_served.phase9_hypothesis` in `artifacts/release/manifest.json`). |
| Ordinal score | RESEARCH ONLY | `veya/experimental/ordinal.py`; served score is categorical only. |

### Behavior on unsupported input

| Input | Behavior |
|---|---|
| Unsupported language | Defer decision with `defer_reason: "unsupported_language"` (`veya/lang.py` rejects non-Latin script and non-English Latin text on the English checkpoint). |
| Excess candidates | `400 invalid_input` from the gateway ("Choice requires 2–100 candidates."); schema violations inside the model service return `defer_reason: "invalid_schema"`. |
| Missing selective policy | Fail-closed: `load_release_manifest` verifies the policy artifact by SHA-256 and refuses startup if it is missing or mismatched. |
| Hash mismatch (any pinned artifact) | Fail-closed: `load_release_manifest` recomputes every runtime/config/tokenizer/router/shortlist/calibration/policy hash and raises before serving; the server never becomes ready. |

## Limitations

Performance envelope from `artifacts/release/runtime_audit.json` (release
`0.3.5-audit-baseline.1`, macOS arm64, CPU device):

| Metric | Value |
|---|---|
| Small packed choice latency | p50 ≈ 105 ms, p99 ≈ 110 ms, ≈ 9.5 req/s |
| High-n choice (77 candidates) | p50 ≈ 163 ms, ≈ 6.1 req/s |
| Noul | p50 ≈ 76 ms |
| Score (categorical) | p50 ≈ 76 ms |
| SystemOne (3 questions) | p50 ≈ 215 ms |
| Sustained throughput per replica | ≈ 9.5 req/s (peak ≈ 19 req/s with `MAX_INFLIGHT=2` before serialization on `_INFER_LOCK`); scale horizontally, do not raise in-process concurrency |
| Memory | ≈ 2030 MB RSS after benchmark; deployments reserve 4 Gi and limit 6 Gi (`docker-compose.yml`, `k8s/deployment.yaml`) |
| Cold start | ≈ 34 s measured (checkpoint hash ≈ 10.9 s + checkpoint load ≈ 22.8 s); can exceed a minute under CPU load, hence the 120 s startup-probe budget |
| In-process concurrency | `MAX_INFLIGHT = 2` (`web/server.py`); the third concurrent inference gets `503 model_busy` (`error_code: "OVERLOADED"`) with `Retry-After: 5` |
| Rate limit | 12 cost units per 60 s sliding window per client IP; choice cost grows with candidate count |
| Request timeout | `VEYA_REQUEST_TIMEOUT`, default 30 s |

Concurrency and scaling details are in [`docs/CONCURRENCY.md`](CONCURRENCY.md).

## Selective Automation Status

**Automation is disabled.** The policy in
`artifacts/release/selective_policy.json` sets `operating_epsilon:
"disabled"` and `thresholds.disabled: null`, and the manifest records
`automation.status: "DISABLED"`.

- **Policy loaded and enforced**: the policy is hash-pinned by the manifest and
  verified at startup (`veya/release_bundle.py: load_release_manifest` →
  `veya/selective.py: load_policy`, which also binds the policy to the route,
  checkpoint hash, shortlist hash, and scorer id). A missing or mismatched
  policy prevents startup.
- **All decisions defer**: with a null operating threshold, every
  shortlist-route decision reports `automation: "defer"` (the gateway surfaces
  `defer_reason: "model_deferred"` on defer payloads); the decision payload
  still carries probabilities and confidence. Packed-route decisions report
  `automation: null`.
- **Why**: the validation-selected nominal 5% threshold (`0.9541679372957224`)
  observed 6.99% risk (26 of 372 accepted answers incorrect) on
  development-exposed reporting data, so no 1% or 5% risk guarantee is
  asserted (`artifacts/decisions/automation_policy.json`).
- **Activation path**: enabling automation requires blind evaluation on
  prospective target-domain data and a validated selective policy for that
  domain; the frozen thresholds (0.02 → `0.9796500892257723`,
  0.05 → `0.9541679372957224`, 0.10 → `0.896261827143334`) are preserved for
  descriptive research only.

## Version Requirements

All production artifacts are hash-pinned via `artifacts/release/manifest.json`
(format `veya-release-manifest-v2`):

- Runtime sources: SHA-256 for every served `veya/*.py` file
  (`runtime_files`, 15 files).
- Model: base checkpoint (`models/production/model.safetensors`), model config,
  tokenizer files.
- Routing and retrieval: choice router (`models/production/choice_router.json`)
  and the K=8 shortlist artifact (`models/production/shortlist_k8.pt`).
- Calibration: packed-choice and hierarchical-route calibration artifacts,
  including the frozen hierarchical temperature.
- Policy: `artifacts/release/selective_policy.json`.
- Evidence: data lineage and the blind evaluation freeze/result records.

The system is fail-closed on mismatch: `load_release_manifest` recomputes
every pinned hash at startup and raises before serving, so a changed pinned
file must be re-minted per [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md)
before the model service will start, and `/readyz` never turns 200 on a
mismatched tree. Post-audit working-tree drift against the runtime pins is
recorded by `research/veyabench/release_metadata.py`
(`python -m research.veyabench.release_metadata`).

For the machine-readable version of this scope, see
`artifacts/release/production_scope.json`. Operational procedures:
[`docs/OPERATIONS.md`](OPERATIONS.md), [`docs/RUNBOOK.md`](RUNBOOK.md).
