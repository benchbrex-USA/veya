# Veya Production Readiness Review

**Format:** `veya-production-readiness-matrix-v1` · **Release:** `0.3.5-audit-baseline.1` · **Generated:** 2026-09-30
**Certification status:** `IMPLEMENTATION_READY_PRODUCTION_CERTIFICATION_BLOCKED`
**Machine-readable source:** [`artifacts/release/production_readiness_matrix.json`](../artifacts/release/production_readiness_matrix.json)

## Supported Scope

Veya v1 — English, choice/noul/score/systemone, ≤100 candidates, selective automation with defer, target-domain deployment.

Per [`artifacts/release/production_scope.json`](../artifacts/release/production_scope.json): English only (`en`); decision primitives choice, noul, score-categorical, systemone; gateway candidate cap 100 (packed ≤ 19, shortlist-k8 above 19); selective automation policy loaded, enforced, and currently **disabled** (every decision defers); fail-closed startup on any hash or policy mismatch. Unsupported: multilingual, moderation, arbitrary schemas, unrestricted zero-shot, ordinal score, autonomous operation without deferral.

## Summary

- **PASS**: 20/25
- **BLOCKED_EXTERNAL**: 5/25 (G6, G8, G15, G20, G24)
- **FAIL**: 0/25
- **NOT_IN_SCOPE**: 0/25

Every node status in this review was **recomputed from actual evidence** (filesystem checks plus executed test suites), not asserted. Executed verification:

- Focused 11-suite gate (artifact integrity, release metadata, telemetry max_probability, shadow deployment, blind verdict, prospective-blind CLI, activation procedure, security hardening, health probes, reliability hardening, concurrency): **169 passed, 0 failed**.
- Full `tests/` suite: **442 passed, 1 failed, 3 skipped** (6 subtests passed). The single failure is a stale manifest-hash cross-reference in the commercialization-track frozen blind record — see [Known non-blocking inconsistency](#known-non-blocking-inconsistency).

## Closure work reflected in this review

- **Release integrity (Task 29)**: `manifest.json` re-minted — SHA-256 `cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de`; all 15 runtime files match, **0 drift**. The former `veya/selective.py` post-manifest drift risk is **resolved** (G14 now clean).
- **Telemetry correctness (Task 30)**: `max_probability` is now captured server-side before the response whitelist strips it; client response unchanged (G16 now clean). 15 dedicated tests in `tests/test_telemetry_max_probability.py`; 80 regression tests around the change passed.
- **G6/G8 pipeline (Task 31)**: one-command CLI wrapper (`research/veyabench/run_prospective_blind.py`), config template, and [`docs/PROSPECTIVE_BLIND_WORKFLOW.md`](PROSPECTIVE_BLIND_WORKFLOW.md); 18 CLI tests. Still **BLOCKED_EXTERNAL**: no target-domain labeled data.
- **G15 legal packet (Task 32)**: `legal_review_packet.json` with 6 flagged datasets and exact legal questions. Still **BLOCKED_EXTERNAL**: requires an authorized legal disposition.
- **G20 shadow deployment (Task 33)**: [`docs/SHADOW_DEPLOYMENT.md`](SHADOW_DEPLOYMENT.md) and 13 tests; real local shadow stack verified end-to-end. Still **BLOCKED_EXTERNAL**: live production routing / cluster access / ground-truth feed.
- **Selective activation (Task 34)**: `activation_procedure.json` (28 tests) and updated [`docs/OPERATIONS.md`](OPERATIONS.md). Policy remains **disabled**, conditional on a G6/G8 PASS.

## Readiness Matrix

| G-Node | Area | Requirement | Status | Evidence / Verification | Remaining Risk |
|--------|------|-------------|--------|--------------------------|----------------|
| G0 | Governance | Repository governance and system map | PASS | `system_map.json`, `production_scope.json` present | None |
| G1 | Scope | Machine-readable production scope contract | PASS | `production_scope.json`, `manifest.json`; strict integrity tests pass | None |
| G2 | Audit | Structured gap audit artifact | PASS | `gap_audit.json` | Baseline predates close-out; superseded by this matrix |
| G3 | Evaluation integrity | Contamination / train-test overlap detection | PASS | `contamination.py` + `tests/test_contamination.py` (passing) | None |
| G4 | Calibration | Calibration reporting (ECE, Brier, reliability, risk-coverage) | PASS | `calibration_report.py`, `veya/calibration.py` + tests (passing) | None |
| G5 | Selective safety | Selective prediction with epsilon guarantee and per-slice gates | PASS | `veya/selective.py`, `selective_gates.py`, `sample_size.py`, `selective_policy.json`, `activation_procedure.json`; 28 activation tests | Operating epsilon `disabled`; frozen thresholds activate only after G8 |
| G6 | Prospective evaluation | Prospective evaluation harness + one-command CLI | **BLOCKED_EXTERNAL** | `run_prospective_blind.py`, config template, `PROSPECTIVE_BLIND_WORKFLOW.md`; 18 CLI tests pass; fail-closed unfilled-template path verified (exit 2) | No target-domain labeled dataset exists |
| G7 | Robustness | Perturbation robustness suites (surface + paraphrase noise) | PASS | `surface_noise.py`, `paraphrase_noise.py` + tests (passing) | None |
| G8 | Production evidence | Blind prospective verdict (P1–P6 pre-registered gates) | **BLOCKED_EXTERNAL** | Verdict orchestrator + CLI verified by tests; no `prospective_blind_result.json` exists | Requires blind evaluation data collection (same blocker as G6) |
| G9 | API contracts | Contract validation and structured errors at the serving boundary | PASS | `web/contracts.py`, `web/server.py` + gateway/security tests (passing) | None |
| G10 | Concurrency | Concurrency-safe model serialization with documented guarantees | PASS | `docs/CONCURRENCY.md`, `tests/test_concurrency.py` (passing) | None |
| G11 | Performance | Latency/throughput runtime performance audit | PASS | `runtime_audit.json` bound to the re-minted manifest hash | Single-host CPU baseline (macOS arm64, n=20/primitive); not production-scale load |
| G12 | Reliability | Graceful shutdown, timeouts, overload 503, structured errors | PASS | `web/server.py` + `tests/test_reliability_hardening.py` (passing) | None |
| G13 | Security | Transport security headers, HSTS, audit logging | PASS | `web/server.py` + `tests/test_security_hardening.py` (passing) | None |
| G14 | Supply chain | Lock file, SBOM, hash-pinned dependencies and artifacts | PASS | `requirements.lock`, `sbom.json`, re-minted `manifest.json` (`cd1e2029…`); 31 strict integrity tests; **15/15 runtime files match, 0 drift** | Commercialization frozen blind record pins the pre-re-mint hash (historical, see below) |
| G15 | Licensing | License and third-party inventory with compatibility review | **BLOCKED_EXTERNAL** | `license_inventory.json`, `legal_review_packet.json` (6 flagged datasets, exact legal questions) | Requires authorized legal disposition for 6 datasets + phase10 derived files |
| G16 | Observability | Shadow-mode serving with structured telemetry and request IDs | PASS | `web/server.py` (`max_probability` captured server-side pre-whitelist), `shadow_schema.json`; 15 telemetry tests + 80 regressions pass | None — telemetry gap closed |
| G17 | Monitoring | Statistical drift monitoring | PASS | `drift_monitor.py` + `tests/test_drift_monitor.py` (passing) | None |
| G18 | Operations | Health endpoints (`/healthz`, `/readyz`, `/startupz`) with fail-closed readiness | PASS | `web/server.py` + `tests/test_health_probes.py` (passing) | None |
| G19 | Deployment | Containerization and orchestration (Docker, Kubernetes) | PASS | `Dockerfile`, `docker-compose.yml`, `k8s/` manifests present | Manifests only — no live cluster; image digest `not_built` |
| G20 | Shadow evaluation | Live-traffic shadow evaluation in production | **BLOCKED_EXTERNAL** | `SHADOW_DEPLOYMENT.md`, `shadow_schema.json`, `k8s/shadow-overlay/`; 13 tests pass; local shadow stack verified end-to-end | Requires live production routing, cluster access, ground-truth labeling feed |
| G21 | CI/CD | Tiered quality gates (PR gate, main gate, release-candidate gate) | PASS | `.github/workflows/ci.yml`, `.github/workflows/release-gate.yml` | Production-evidence-gate job is a manual placeholder pending G8 |
| G22 | Release engineering | Reproducible release metadata (build provenance) | PASS | `release_metadata.py` + `tests/test_release_metadata.py`; metadata regenerated after matrix/report update | CI provenance fields are `local` |
| G23 | Documentation | Operator documentation and runbooks | PASS | `OPERATIONS.md` (incl. activation procedure), `RUNBOOK.md`, `PRODUCTION_SCOPE.md`, `CONCURRENCY.md`, `PROSPECTIVE_BLIND_WORKFLOW.md`, `SHADOW_DEPLOYMENT.md` | None |
| G24 | Governance | Final production-readiness verdict matrix | **BLOCKED_EXTERNAL** | This matrix recomputed from evidence: 20 PASS / 5 BLOCKED / 0 FAIL | Cannot PASS while G6/G8/G15/G20 are blocked |

## Blocked items — minimum external actions

| Blocks | Action |
|--------|--------|
| G6, G8 | Supply a target-domain labeled dataset matching the schema in [`docs/PROSPECTIVE_BLIND_WORKFLOW.md`](PROSPECTIVE_BLIND_WORKFLOW.md) and run the one-command CLI |
| G15 | An authorized legal reviewer provides an immutable disposition for the 6 flagged datasets (see `artifacts/release/legal_review_packet.json`) |
| G20 | Provide production cluster access, traffic-mirroring credentials, and a ground-truth labeling feed |

## Release integrity

The manifest was re-minted during closure (Task 29). Current state, verified by `tests/test_artifact_integrity.py` (31 strict tests):

- Manifest SHA-256: `cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de`
- Runtime files matching pins: 15/15
- Drift: **0**

The previous matrix's residual risk ("`veya/selective.py` drifted after the manifest was minted; manifest re-mint required") is **removed**.

## Known non-blocking inconsistency

`tests/test_commercialization.py::test_commercialization_freezes_match_artifacts` (1 failure in the full-suite run) asserts that the commercialization frozen blind record `artifacts/commercialization/blind_hwu64_result.json` pins the *current* manifest hash. That record was produced under the **pre-re-mint** manifest (`c7febd238afa…`) and was deliberately left unmodified: it is historical evidence of which runtime mint produced the blind result. This is a commercialization-track bookkeeping item (the re-pointing belongs to a future commercialization freeze refresh), not a G0–G24 requirement failure, and no node status in this matrix depends on it.

## Selective automation state

`selective_policy.json` `operating_epsilon` is `"disabled"` and the manifest records `automation.status = DISABLED` — every decision defers. The frozen thresholds (0.02/0.05/0.10) and the pre-registered ε=0.10 activation path are governed by `artifacts/release/activation_procedure.json` and `docs/OPERATIONS.md`, and remain inactive until a G6/G8 PASS on target-domain data.

## Conclusion

```
IMPLEMENTATION READY — PRODUCTION CERTIFICATION BLOCKED
```

All local implementation for the v1 scope is complete and verified by 169 focused-gate tests (0 failures) and 442/445 full-suite passes. Production certification is blocked on exactly three external inputs: target-domain labeled data (G6/G8), an authorized legal disposition (G15), and live shadow traffic with ground truth (G20). This document does **not** claim production readiness.
