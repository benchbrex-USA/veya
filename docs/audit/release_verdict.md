# Audited release verdict

## Executive verdict

**Veya is not currently trustworthy as a release candidate.** The repository now supports a reproducible, security-hardened **research release** for English Choice. English Noul, categorical Score, all multilingual primitives, ordinal Score, and Phase 9 remain **RESEARCH ONLY**.

The primary blind bottleneck is high-cardinality retrieval, not reranking: on untouched CLINC150 validation data, Recall@8 was 35.6%, conditional accuracy was 98.03%, and end-to-end accuracy was 34.9%.

## Actual served architecture

```text
English input
  ├─ 1–19 candidates -> packed base decision head
  └─ 20+ candidates  -> learned 128-d shortlist -> top 8 -> packed base decision head

Non-English detected -> defer unsupported_language
Automation            -> disabled
Phase 9               -> not loaded
Ordinal Score         -> not loaded
```

The verified startup path is `veya.release_bundle.load_release_manifest`. It rejects path traversal, file-hash mismatch, checkpoint/shortlist/calibration/policy incompatibility, scorer mismatch, and router/code cutoff mismatch.

## Release status by primitive and language

| Scope | Status | Reason |
|---|---|---|
| English Choice | **RESEARCH RELEASE** | Clean load and serving verified, but blind end-to-end accuracy/calibration are not release-candidate grade |
| English Noul | **RESEARCH ONLY** | Real endpoint and language gate verified; no independent audited NLL/Brier/ECE artifact |
| English categorical Score | **RESEARCH ONLY** | Real endpoint verified; no independent audited calibration artifact |
| English ordinal Score | **RESEARCH ONLY** | Implemented experiment only; no valid ordinal dataset |
| Multilingual Choice | **RESEARCH ONLY** | Real checkpoint runs, but shipped temperatures are unvalidated for release use |
| Multilingual Noul | **RESEARCH ONLY** | Same calibration limitation |
| Multilingual Score | **RESEARCH ONLY** | Same calibration limitation |
| Phase-9 Choice | **RESEARCH ONLY** | Reproducible, but development-exposed and not security-cleared |

## Sealed blind Choice metrics

Dataset: CLINC150 `val`, 3,000 in-scope examples, 150 candidates, evaluated once. No per-example or class-level output was written.

| Metric | Result |
|---|---:|
| Recall@8 | 0.3560 (95% Wilson 0.3391–0.3733) |
| Conditional accuracy given hit | 0.9803 (0.9701–0.9871) |
| End-to-end accuracy | 0.3490 (0.3321–0.3662) |
| Conditional NLL on hits | 0.1093 |
| Conditional Brier on hits | 0.04185 |
| Honest end-to-end NLL, retrieval misses floored at `1e-12` | 17.8333 |
| Honest 150-class Brier | 0.8600 |
| End-to-end ECE | 0.2673 |
| Retrieval misses | 1,932 |
| Reranker misses with target present | 21 |
| Permutation trials | 256 |
| Shortlist changes / top-1 flips / max remapped logit delta | 0 / 0 / 0.0 |

## Reliability

Runtime automation is disabled. Frozen development thresholds were applied to blind data descriptively only:

| Nominal development target | Blind coverage | Blind errors / accepted | Blind observed risk | 95% Wilson interval |
|---|---:|---:|---:|---:|
| 2% | 19.77% | 31 / 593 | 5.23% | 3.71%–7.32% |
| 5% | 29.53% | 84 / 886 | 9.48% | 7.72%–11.59% |
| 10% | 36.50% | 156 / 1,095 | 14.25% | 12.30%–16.44% |

No threshold supports a production risk guarantee.

## Runtime evidence

Hardware: Apple arm64, CPU, Python 3.11. Final sealed-manifest load was 33.85 s and RSS after benchmark was 2.03 GB. File-system cache state materially affects the hash/load stage.

| Path | p50 | p90 | p95 | p99 | Sequential throughput |
|---|---:|---:|---:|---:|---:|
| Small packed Choice | 105.3 ms | 106.4 ms | 108.6 ms | 109.7 ms | 9.48/s |
| High-N 77-way Choice | 163.4 ms | 164.9 ms | 165.2 ms | 166.1 ms | 6.12/s |
| Noul | 76.0 ms | 99.5 ms | 105.8 ms | 156.8 ms | 11.68/s |
| Categorical Score | 76.3 ms | 80.1 ms | 83.1 ms | 89.7 ms | 12.89/s |
| Three-question SystemOne | 214.8 ms | 231.8 ms | 237.4 ms | 293.7 ms | 4.52/s |

The blind 150-way path measured p50 214.9 ms, p95 242.9 ms, p99 262.0 ms, and 4.71 examples/s. SystemOne encoded four times as many input tokens for three mixed questions as the single Noul probe. Request-scoped hidden-state reuse was not added because state representations are question-conditioned.

## Security and serving

- Real `/health`, `/version`, `/v1/choice`, `/v1/noul`, `/v1/score`, and `/v1/systemone` requests passed.
- `/version` matched manifest, checkpoint, shortlist, router, scorer, and policy hashes.
- English direct API and all HTTP primitives deferred unsupported French/non-Latin text.
- Duplicate JSON keys, duplicate/near-duplicate candidate labels, malformed criteria, empty inputs, invalid primitives, and requests over 1 MiB are rejected.
- Candidate/state injection strings remain unprivileged input data at the API boundary. Phase-9 model-level adversarial sensitivity remains a blocker for that experimental scorer.

## Data integrity

- Banking77 train/test and used AG News rows are **DEVELOPMENT-EXPOSED**.
- The original CLINC150 `test` blind process failed after scoring during final optional RSS collection; it was not rerun and is **DEVELOPMENT-EXPOSED**.
- CLINC150 `val` was frozen afterward, evaluated once, and sealed with aggregate-only output.
- Full hashes and roles are in `docs/audit/data_lineage.json` and the release manifest.

## Verification commands

| Command | Result |
|---|---|
| `python3 -m pytest -q` | 59 passed, 1 intentional real-checkpoint skip, 6 subtests passed |
| `VEYA_DEVICE=cpu python3 tests/test_local_e2e.py models/production` | 24 passed, 0 failed |
| `python3 -m mypy veya` | Success, 22 source files |
| `python3 -m ruff check veya tests research/veyabench` | All checks passed |
| `python3 -m compileall -q veya tests research/veyabench` | Passed |
| `python3 -m build --no-isolation --outdir /tmp/veya-audit-dist-final` | sdist and wheel built |
| JSON schema parse sweep | 94 JSON files parsed |
| `python3 -m research.veyabench.run_phase9_hypothesis --stage clean-load` | 32 fingerprints, max delta `9.73e-05`, passed |
| `python3 -m research.veyabench.run_runtime_audit` | Real endpoints and manifest/version checks passed |
| `python3 -m research.veyabench.run_blind_clinc150` | Replacement blind run sealed once |

Full-repository Ruff still reports legacy style debt in historical notebooks and `research/scripts` (primarily multi-statement formatting and notebook import layout). The release package, tests, and Phase 1–9 audit/evaluation code are clean; the one material legacy undefined-name issue was repaired.

## Exactly one next research target

Train a **domain-general shortlist retriever** on diverse intent schemas with schema-held-out validation, while keeping K=8 and the current base reranker frozen. The blind result makes this the highest-leverage target: improving the scorer cannot recover the 1,932 examples whose true label never reached it.
