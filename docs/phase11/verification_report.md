# Phase 11 Verification Report

## Verdict

**PASS for the Phase-11 research evidence; BLOCKED for the repository-wide regression gate; FAIL for promotion.**

The Phase-11 implementation, experiment artifacts, clean reload, and focused tests pass. A concurrent, user-owned Phase-12 file contains Python 3.10/3.11-invalid f-string syntax, preventing an unqualified full-suite/Ruff/compile pass. That unrelated file was preserved rather than modified.

## Claims checked

| Claim | Result |
|---|---|
| Noise is deterministic, generic, label-agnostic, and TRAIN-only | PASS |
| Token retention and useful-span position are measured | PASS |
| RB0/RB1/RB1I/RB2 are matched frozen-backbone ablations | PASS |
| Second hard negatives have source/rank/score/screening lineage | PASS |
| Whole CLINC150 schema remained unseen in training | PASS |
| Consumed blind/external partitions were not used | PASS |
| 1/2/3/4/adaptive views and max/mean/RRF are measured | PASS |
| Late interaction obeyed the plateau gate | PASS — not executed because dense gains remained |
| Frozen scorer, K, calibration, router, automation, and production manifest stayed fixed | PASS |
| Artifact clean-load and permutation invariance reproduce | PASS |
| Development-selected winner meets promotion gates | FAIL |
| Entire repository regression gate is clean | BLOCKED by concurrent Phase 12 syntax |

## Verification results

- Focused Phase 10/11 suite: **26 passed**.
- Existing suite excluding only `tests/test_phase12_context.py`: **98 passed, 1 skipped, 6 subtests passed** with loopback permission.
- Unqualified full suite: collection blocked at `research/veyabench/phase12_context.py:82` because Python 3.11 rejects backslashes inside that f-string expression.
- Real-model local sweep: **24 passed, 0 failed**.
- Node/browser contracts: **11 passed**.
- mypy: success across **23 source files**, plus both Phase-11 harnesses with explicit package bases.
- Ruff: Phase-11 and all files excluding the concurrent `phase12_context.py` pass; unqualified run reports only its two syntax errors.
- compileall: all files excluding the same concurrent file pass; unqualified run reports only its syntax error.
- Package build: sdist and wheel built successfully.
- JSON validation: all Phase-11, graph, and decision JSON files parse successfully.
- Secret-pattern scan: no private-key, GitHub token, OpenAI-style key, or AWS access-key pattern found.
- `git diff --check`: passed.

## Clean reload

The artifact was loaded twice in the fresh validation process. Query embeddings, candidate embeddings, and scores had max absolute delta 0.0. There were 0 rank changes, 0 top-8 set changes, and 0/32 candidate-permutation score-map changes.

## Data and negative lineage

The training harness used Banking77 train and SGD train only. Selection used held-out SGD service schemas. CLINC150 train was opened only in the separate validation process. The second refresh accepted 37,080 negatives over 4,635 rows and records its full source, retriever, rank, similarity, schema, and false-negative screen.

## Known build warnings

Setuptools retains the existing deprecated license-table/license-classifier warnings and the existing `web.static` package-discovery warning. The build succeeds; none was introduced by Phase 11.

## Regression blocker ownership

`research/veyabench/phase12_context.py` and `tests/test_phase12_context.py` were created/updated concurrently after Phase-11 work began. Phase 11 did not edit them. The exact blocker is the escaped `r'\r'` / `r'\n'` replacement expression inside an f-string at line 82.
