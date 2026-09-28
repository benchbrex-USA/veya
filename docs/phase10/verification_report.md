# Phase 10 Verification Report

## Verdict

**PASS WITH GAPS for the research claim; FAIL for serving promotion.**

The experimental retriever reproducibly improves unseen-schema Recall@8 and end-to-end accuracy. The release gate correctly rejects promotion because absolute blind recall remains low and distractor/long-state robustness fails. The audited production release is unchanged.

## Claims checked

| Claim | Evidence | Result |
|---|---|---|
| Whole schemas are absent from retriever training | Frozen split ledger: 22 train schemas, 5 development schemas, CLINC150 held out | PASS |
| Prohibited partitions were not used for Phase-10 selection | Phase-10 loaders only open Banking77 train, SGD train, and CLINC150 train | PASS |
| Current baseline is reproducible | Fresh process reproduced CLINC150-train Recall@8 0.355333 | PASS |
| Phase 10 materially improves unseen-schema Recall@8 | CLINC delta +0.0757, CI [0.0563, 0.0960]; MASSIVE blind +0.0646 | PASS |
| End-to-end improves with frozen scorer | Validation +0.061, CI [0.0280, 0.0950]; blind +0.0440, CI [0.0245, 0.0625] | PASS |
| Clean reload preserves top-8 behavior | 0/3,000 top-8 set changes; exact Recall@K match | PASS |
| Candidate ordering is clean | 0/256 score-map changes on validation and blind | PASS |
| Cache invalidates on retriever/schema/text changes | Focused tests | PASS |
| Only retrieval changes | K, scorer, calibration, automation, serving default, and release manifest unchanged | PASS |
| Latency remains tractable without ANN | Exact score/top-8 p50 0.024 ms @150, 0.062 ms @1,000; candidate encodings cacheable | PASS |
| Query robustness is acceptable for promotion | Distractor Recall@8 0.4267→0.1400; long suffix →0.0533 | FAIL |
| Generic behavior does not regress | Full/unit/HTTP/real-model/browser/build checks below | PASS |

## Commands and harnesses

- `python3 -m pytest -q` → 77 passed, 1 intentional skip, 6 subtests passed. The first sandboxed run could not bind localhost; the identical suite passed with ephemeral-loopback access.
- `python3 -m pytest -q tests/test_phase10_retriever.py` → 8 passed.
- `python3 -m mypy veya` → success across 23 source files.
- `python3 -m ruff check veya tests research/veyabench` → all checks passed.
- `python3 -m compileall -q veya tests research/veyabench` → passed.
- `VEYA_DEVICE=cpu python3 tests/test_local_e2e.py models/production` → 24 passed, 0 failed.
- `node --test tests/test_model_client.mjs tests/test_site_content.mjs` → 9 passed.
- `python3 -m build --no-isolation --outdir /tmp/veya-phase10-dist` → sdist and wheel built.
- `python3 -m research.veyabench.run_phase10_train` → finalist frozen.
- `python3 -m research.veyabench.run_phase10_validate` → clean-load, matched end-to-end, robustness, performance.
- `python3 -m research.veyabench.run_phase10_blind_massive` → one-shot consumed blind, aggregate-only.
- `python3 -m research.veyabench.run_phase10_diagnostics` → post-selection query/cardinality/runtime diagnostics; no retuning.

## Important observations

- The hard-negative refresh produced the largest controlled gain: Recall@8 0.3673→0.4310.
- Official-definition training did not beat the simpler name-only hard-negative model (0.4263 versus 0.4310).
- The Phase-10 shortlist is harder: blind conditional accuracy fell 0.8042→0.7817 even while end-to-end accuracy improved.
- Base-encoder values vary slightly with CPU batch padding (max absolute delta 0.00921), but minimum embedding cosine was 0.99999976, projected-score drift was 0.000372, and top-8 sets were identical.
- Exact search is not the latency bottleneck. Uncached text encoding is; the candidate cache is therefore the correct optimization seam.

## Failure-path coverage

Malformed/duplicate/confusable candidates, hostile candidate text, incompatible checkpoint artifacts, unknown feature flags, cache-version changes, candidate permutations, unsupported languages, request limits, malformed JSON, and server route contracts were exercised.

## Regressions and baseline comparison

The current and Phase-10 paths used identical examples, K=8, scorer weights, temperature, and candidate text for reranking. The production manifest hash remained `c7febd238afa79144d907cdc4eb4bd229a0bca705c952352302772f8269d9648` throughout. The Phase-10 mode is opt-in and the default remains `current`.

## Gaps

- Base-checkpoint pretraining provenance cannot independently exclude MASSIVE exposure.
- SGD is CC-BY-SA-4.0; legal/licensing compatibility for distributing trained weights was not adjudicated.
- No production traffic, GPU benchmark, or live canary was run because the artifact failed research promotion gates.
- Setuptools reports existing license-field and `web.static` package-discovery warnings; builds still succeed.

## Confidence

**HIGH** for the measured research improvement and rejection decision; **LOW** for any production-readiness claim, which the evidence explicitly does not support.
