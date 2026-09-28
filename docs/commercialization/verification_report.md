# Commercialization Increment Verification

## Verdict

**PASS for the research claims; FAIL for commercial promotion.**

## Evidence integrity

- The max-view policy was selected on the previously development-designated
  CLINC150 train partition and frozen before HWU64 download.
- HWU64 source commit, protocol, data, license, sample IDs, retriever, policy,
  and production manifest are hash-pinned.
- The blind report is aggregate-only. It was not used to modify weights,
  policy, K, scorer, calibration, or the production manifest.
- Four exact text overlaps with Phase-10 data are disclosed. The result is
  one-shot external evidence, not a pristine contamination-free blind claim.
- Base-checkpoint pretraining exposure cannot be excluded.

## Claims checked

| Claim | Result |
|---|---|
| Existing Phase-10 behavior remains available unchanged | PASS |
| Multi-view behavior requires an explicit feature flag | PASS |
| Single-line clean queries preserve Phase-10 rankings | PASS |
| Query views are bounded and deterministic | PASS |
| Candidate cache remains versioned and reusable | PASS |
| Candidate strings remain data, never executable instructions | PASS |
| Development suffix robustness materially improves | PASS |
| Improvement transfers to a new external schema | PASS |
| Absolute quality is commercially credible | FAIL |
| Route calibration/selective risk is validated | FAIL |
| Production SLO/cost is known | FAIL |
| Paid-beta product/security/revenue gates pass | FAIL |

## Reproducible commands

```text
python3 -m research.veyabench.run_commercial_query_robustness
python3 -m research.veyabench.run_commercial_blind_hwu64
python3 -m pytest -q tests/test_phase10_retriever.py
python3 -m ruff check veya tests research/veyabench
python3 -m mypy veya
python3 -m compileall -q veya tests research/veyabench
```

## Final regression results

- Full Python suite: **84 passed, 1 skipped, 6 subtests passed**. The first
  sandboxed run could not bind ephemeral localhost ports; the identical suite
  passed with loopback permission.
- Focused multi-view and commercialization artifact suite: **14 passed**.
- Real-model local suite: **24 passed, 0 failed**.
- Browser/Node contracts: **11 passed**.
- Ruff: all checks passed.
- mypy: success across 23 source files.
- compileall: passed.
- sdist and wheel: built successfully.
- JSON validation: all commercialization graph, freeze, policy, decision,
  registry, and result artifacts parsed successfully.

The build retains existing setuptools warnings about deprecated license-table
metadata and `web.static` package discovery. They did not fail the build and
were not introduced by this increment.
