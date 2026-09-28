# Phase 12 verification report

## Verdict

**FAIL for the frozen 98% research gate.** The best completed train-only scorer has 93.91% raw accuracy on 10,905 development decisions. The sealed test remains unopened. The released model's full-history control is still running; its other two controls are recorded in `context_baselines.json`.

## Claims checked

| Claim | Evidence | Result |
|---|---|---|
| Source and splits are pinned | Official SGD commit `e852981ae34990f4358979625854259302feaa78`; all file SHA-256 values and whole-service IDs in `sgd_inventory.json` | PASS |
| Every eligible USER frame is represented | 158,042 TRAIN and 10,905 DEV decisions in `context_audit.json`; 749 DEV `NONE` and 894 service transitions retained | PASS |
| Model input is causal and preserves speaker boundaries | `phase12_context.py`; synthetic future-turn, transition, overlength, and embedded-newline checks in `tests/test_phase12_context.py` | PASS for tested invariants |
| Candidate access can support 98% | `candidate_access.json`: all 10,156 active DEV labels present among exactly two official candidates per service | PASS for candidate access |
| Current released scorer satisfies 98% | Matched DEV current-only 67.25%; previous-turn 78.81%; full-history run pending | FAIL |
| Workflow-specific scorer satisfies 98% | Saved train-only previous-turn weights reload to 10,241/10,905 = 93.91%; full-history weights reload to 10,117/10,905 = 92.77% | FAIL |
| Deferral can satisfy risk at 95% coverage | Previous-turn scorer at the top 95% by its uncalibrated confidence has 4.19% accepted error, Wilson 95% upper bound 4.59%; raw accuracy remains below 98% | FAIL on DEV diagnostic |
| Finalist is safe to evaluate on sealed test | Scorer quality and downstream gates have not passed | NOT RUN |

## Commands and artifacts

```text
python3 -m research.veyabench.phase12_sgd_inventory
python3 -m research.veyabench.run_phase12_context_audit
python3 -m research.veyabench.run_phase12_candidate_access
python3 -m research.veyabench.run_phase12_context_baselines
python3 -m research.veyabench.run_phase12_sparse_scorer --mode full_context --max-train 158042 --epochs 3 --write-details --save-weights
python3 -m research.veyabench.run_phase12_sparse_scorer --mode previous_turn --max-train 158042 --epochs 3 --write-details --save-weights
python3 -m pytest -q tests/test_phase12_context.py tests/test_phase12_sparse_scorer.py
python3 -m ruff check research/veyabench/phase12_context.py research/veyabench/phase12_sgd_inventory.py research/veyabench/run_phase12_context_audit.py research/veyabench/run_phase12_context_baselines.py research/veyabench/run_phase12_candidate_access.py research/veyabench/run_phase12_sparse_scorer.py
```

The saved sparse weights were reloaded independently and reproduced both reported development correct counts. The initial inventory's obsolete descriptive split rule was corrected without changing source hashes or service membership; `sgd_inventory_before_selection_rule_correction.json` preserves the hash bound by the first two released-scorer controls.

## Important observations

- The TRAIN regression set contains 32 pairs with identical current utterances, different active labels, and different preceding context. Current-only deterministic routing cannot classify both members of a pair correctly.
- Exact full-history collisions affect 38 TRAIN input groups and impose at least 49 unavoidable errors. Human agreement remains unmeasured.
- The best sparse scorer makes 664 DEV errors. One-way versus round-trip flight labels account for 296 of them. Its 95%-coverage confidence policy would still make 434 accepted errors.
- An oracle choosing between the two saved sparse predictions when either is correct reaches only 10,634/10,905 = 97.51%; this is a feasibility diagnostic, not a deployable ensemble.

## Failure paths exercised

Empty or over-budget current turns explicitly defer; embedded line breaks cannot spoof speaker boundaries; unknown SGD speaker values and absent official gold candidates raise errors; test dialogue loading is rejected by the development loader.

## Gaps

The released scorer's full-history control is pending. Human adjudication, a separate service-disjoint calibration set, robustness, production latency tails, hardware cost, customer representativeness, and licensing review are pending. Official test dialogue contents have not been parsed or scored. Base-checkpoint pretraining exposure to public SGD is unknown.

**Confidence: HIGH** that the completed development candidates fail the 98% gate on this pinned evaluation; **LOW** for any commercial-readiness claim.
