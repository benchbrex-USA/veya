# Prospective Blind Evaluation Workflow (G6/G8)

**Status: BLOCKED_EXTERNAL.** The full pipeline — runner, frozen config template,
provenance verification, P1–P6 verdict — is implemented, tested end-to-end with
synthetic data, and waiting on exactly one thing:

> **Minimum action to unblock:** supply a target-domain labeled dataset matching
> the schema below. Do not substitute benchmark data (Banking77/CLINC150 are the
> controlled training side and are screened out by P1).

This document is the single workflow for turning that dataset into a verdict.
Selects nothing, fits nothing, and tunes nothing on the evaluation data.

## 1. Exact command

```bash
cd "/Volumes/Parth Patel/Veya"
.venv/bin/python -m research.veyabench.run_prospective_blind \
  --config artifacts/release/prospective_blind_config.json \
  --data /path/to/target-domain-labeled.jsonl \
  --output artifacts/release/prospective_blind_result.json
```

| Exit code | Meaning |
|-----------|---------|
| 0 | PASS — all six primary gates hold |
| 1 | FAIL — the verdict is a documented failure (see the one-line summary and `primary_gates`) |
| 2 | Validation or runtime error — nothing was evaluated, no result written |

Setup before the first run:

1. Copy `artifacts/release/prospective_blind_config.template.json` to
   `artifacts/release/prospective_blind_config.json`.
2. Fill in `test_data.path` and `test_data.sha256` (the SHA-256 of the supplied
   dataset file) and nothing else. The runner refuses the unfilled template with
   exit 2 and repeats the minimum action above.
3. Fill in `operational` (P6 measurements) if operational viability is being
   claimed; `null` can only produce a FAIL on P6.

## 2. Input dataset schema

`--data` accepts **JSONL** (one JSON object per line, blank lines skipped) or a
**JSON array** of records. Every record is validated under a strict schema;
unknown fields, duplicate ids, and self-inconsistent records are rejected with
exit 2 before any statistic is computed.

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `id` | yes | non-empty string | unique across the dataset |
| `text` | yes | non-empty string | the request text |
| `label` | yes | non-empty string or non-negative int | gold label; int labels are probability-column indices |
| `scores` | yes | list of ≥ 2 numbers | probabilities over the candidate shortlist; each in (0, 1]; must sum to 1.0 within 1e-3 |
| `prediction` | optional | same type as `label` | the model's chosen candidate; defaults to `argmax(scores)` (`candidates[top]` for string labels) |
| `correct` | optional | boolean | defaults to `prediction == label`; when supplied it must agree with it, otherwise the record is rejected |
| `candidates` | optional | list, one entry per score | the shortlist; used to derive string predictions |
| `slice_tag` | optional | string | `"clean"` (default) or a registered protocol slice (see below) |
| `confidence` | optional | number in (0, 1] | defaults to `max(scores)` |

Additional rules enforced by the runner:

- Labels and predictions must use one consistent type across the whole dataset
  (all strings or all integer indices).
- Integer labels/predictions must be inside the score width.
- `slice_tag` must be a **registered** protocol slice — `clean` plus the 18
  registered perturbation conditions across the frozen noise families
  (`phase11-generic-noise-v1`, `blind-surface-noise-v1`,
  `blind-template-paraphrase-v1`): `short_prefix`, `short_suffix`,
  `long_prefix`, `long_suffix`, `middle_insertion`, `duplicate_metadata`,
  `reordered_fields`, `boilerplate_wrapper`, `json_noise`,
  `repeated_low_information`, `typo_injection`, `ocr_substitution`,
  `punctuation_strip`, `abbreviation`, `polite_prefix`, `formal_rewrite`,
  `casual_rewrite`, `question_form` (see
  `research.veyabench.blind_verdict.KNOWN_SLICE_CONDITIONS`). Unregistered slice
  names fail closed: the pre-registered protocol cannot evaluate a slice it
  never registered.
- The runner derives each case's `logits` as `ln(scores)` so the frozen
  temperature is applied to the supplied distribution unchanged.

Minimal record example:

```json
{"id": "td-0001", "text": "reschedule my delivery to friday", "label": "delivery_reschedule",
 "prediction": "delivery_reschedule", "correct": true,
 "scores": [0.97, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0], "candidates": ["delivery_reschedule", ...],
 "slice_tag": "clean"}
```

## 3. Provenance requirements (verified before any evaluation)

The run config (`veya-prospective-blind-config-v1`, see the template) is the
pre-registration. The runner fail-closes (exit 2) unless every pin matches:

1. **Config identity** — `format` must be `veya-prospective-blind-config-v1`,
   no unknown keys, all required keys present.
2. **Release manifest** — `manifest.path`/`sha256` must hash-match the
   re-minted `artifacts/release/manifest.json`, and the manifest must agree
   with the freeze record on checkpoint, shortlist, selective policy,
   temperature artifact, and git commit. The manifest's automation status must
   remain `DISABLED`.
3. **Frozen selective policy** — the artifact pinned by
   `selective_policy_artifact` must hash to both its pin and the freeze
   record's `selective_policy_sha256`. The run `policy` must be a byte-exact
   copy of the frozen policy **with only `operating_epsilon` switched** to one
   of the frozen thresholds (`"0.02"`, `"0.05"`, `"0.10"`; the template pins
   `"0.10"`). Thresholds, splits, hashes, and scorer identity may not change;
   `selective_policy.json` itself is never modified and selective automation
   stays disabled at runtime.
4. **Frozen temperature** — the pinned temperature artifact must hash to the
   freeze record's `temperature_artifact_sha256`, and the config `temperature`
   must equal the artifact's fitted temperature (1.971127355603397). The
   temperature is applied, never refit on evaluation data (P5).
5. **Dataset identity** — the `--data` file must hash to the pinned
   `test_data.sha256`. A hash mismatch is exit 2.
6. **Training-side corpus** — `training_texts` / `training_texts_path`
   (exactly one) supplies the controlled texts for the contamination screens;
   the template pins the Banking77 `train.csv` + `test.csv` corpora.

Ordering guarantee: inside the verdict, **split-provenance and contamination
screens run before any statistic** (P1 fails before later gates are
considered). This is locked by
`tests/test_blind_verdict.py::test_contamination_precedes_statistical_evaluation`
and `test_split_isolation_precedes_contamination_screens`.

## 4. Calibration/test split isolation

The freeze record and the run config pin **both** sides:

- `calibration_data` — `research/data/banking77/test.csv` (the frozen
  temperature was fit on its rows 1100–1300, `fitting_split
  banking77_test_1100_1300`), pinned by path and SHA-256.
- `test_data` — the external target-domain dataset, pinned by path and SHA-256.

The orchestrator (`research/veyabench/blind_verdict.py`) fails closed (before
contamination and every statistic) when the pins declare only one side, reuse
one path for both sides, or reuse one content hash for both sides. The verdict
echoes the validated pins as `split_provenance`. Locked by
`tests/test_blind_verdict.py::test_split_isolation_fails_closed_on_collisions`
and exercised end-to-end by
`tests/test_prospective_blind_cli.py::test_split_hash_collision_fails_closed`.

## 5. Expected outputs

One machine-readable JSON envelope (`veya-prospective-blind-result-v1`) at
`--output`, plus a **one-line verdict summary** on stdout. The output path is
written once with exclusive creation — a second run against the same output
refuses to overwrite (exit 2); the protocol runs once per output path.

Envelope shape:

```
format, status ("PASS"|"FAIL"), summary (the one-line verdict),
config {path, sha256}, dataset {path, sha256, n_records},
training_texts_count, verdict { … veya-blind-verdict-v1 … }
```

The `verdict` layer carries every required metric:

- `freeze_record_hash`, `split_provenance`, `contamination` (P1 evidence)
- `gates.slices.<tag>`: per-slice `risk_ucb95`, `coverage`, `observed_risk`,
  `wilson95`, `accuracy`, plus `gates.gates` G1–G4 with shift-slice risk
  (`G2_shift_safety.per_slice`), shift coverage/retention
  (`G3_coverage_retention.per_slice`), and collapse-floor checks
- `calibration.slices.<tag>`: ECE, Brier, reliability diagram, and the
  risk–coverage curve (`risk_coverage_deciles`), accepted vs deferred split
- `primary_gates` (P1–P6 booleans), `all_primary_gates_pass`
- `secondary`: clean-slice accuracy, macro F1, ECE, Brier, per-class rows,
  bootstrap CI
- `sample_size_check`, `result_language` (Section-16 language),
  `operational`, `slice_families`

No per-example predictions, scores, or error rows are ever written.

## 6. Gate criteria (P1–P6)

| Gate | Criterion | Source |
|------|-----------|--------|
| P1 data validity | Contamination screens flag ≤ `contamination_tolerance` (default 0) blind texts vs the controlled training side | `research/veyabench/contamination.py` |
| P2 selective safety | Clean-slice accepted-risk UCB95 ≤ frozen `epsilon` (G1) | `selective_gates.py` |
| P3 useful coverage | Clean-slice coverage ≥ frozen `c_min` | `blind_verdict.py` |
| P4 shift robustness | G2 (per-shift-slice risk UCB95 ≤ epsilon) AND G3 (shift coverage ≥ `coverage_retention_min` × clean coverage) AND G4 (no slice below the collapse floor) | `selective_gates.py` |
| P5 calibration transfer | Freeze record pins the temperature artifact; frozen temperature applied, never refit | `blind_verdict.py` |
| P6 operational viability | Caller-supplied `operational` measurements with no failures | `blind_verdict.py` |

Failure precedence is P1 > P2 > P3 > P4 > P5 > P6: the earliest failing gate
decides the result language.

`epsilon` (0.10) and `c_min` (0.80) in the template are **pre-registration
decisions**. Fix them before the dataset is unblinded; changing them afterwards
invalidates the pre-registration. The frozen thresholds themselves
(`selective_policy.json`) may never be changed. The required sample size is
reported by `sample_size_check` (Section 7 protocol table; e.g. 35 accepted
clean decisions with zero accepted errors clear the epsilon 0.10 bound).

## 7. PASS vs FAIL

- **PASS (exit 0)** — all six primary gates hold on the supplied dataset. The
  result language states the achieved coverage and the accepted-risk UCB95.
- **FAIL (exit 1)** — the run completed and produced a verdict, but at least
  one primary gate failed. The one-line summary names the failed gates; the
  envelope carries the full evidence. A FAIL is an evaluation outcome, not an
  error: the data was processed end-to-end.
- **Exit 2** — validation or runtime error (bad schema, hash mismatch, split
  collision, unfilled template, unwritable/second-run output). Nothing was
  evaluated.

## 8. How this workflow is verified without target-domain data

- `tests/test_blind_verdict.py` — the orchestrator contract: output coverage,
  gate wiring, ordering (contamination before statistics; split isolation
  before contamination), split-isolation fail-closed behavior, P1 precedence.
- `tests/test_prospective_blind_cli.py` — the CLI end-to-end on synthetic
  datasets only: PASS/FAIL/exit-2 semantics, full metric coverage in the
  result JSON, JSONL and JSON inputs, prediction derivation, split-hash
  collision fail-closed, placeholder fail-closed, tamper detection.
- The template itself was smoke-run against the real release artifacts with a
  synthetic dataset (PASS, exit 0) and unfilled (exit 2 with the blocker
  message).

## 9. External blocker

`artifacts/release/gap_audit.json` G6/G8 remain BLOCKED_EXTERNAL /
EXTERNAL_BLOCKER on the missing target-domain labeled dataset. The exact
minimum action: **supply a target-domain labeled dataset matching the schema in
section 2** — then the single command in section 1 produces the verdict.
