# Phase 12 SGD data audit

**Primary evaluation now uses every eligible USER frame with causal history.** The first-turn figures below are retained as a diagnostic only. The official test dialogue contents remain unopened.

| Primary all-turn population | TRAIN (22 services) | DEV (7 unseen services) |
|---|---:|---:|
| USER-frame decisions | 158,042 | 10,905 |
| `NONE` decisions | See `context_audit.json` | 749 |
| Service transitions | See `context_audit.json` | 894 |
| Exact-input deterministic ceiling, current utterance only | 97.5861% | 99.0371% |
| Exact-input deterministic ceiling, previous turn + current | 99.9519% | 100% |
| Exact-input deterministic ceiling, full preceding context + current | 99.9690% | 100% |

These ceilings are **in-sample collision bounds**, not model scores. The 32-pair TRAIN-only regression set in `artifacts/experiments/phase12/context_regression.json` contains identical current utterances with different intent labels and different preceding turns. Development class counts show both official intents represented in each of the seven services once all turns are included. Full aggregates: `artifacts/experiments/phase12/context_audit.json`.

Within TRAIN, 38 identical `(service, full dialogue through current USER turn)` input groups have conflicting official labels, causing at least 49 unavoidable errors. Reviewed examples are broad opening requests such as needing a rental car or a doctor; the utterance does not always distinguish search from booking. The all-turn DEV population has no exact full-history conflicts. These are lower bounds on annotation ambiguity, not a human agreement estimate.

The full-context key repeats with the **same** label in 1,903 TRAIN rows and 107 DEV rows beyond the first occurrence. These counts can include legitimate repeated dialogue templates; they do not establish accidental source duplication. The artifact also reports repeats for the other two input modes.

A deterministic 147-case development review queue is ready at `artifacts/experiments/phase12/label_review_blind.jsonl`: 21 cases per service, including high-confidence errors, sampled errors, and correct controls. Give reviewers only that blind file; `label_review_key.jsonl` holds the official label and model prediction for later adjudication. Two qualified reviewers should independently choose an official intent, `NONE`, `AMBIGUOUS`, or `INSUFFICIENT_INFORMATION` from the causal dialogue shown. The queue is deliberately enriched for errors, so its disagreement rate must not be treated as the population rate without weighting or a separate random sample. No reviewer judgments have been collected yet.

Nine service definitions shared by official `train/` and `dev/` are byte-for-byte identical. A token-overlap screen flags three TRAIN intent-description pairs and one DEV pair at Jaccard >=0.5, especially one-way versus round-trip flight search. This screen prioritizes human review; it is not itself a semantic ambiguity judgment.

Source: official SGD commit `e852981ae34990f4358979625854259302feaa78`. The local Phase 10 train/dev schema hashes exactly match the pinned official checkout. All official source file hashes and exact service split IDs are in `artifacts/experiments/phase12/sgd_inventory.json`.

The inventory's descriptive split rule was corrected from the obsolete initial-turn wording to all eligible USER frames. Source files, service membership, and hashes did not change. The pre-correction inventory remains at `artifacts/experiments/phase12/sgd_inventory_before_selection_rule_correction.json` because early baseline runs bind its hash.

The earlier initial-turn audit found 16,142 TRAIN and 2,482 DEV initial USER frames, with no missing text, missing labels, or unknown intents. Its normalized-text duplicate counts were 2,070 TRAIN and 205 DEV. These are recurring `(service, utterance, label)` rows, not necessarily source duplication; SGD is dialogue data and repeated requests can be legitimate. The earlier all-turn normalized-text probe used a different input key and population rule, so its ceiling should not be compared directly with the primary causal-input bounds above. Human inter-annotator disagreement, semantic ambiguity, and taxonomy drift require manual audit. There is no evidence here that humans can meet 98% on unseen services.

**Class-diversity failure for initial-turn evaluation:** six of the seven dev-only validation services show only one intent on their first turns. The per-service majority baseline is 971/1,000 = **97.1%**. This reflects the dialogue collection process, not robust intent understanding. Initial-turn routing is a diagnostic sub-benchmark and must not be used for 98% promotion.

The user-approved contextual contract compares current-only, immediately previous turn, and full preceding context with speaker labels. A prior SYSTEM-only audit also showed that context resolved most exact collisions, but the controlled three-mode model evaluation supersedes that feasibility probe.

The official [dataset documentation](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) defines `state.active_intent`, `NONE`, and the schema intent descriptions. The dataset is public research evidence. `service_name` does not stand in for a real customer, and its CC-BY-SA-4.0 terms need commercial review before using trained weights in a paid product.
