# Veya Phase 12: schema-guided service intent routing

**Verdict: 98% COMMERCIAL GATE NOT YET PASSED.** The strongest completed development scorer reaches 93.91% raw accuracy, below the frozen 98% target. The nine test-only service dialogues remain sealed. The released scorer's full-context control is still running.

## A. Workflow

At each eligible SGD USER frame, use only preceding USER/SYSTEM utterances plus the current USER utterance and the applicable official service schema to output an official `active_intent` or a separate `NONE` no-route outcome. `service_name` is a schema ID, not a customer ID. The [frozen contract](contract.md) requires >=98% raw end-to-end correctness, >=95% automation coverage, and an upper Wilson 95% bound <=2% accepted error. Deferred rows stay in the raw denominator. These are internal research gates, not a customer-agreed SLA.

## B. Data quality

Official [Google SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) is pinned to commit `e852981ae34990f4358979625854259302feaa78` under CC-BY-SA-4.0. The split has 22 TRAIN, seven development-only, and nine test-only multi-intent services. The all-turn development population has 10,905 USER-frame decisions, including 749 `NONE` and 894 service transitions. In TRAIN, 38 identical full-history input groups carry conflicting labels, imposing at least 49 unavoidable errors out of 158,042 decisions. Human agreement and commercial representativeness remain unmeasured. See [data quality](data_quality.md) and the [source inventory](../../../../artifacts/experiments/phase12/sgd_inventory.json).

The earlier 82.2% initial-turn-only result is retained only as a diagnostic. Six of seven development services had one observed first-turn intent; a per-service majority rule already scored 97.1% there. It is not the Phase 12 workflow population.

## C. Candidate access

All 10,156 active-intent DEV decisions have the correct label in their official candidate set. Every selected DEV service has two intents. K8, K16, K32, and all therefore expose the same set; Recall@8/16/32/All is **100%**. Recall@1 is a scorer metric here because there is no retrieval ranker. Source: `artifacts/experiments/phase12/candidate_access.json`.

With the best scorer's measured 93.92% conditional active-intent accuracy, the simple recall requirement `0.98 / 0.9392` is about **104.34%**, which is impossible. `NONE` is handled outside the candidate set, so this ratio is an active-intent diagnostic; the raw end-to-end gate independently fails at 93.91%.

## D. Selected candidate architecture

Score every official intent and include a distinct `NONE` no-route decision. Retrieval cannot raise candidate access above 100% and would add another failure mode. The current release already uses packed full scoring for this candidate count.

## E. Scorer comparison on matched development decisions

| Scorer | Context | Correct / 10,905 | Raw accuracy | Notes |
|---|---|---:|---:|---|
| Released packed Choice | Current USER only | 7,334 | 67.25% | Cannot emit `NONE` |
| Released packed Choice | Previous turn + current USER | 8,594 | 78.81% | Cannot emit `NONE` |
| Released packed Choice | Full preceding dialogue + current USER | Pending | Pending | Same checkpoint and DEV decisions |
| Train-only sparse schema scorer | Full preceding context | 10,117 | 92.77% | 158,042 TRAIN decisions; `NONE` modeled |
| Train-only sparse schema scorer | Previous turn + current USER | 10,241 | **93.91%** | 158,042 TRAIN decisions; `NONE` modeled |

The saved sparse weights were independently reloaded and reproduced both correct counts. Its previous-turn formulation beats the full-context formulation by 1.14 percentage points; a dialogue-cluster bootstrap interval is +0.53 to +1.73 points. Full context helps flights but hurts banks and restaurants. Even an oracle choosing the correct output when either sparse scorer is right reaches only 97.51%; this is a diagnostic bound for those two predictions, not a deployable ensemble.

With 100% active-intent candidate access, the best scorer's conditional accuracy on active-intent decisions is 9,539/10,156 = **93.92%**. The additional `NONE` outcome is assessed separately below.

## F. End-to-end development result

The best completed scorer's 10,241/10,905 raw accuracy is **93.91%** (Wilson 95% interval 93.45–94.34%). It needs at least 446 additional correct decisions on this population to reach 98%. It makes 664 errors. One-way versus round-trip flight search accounts for 296; `NONE` recall is 702/749 = 93.72%, precision is 702/731 = 96.03%, and 29 active-intent rows are incorrectly assigned no-route. The most difficult service is `Flights_3` at 84.17%. Per-service and confusion results are in `artifacts/experiments/phase12/sparse_scorer_158k_3_previous_turn.json`.

Across the 21 service-specific intent/`NONE` classes, macro-F1 is **93.73%**. The lowest class recall is **80.23%** for `Media_2/NONE` (86 cases), and `Flights_3/SearchRoundtripFlights` has **77.14% F1**. No important-class floor has been qualified for a finalist.

## G. Deferral

At the top 95.00% of decisions by the sparse scorer's **uncalibrated** confidence, observed accepted error is 434/10,360 = **4.19%** (Wilson upper bound **4.59%**). This is a development diagnostic, not a fitted production policy. Deferral cannot raise the raw 93.91% to 98%. The calibration and selective-deferral gates are therefore not promoted.

## H. Robustness

The causal-input tests cover future-turn exclusion, speaker boundaries, service transitions, overlength current turns, and embedded-newline speaker spoofing. Noise, paraphrase, production length tails, and customer-schema transfer remain untested for a finalist because none has passed the core quality gate.

## I. Sealed result

**Not run.** Test-only service dialogue contents have not been parsed or scored. Opening them now would consume the single sealed evaluation without a qualified finalist.

## J. Latency and cost

The released scorer's matched-mode throughput is recorded in `context_baselines.json`; the sparse scorer's full-context development loop took 8.27 seconds for 10,905 examples on local CPU. These are research throughput measurements. Production p50/p95/p99 latency, peak memory, named production hardware, and dated cost per 1,000 decisions have not been established; no SLA claim follows from them.

## K. 98% verdict

**98% COMMERCIAL GATE NOT YET PASSED.** No sealed or customer workflow result exists.

## L. Limiting factor

**Scorer discrimination under schema transfer.** Candidate access is complete, while sibling-intent and workflow-stage errors keep raw development accuracy below 98%. Official descriptions do not always spell out the search-before-booking stage represented by `active_intent`, so expert label adjudication is also needed before any commercial claim.

## M. Next action

Run **WF98-G11**: train one schema-conditioned contextual scorer on TRAIN services with explicit speaker/turn positions, learned relevance over earlier turns, and hard negatives for sibling intents and stage transitions. Preserve the seven-service DEV population for selection and keep the nine-service test sealed. If this cannot reach the frozen development gates, adjudicate the hard examples with qualified annotators before any taxonomy or target change.

Reproduction details and evidence gaps: [verification report](verification.md). Workflow graph: `artifacts/experiments/phase12/graph.json`.
