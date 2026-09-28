# Phase 12 contract: schema-guided service intent routing

**Workflow ID:** `schema_guided_service_intent_routing`  
**Research contract frozen:** 2026-09-23, before contextual scorer evaluation. The 98% gate will not change after development results. The `NONE` output semantics below were clarified before model results because mandatory deferral for every `NONE` contradicted the predeclared 95% coverage gate on the development population. This is not a customer-approved commercial SLA.

## Decision unit

Given the dialogue history available **up to and including a USER turn** and the applicable official SGD `service_name`, select that turn's `state.active_intent` from the service's official intents. The caller supplies the applicable service; service detection is outside this workflow. `service_name` is a service/schema ID, **not a customer ID**.

Each `(dialogue_id, turn_index, frame_index)` is one decision. If a USER turn has multiple service frames, evaluate each frame using its applicable service. Prior turns remain in chronological order with `USER:` and `SYSTEM:` boundaries. No future turn, future state, slot/action annotation, or later service identity enters model input. Current `active_intent` is only the target.

```yaml
workflow_id: schema_guided_service_intent_routing
workflow_version: 1
business_problem: Select the active API intent for the current user turn within a known service schema.
input_schema:
  dialogue_history: previous USER/SYSTEM utterances in chronological order, plus current USER utterance
  service_name: official SGD service identifier legitimately applicable to this turn
  schema_version: pinned official SGD commit and schema file hash
required_fields: [dialogue_history, service_name, schema_version]
optional_fields: []
candidate_schema: official schema.intents[].name and schema.intents[].description; schema.description may appear in the instruction; NONE is a separate no-route outcome, not a schema intent
candidate_count: 2-4 official intents per selected service; one-intent services are outside this choice workflow
supported_languages: [en]
expected_input_length: measure all turns and tokens; pack with an explicit most-recent-turn-preserving model budget
latency_target:
  p50: <= 250 ms
  p95: <= 500 ms
  p99: <= 1000 ms
accuracy_target: >= 98% correct intent-or-NONE action on the sealed service-held-out population
automation_coverage_target: >= 95% of eligible USER-frame decisions
maximum_accepted_risk: upper Wilson 95% bound for accepted-decision error <= 2%
defer_behavior: typed defer with reason; caller handles manually
unsupported_behavior: defer unknown service/version, unsupported language, empty or over-limit current utterance, or low-confidence intent/NONE prediction
cost_target: <= USD 1 per 1000 eligible decisions on named deployment hardware with dated price source
evaluation_population: all USER-frame decisions for selected >=2-intent services, with original prevalence and no post-hoc row filtering
known_exclusions: one-intent services because no choice exists; SYSTEM turns because intent is selected at USER turns
```

`NONE` is an official `active_intent` state but not a schema intent. It stays in the denominator. A confident `NONE` prediction is an automated, typed **no-route** decision and must be correct to count as correct. An uncertain case is a distinct typed **defer** decision and does not count toward automation coverage. Report `NONE` prevalence, `NONE` recall/precision, and false no-route decisions separately. Candidate Recall@K applies to active-intent rows; coverage/risk and end-to-end correctness use all eligible rows. This preserves the fixed 98% accuracy and 95% coverage gates without excluding any `NONE` rows.

For raw end-to-end accuracy, every deferred row remains in the denominator and counts as not correctly automated. Accepted-decision accuracy is computed only over non-deferred rows. Thus deferral cannot raise the raw 98% figure; it can only trade coverage for lower accepted-decision risk, which is reported separately.

## Controlled input comparison

Score the same development decisions and checkpoint under exactly three input modes:

1. `utterance_only`: `USER: <current utterance>`.
2. `previous_turn`: immediately preceding USER or SYSTEM utterance, then current USER utterance.
3. `full_context`: every preceding USER/SYSTEM utterance in chronological order, then current USER utterance.

For bounded model input, pack newest **complete** turns first and drop oldest complete turns only. Always preserve the entire current USER turn. If the current turn alone exceeds the budget, defer and count the case in coverage. Record raw and packed turns/tokens, dropped turns, service transitions, and over-limit cases. Never silently truncate away the current turn. Choose and freeze one formulation on development data before opening sealed test dialogues.

The current-only mode is a controlled diagnostic. Since the workflow requires preceding dialogue context, only `previous_turn` and `full_context` are eligible final inputs. Select the one with higher raw all-turn development accuracy on the matched population; if tied, prefer fewer context drops, then shorter input. Preserve the selected mode for scorer training, calibration, and sealed evaluation.

## Source and split

- Official [Google SGD repository](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), commit `e852981ae34990f4358979625854259302feaa78`, CC-BY-SA-4.0. Project-owned checkout: `data/external/sgd/` (Git ignored). Exact file hashes: `artifacts/experiments/phase12/sgd_inventory.json`.
- **TRAIN:** 22 multi-intent services from official `train/`. Phase 10 used 21 train schemas for training and five for development; all 26 are development-exposed for sealed claims.
- **DEVELOPMENT:** seven multi-intent `dev/` services absent from `train/`: `Alarm_1`, `Banks_2`, `Flights_3`, `Hotels_4`, `Media_2`, `Restaurants_2`, `Services_4`.
- **SEALED TEST:** nine multi-intent `test/` services absent from `train/` and `dev/`: `Buses_3`, `Events_3`, `Flights_4`, `Homes_2`, `Media_3`, `Music_3`, `Payment_1`, `RentalCars_3`, `Trains_1`. Test dialogue files are hashed but **not parsed** before finalist freeze.
- Service identity is known; this measures schema transfer, not customer transfer. Base-checkpoint pretraining exposure is unknown. Public SGD alone cannot prove customer demand or paid-pilot suitability.

## Metrics and gates

Report raw end-to-end correctness, accepted-decision accuracy, coverage, defer rate, false accepts, Wilson intervals, per-service/class recall, macro-F1, candidate access, calibration, context truncation, transition slices, p50/p95/p99, throughput, memory, and cost. Contextually ambiguous repeated utterances from TRAIN/DEV form a regression set; none come from test.

All selected services have at most four official intents. Full candidate scoring gives 100% candidate access for active-intent cases; K8/K16/K32/All yield identical candidate sets. Current packed Choice is the control. Retrieval and late interaction are not justified unless a future contract expands the candidate schema.

Sealed success requires accuracy >=98%, coverage >=95%, accepted-risk upper 95% bound <=2%, an evidence-based class floor frozen before test, latency/cost targets, and no robustness blocker. A paid pilot further requires customer traffic, an approved error-cost matrix, production hardware/pricing, security review, and CC-BY-SA-4.0 licensing review.

The earlier initial-turn-only 82.2% result is a **diagnostic sub-benchmark**. Six of seven development services had only one observed first-turn intent, giving a 97.1% per-service majority baseline. It cannot establish this workflow's 98% result.
