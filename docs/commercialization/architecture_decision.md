# Commercialization Architecture Decision

## Decision

Run the commercialization program as a gated graph. The active branch is
noise-robust, unseen-schema retrieval. The production release, shortlist size
(`K=8`), scorer, calibration, automation policy, and serving default remain
frozen until a retrieval candidate passes development and new-blind gates.

The first implementation is an opt-in multi-view query policy around the
Phase-10 retriever. It separates an unstructured state's leading semantic
segment from trailing lines, embeds the bounded views in one batch, and
compares deterministic score/rank fusion policies. This directly tests the
measured Phase-10 failure without changing weights or reading a blind split.

## Current evidence

- Audited production remains a research release with automation disabled.
- Phase 10 improved schema-held-out Recall@8 from 35.53% to 43.10% and blind
  Recall@8 from 28.51% to 34.97%.
- The frozen scorer remains strong when retrieval succeeds, but retrieval
  misses account for 1,934 of 2,161 Phase-10 blind errors.
- On the frozen 300-row robustness sample, a short irrelevant suffix reduced
  Recall@8 from 42.67% to 14.00%; a long suffix reduced it to 5.33%.
- The Phase-10 artifact is research-only and has an unresolved SGD
  CC-BY-SA-4.0 release-compatibility question.

## Component boundaries

| Component | Responsibility | State in this increment |
|---|---|---|
| Audited release | Existing default behavior and artifacts | Immutable |
| Query-view policy | Deterministic, bounded views of untrusted state data | New, experimental |
| Retriever | Phase-10 projection and exact top-K | Frozen weights |
| Scorer | Existing base decision head | Frozen |
| Evaluation | Clean, suffix-noise, and latency comparisons | Extended |
| Product/API | Existing research gateway | No maturity promotion |
| Revenue/enterprise | Metering, billing, tenant controls | Gated, not started |

## Interfaces and data flow

```text
state (untrusted data)
  -> bounded deterministic query views
  -> one batched frozen-backbone encode
  -> frozen Phase-10 query projection
  -> exact scores against versioned candidate cache
  -> deterministic fusion
  -> top 8
  -> frozen decision scorer
```

Query views never become instructions, code, shell input, or templates. A
single-line state produces exactly one view. Multi-view behavior requires an
explicit experimental retriever mode; `current` and
`phase10_domain_general` retain their existing semantics.

## Alternatives considered

- **Increase retriever/scorer capacity:** rejected for this loop because the
  measured failure is input robustness, not demonstrated capacity.
- **Silently strip metadata-looking lines:** rejected because it creates
  domain-specific hidden semantics and can delete legitimate content.
- **Cross-encode every candidate:** rejected because it changes the system
  cost envelope before the dense retrieval failure is resolved.
- **Late interaction:** retained as the next architecture escalation only if
  deterministic multi-view and dense-training work plateau.
- **Build auth/billing now:** rejected by the program's quality-first gate; it
  would productize a route that is not commercially credible.

## Acceptance gate

The multi-view candidate may advance to a frozen retriever candidate only if:

1. clean schema-held-out Recall@8 is non-inferior to Phase 10 by no more than
   1.0 percentage point;
2. short- and long-suffix Recall@8 each retain at least 95% of clean Recall@8;
3. the candidate improves the worst robustness slice by at least 20 points;
4. inference remains deterministic and candidate-order invariant;
5. query views are bounded and hostile strings remain tokenizer-only data;
6. production manifest, scorer, K, calibration, and default mode are unchanged;
7. a new blind dataset is not opened until the policy and thresholds freeze.

Among policies that pass those gates, select for the best worst-case Recall@8
across all positional and structured-state diagnostics. This prevents an exact
suffix heuristic from winning while failing the same noise as a prefix.

Failure keeps the work at `VALIDATED / NOT SHIPPABLE` and routes the next graph
to trained query robustness or late interaction. Passing development does not
authorize production promotion; it only authorizes a new blind evaluation.

## Migration and rollback

The migration is an explicit feature flag on one loaded bundle instance. The
rollback is selecting `retriever: current` or the unchanged
`phase10_domain_general` mode. No database, manifest, or production artifact
must be restored.

## Verification contract

- Unit-test view extraction, bounds, deduplication, fusion, cache reuse, and
  feature-flag isolation.
- Re-run the frozen Phase-10 robustness sample with identical IDs and
  candidates.
- Record clean and perturbed Recall@K, MRR, ranking stability, view count, and
  query-path latency.
- Run repository unit, static, packaging, local-model, and browser regressions
  in proportion to the touched surface.
