# Phase 11 Architecture Decision Packet

## Outcome

Improve clean and noisy whole-schema-held-out Recall@8 without changing `K=8`,
the scorer, calibration, router cutoff, production manifest, production
retriever, or production serving path. Bound query views and CPU latency, and
produce evidence sufficient to decide whether the current dense architecture
is still improving or has plateaued.

## Current-state evidence

- Phase 10 uses the frozen ModernBERT backbone, right truncation at 96 tokens,
  attention-mask-aware mean pooling, and a shared bias-free `1024 -> 128`
  projection.
- Phase-10 clean CLINC150-train Recall@8 is 43.10%. The consumed MASSIVE and
  HWU64 aggregates are historical context only and are prohibited for Phase-11
  selection.
- Max fusion over bounded query views substantially improved positional noise,
  but local CPU query-path p50 increased from 70.0 ms to 108.2 ms.
- Right truncation means a sufficiently long prefix can remove the useful span
  completely. Projection-only training cannot reconstruct absent tokens.
- Mean pooling can dilute a retained useful span with irrelevant suffix tokens.
  That failure is addressable with TRAIN-only augmentation and invariance.

## Decision

Use a two-part dense strategy. First, train the existing Phase-10 shared
projection on a deterministic curriculum of clean plus generic TRAIN-only noise
whose useful span remains token-visible, with a separately ablated small
clean/noisy invariance loss. Then run exactly one second TRAIN-only hard-negative
refresh with duplicate and high-similarity false-negative filtering. Separately,
compare deterministic 1/2/3/4/adaptive view policies for position recovery and
select a quality-latency Pareto point.

Use frozen SGD development services for hyperparameter/ablation choices and the
whole CLINC150 train schema for final development validation. No consumed
external dataset participates in selection. Late interaction is allowed only if
both robust training and the second hard-negative refresh fail to materially
improve held-out robustness and remaining misses show token-level evidence.

## Alternatives considered

- **Heuristic views only:** reversible and already effective, but does not
  improve clean semantic alignment and carries a material latency cost.
- **Projection training on every long corruption:** rejected because long-prefix
  truncation can remove all label-bearing tokens; forcing invariance in that case
  teaches contradictory mappings.
- **Fine-tune the backbone:** potentially stronger but substantially more compute
  and confounds the first causal test. Deferred.
- **Immediate late interaction:** rejected until two credible dense improvements
  are measured and fail the plateau rule.

## Decision drivers

- Causal isolation: projection training, hard-negative refresh, and query-view
  policy are independently ablated.
- Data integrity: augmentation and mining use TRAIN only; whole schemas remain
  unseen during training.
- Position correctness: bounded windows recover spans that right truncation
  would otherwise delete.
- Latency: adaptive views avoid paying four encodes for short inputs.
- Reversibility: all behavior remains experimental and feature-flagged.

## Change boundary

- Components: Phase-11 noise generator, tokenizer audit, projection trainer,
  hard-negative ledger, validation/Pareto harness, artifact loader integration,
  reports, and tests.
- Interfaces: add experimental mode `phase11_noise_robust`; preserve existing
  `current`, Phase-10, and Phase-10 multi-view semantics.
- Data/state: new immutable research artifact, cache metadata, negative lineage,
  graph, and aggregate reports. No production artifact mutation.
- Security: state remains tokenizer-only data; byte/character, token, view, and
  candidate bounds remain enforced. View selection cannot inspect labels.
- Observability: raw/token length, truncation, retained/dropped counts, useful
  span position, views/query, cache memory, and latency distributions.

## Failure modes and mitigations

- Noise changes semantics -> fixed generic templates, label-preservation tests,
  deterministic audit sample, no candidate text in noise.
- Truncated positives collapse embeddings -> exclude variants whose useful span
  is not token-visible from projection training; retain them for view testing.
- False negatives are synonyms -> normalized duplicate rejection, candidate
  base-similarity threshold, lineage counts, and manual aggregate audit.
- Robustness erodes clean distinctions -> 1-point clean noninferiority floor and
  MRR/rank checks.
- View compute explodes -> maximum four views and fixed character bounds; reject
  malformed configuration.
- Validation leakage -> no MASSIVE, HWU64, CLINC val/test, or Banking77 test
  readers in Phase-11 selection code.
- Research leaks to production -> new explicit flag only; manifest hash check and
  rollback to `current`/Phase 10.

## Migration and rollback

Phase 11 produces a research-only artifact and explicit bundle configuration.
Rollback selects the existing Phase-10 or current retriever. No database,
manifest, scorer, calibration, or production weights require restoration.

## Verification contract

1. Baseline clean/noisy metrics reproduce from allowed data.
2. Noise generation is deterministic, generic, label-preserving, and TRAIN-only.
3. The tokenizer audit proves which conditions truncate or dilute useful text.
4. RB1 isolates augmentation; RB1I isolates invariance; RB2 adds only the second
   audited hard-negative refresh.
5. Clean Recall@8 does not fall more than 1 point from Phase 10.
6. Robustness curves, position sensitivity, length buckets, ranks, bootstrap
   uncertainty, and query-policy latency are reported under matched conditions.
7. A winner clean-loads with numerically equivalent top-8 sets and remains
   permutation-equivariant.
8. Frozen-scorer end-to-end evaluation uses identical rows, K, scorer, and
   temperature.
9. Production manifest, defaults, calibration, automation, and scorer remain
   unchanged.
10. Full regression, real-model, browser, static, package, JSON, and secret
    checks pass before any promotion decision.

## Open assumptions

- The frozen checkpoint tokenizer exposes stable token offsets or token IDs
  sufficient for aggregate truncation instrumentation.
- Projection-only training has enough capacity to improve dilution robustness;
  failure is valid plateau evidence, not a reason to tune on consumed blinds.
- SGD CC-BY-SA trained-weight compatibility remains unresolved and independently
  blocks commercial promotion.

