# Phase 10 Architecture Decision Packet

## Outcome

Improve high-cardinality English Choice retrieval on candidate schemas that are absent from retriever training, while keeping `K=8`, the base decision head, calibration, automation policy, and production serving path frozen.

## Current-state evidence

- The audited release uses the frozen Veya ModernBERT-large encoder, one shared bias-free `1024 -> 128` projection, dot-product ranking, and a Banking77-trained artifact (`models/production/shortlist_k8.pt`).
- The sealed CLINC150 validation result was Recall@8 `0.356`; conditional base-scorer accuracy was `0.9803`. Retrieval, not reranking, dominates blind errors.
- `ReleaseBundle.shortlist` already isolates retrieval from the packed scorer and caches base-encoder vectors by candidate text.
- The production manifest and its hashes must remain unchanged. Phase 10 therefore lives under `veya.experimental` and `artifacts/experiments/phase10` until promotion.

## Decision

Use the current frozen ModernBERT checkpoint as a shared text backbone and train only small query and candidate projections with a single contrastive cross-entropy objective. Compare a shared projection with asymmetric query/candidate projections; keep exact dot-product search and `K=8`. Train episodically across Banking77 and licensed Schema-Guided Dialogue service schemas, select on CLINC150 **train** rows as a whole-schema-held-out validation set, and never read CLINC150 validation/test results for Phase-10 selection.

The experimental retriever is loaded explicitly by `retriever: phase10_domain_general`; `retriever: current` leaves the audited bundle untouched. Candidate embeddings are cached by retriever ID, candidate-schema version, and candidate-text hash.

## Alternatives considered

- Fine-tune or duplicate the encoder backbone: potentially more capacity, but materially higher compute/memory and confounds retrieval-head gains with backbone changes. Rejected for the first loop.
- Reuse one shared projection only: simplest and the required baseline, but forces query and label text into the same transform. Retained as an ablation.
- Add an ANN service: operational complexity without evidence of need below roughly 1,000 candidates. Rejected pending exact-search profiling.

## Decision drivers

- Causal isolation: only retrieval parameters change.
- Schema transfer: no fixed label IDs or schema identifiers enter the model.
- Reversibility: production artifacts and manifest remain byte-for-byte preserved.
- Operability: candidate vectors can be reused; exact scoring is deterministic and cheap at low hundreds of candidates.
- Data integrity: whole candidate schemas, not random rows, are absent from training.

## Change boundary

- Components: add `veya.experimental.domain_retriever`, Phase-10 data/training/evaluation harnesses, registry, graph, reports, and tests.
- Interfaces: `load_domain_retriever(path, checkpoint_hash)`; `rank(query, candidates, schema_version, k)`; an explicit experimental bundle hook.
- Data/state: immutable versioned retriever artifact plus derived embedding caches; no production model or calibration mutation.
- Security: candidate/query strings remain tokenizer data and are never interpolated into privileged instructions, shell commands, or executable templates.
- Observability: stage latency, cache hit/miss, truncation, memory, cardinality, and exact artifact/config hashes are reported.

## Failure modes and mitigations

- Schema memorization -> exclude explicit dataset/service identifiers from text; validate on the full unseen CLINC150 schema.
- False cross-schema negatives -> deduplicate normalized candidate text and exclude exact semantic-name collisions.
- Candidate-definition mismatch -> compare raw, naturalized, and official-definition training separately; never fabricate definitions.
- Harder shortlists reduce scorer accuracy -> run matched `K=8` end-to-end evaluation with the unchanged scorer and compare margins.
- Cache staleness -> include retriever ID, schema version, and ordered candidate-text hash in every key; test invalidation.
- Research result leaks into production -> experimental module and explicit feature flag only; no production manifest edit.

## Migration and rollback

Phase 10 first produces a research-only artifact. Enabling it requires an explicit experimental configuration. Rollback is selecting `retriever: current`; no weights, calibration files, or manifest entries need restoration.

## Verification contract

1. The current retriever reproduces Recall@K on allowed development rows under the same harness.
2. No Phase-10 selection code reads Banking77 test or CLINC150 validation/test examples.
3. The selected retriever materially improves CLINC150-train Recall@8 and matched end-to-end accuracy with bootstrap uncertainty reported.
4. `K=8`, the base scorer, checkpoint, calibration, automation policy, and production manifest remain unchanged.
5. Fresh-process reload reproduces candidate embeddings, ranks, and Recall@K within tolerance.
6. Exact retrieval remains permutation-equivariant and cache invalidation follows all three version keys.
7. Existing generic, security, packaging, server, Noul, Score, and language regressions pass.
8. Latency and memory are measured at 50, 100, 150, 500, and 1,000 candidates.

## Open assumptions

- CLINC150 train rows are permitted development data; only its previously consumed validation/test partitions are prohibited.
- The first loop is English-only and uses the existing frozen checkpoint encoder, not a newly downloaded backbone.
- A future blind source will be chosen and frozen only after architecture and hyperparameters are locked.
