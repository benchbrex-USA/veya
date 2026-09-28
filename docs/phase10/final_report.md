# Veya Phase 10 Final Report

## A. Core result

**Yes, retrieval generalized to unseen schemas, but not strongly enough for promotion.**

The selected shared-projection retriever with one hard-negative refresh improved whole-schema-held-out CLINC150-train Recall@8 from **35.53% to 43.10%** (delta **+7.57 points**, paired bootstrap 95% CI **+5.63 to +9.60**). On the untouched MASSIVE English blind test, Recall@8 improved from **28.51% to 34.97%** and frozen-scorer end-to-end accuracy from **22.93% to 27.34%** (delta **+4.40 points**, 95% CI **+2.45 to +6.25**).

The result is statistically real but operationally insufficient. Absolute blind recall remains low, the new shortlist is harder for the frozen scorer, and irrelevant/long query suffixes cause severe retrieval collapse. Decision: **KEEP AS RESEARCH**.

## B. Dataset registry

| Dataset | License | Role | Examples used | Labels/schemas | Contamination status |
|---|---|---|---:|---:|---|
| Banking77 train | CC-BY-4.0 | TRAIN | 3,080 | 77 / 1 | Development-exposed; test prohibited |
| Schema-Guided Dialogue train | CC-BY-SA-4.0 | TRAIN | 1,555 | 40 / 21 | New in Phase 10 |
| Five SGD services | CC-BY-SA-4.0 | DEVELOPMENT | 260 | 13 / 5 | Entire services excluded from training |
| CLINC150 train | local upstream license | SCHEMA-HELD-OUT VALIDATION | 3,000 | 150 / 1 | Train partition only; prior val/test prohibited |
| MASSIVE en-US test | CC-BY-4.0 | ONE-SHOT BLIND | 2,974 | 60 / 1 | Downloaded only after finalist freeze; now consumed |

The training artifact inherits a release-risk question from SGD's CC-BY-SA-4.0 terms. It must remain research-only until licensing compatibility is reviewed. The base checkpoint's pretraining provenance cannot independently exclude prior MASSIVE exposure, so the blind claim is specific to Phase-10 development exposure.

## C. Retriever comparison

CLINC150 train partition, 3,000 examples, all 150 candidates:

| Retriever | Recall@1 | Recall@4 | Recall@8 | Recall@16 | MRR | Cached query encode | Exact rank @150 | Projected cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 current, production-prefixed | 12.70% | 26.27% | 35.53% | 46.93% | 0.2170 | same backbone | <0.1 ms | 76.8 KB |
| B1 current, naturalized | 9.10% | 23.67% | 34.87% | 47.43% | 0.1863 | same backbone | <0.1 ms | 76.8 KB |
| B2 multi-schema | 13.43% | 26.53% | 36.73% | 48.20% | 0.2238 | same backbone | <0.1 ms | 76.8 KB |
| **B3 + hard negatives** | **17.17%** | **33.23%** | **43.10%** | **54.03%** | **0.2707** | ~80 ms CPU | 0.024 ms | 76.8 KB |
| B4 + official SGD definitions | 18.87% | 33.37% | 42.63% | 54.77% | 0.2816 | same backbone | <0.1 ms | 76.8 KB |

At 150 candidates, uncached candidate encoding/projection was ~1.26 seconds on CPU; it is cacheable. Exact scoring remained below 0.063 ms even at 1,000 candidates, so ANN infrastructure is not justified.

## D. End-to-end comparison

Same K=8, base decision head, temperature, and 1,000-example frozen validation subset:

| Retriever | Recall@8 | Conditional accuracy | End-to-end accuracy | Mean top1-top2 margin |
|---|---:|---:|---:|---:|
| Current | 36.40% | 98.08% | 35.70% | 0.781 |
| Phase 10 | **43.10%** | 96.98% | **41.80%** | 0.741 |

Untouched MASSIVE blind:

| Retriever | Recall@8 | Conditional accuracy | End-to-end accuracy |
|---|---:|---:|---:|
| Current | 28.51% | 80.42% | 22.93% |
| Phase 10 | **34.97%** | 78.17% | **27.34%** |

The new retriever improved end-to-end performance despite a 2.25-point blind conditional-accuracy drop. It retrieves more relevant but harder competing labels. The validation analytical ceiling was 42.27%; measured Phase-10 accuracy was 41.80%.

## E. Candidate representation ablation

With the current projection on CLINC150, raw machine labels reached 31.30% Recall@8, deterministic naturalization reached 34.87%, and the existing `intent:`-prefixed form reached 35.53%. Multi-schema training with naturalized labels reached 36.73%. Training with official SGD definitions plus hard negatives reached 42.63%, slightly below the 43.10% name-only hard-negative winner. No definitions were fabricated for Banking77, CLINC150, or MASSIVE.

## F. Hard-negative result

One offline refresh raised held-out Recall@8 from **36.73% to 43.10%** (+6.37 points). This was the largest controlled gain and selected B3. Additional refresh cycles were not attempted.

## G. Cardinality scaling

Deterministic CLINC150-train queries with the truth retained and distractors drawn from the combined CLINC150 plus training/development candidate pool:

| Candidates | Current Recall@8 | Phase-10 Recall@8 | Current MRR | Phase-10 MRR |
|---:|---:|---:|---:|---:|
| 8 | 100.0% | 100.0% | 0.608 | 0.673 |
| 16 | 84.87% | 87.20% | 0.479 | 0.559 |
| 32 | 63.70% | 71.97% | 0.368 | 0.448 |
| 50 | 51.30% | 61.93% | 0.307 | 0.382 |
| 77 | 42.07% | 52.43% | 0.256 | 0.328 |
| 100 | 37.53% | 47.80% | 0.227 | 0.295 |
| 150 | 31.47% | 41.73% | 0.189 | 0.254 |
| 250 | 25.20% | 33.93% | 0.150 | 0.210 |

Candidate encoding scales linearly (CPU p50: 448 ms at 50, 1.26 s at 150, 8.24 s at 1,000) but is cached. The projected 1,000-candidate cache is 0.512 MB.

## H. Generalization and robustness

| Held-out schema | Current Recall@8 | Phase-10 Recall@8 | Role |
|---|---:|---:|---|
| CLINC150 | 35.53% | **43.10%** | Schema-held-out validation |
| MASSIVE en-US | 28.51% | **34.97%** | One-shot blind |

On 600 CLINC development examples, state-only was best (42.83% Recall@8); state+question reached 31.33%, a structured `{"text": ...}` wrapper reached 36.83%, and question-only reached 4.33%. State-only therefore stayed frozen.

Robustness failed: on a fixed 300-example sample, a short irrelevant metadata suffix reduced Recall@8 from 42.67% to 14.00%; a long irrelevant suffix reduced it to 5.33%. This is a promotion blocker, not a minor slice issue.

## I. Artifact

- Retriever ID: `phase10-domain-general-k8-v1`
- Artifact: `artifacts/experiments/phase10/domain_general_k8.pt`
- SHA-256: `a82e369346d9ee3bbd9bc0df09bbebebea5c92c1a42b6c667b08675e918d713f`
- Architecture: frozen shared ModernBERT backbone, shared bias-free `1024 -> 128` projection, cosine similarity, exact top-K
- Objective: episodic InfoNCE cross-entropy plus one hard-negative refresh
- K: 8
- Feature flag: `retriever: phase10_domain_general`; default remains `current`

Fresh-process verification reproduced Recall@K exactly. Base-embedding minimum cosine versus the training cache was 0.99999976; maximum projected-score drift was 0.000372; top-8 candidate-set changes were 0/3,000. Candidate permutation produced 0 score-map changes in 256 trials.

## J. Regression

Only the opt-in experimental retriever, Phase-10 harnesses, data registry, tests, and research documentation changed. The scorer, calibration, automation policy, K, router cutoff, production retriever, and release manifest were not changed. The production manifest SHA-256 remains `c7febd238afa79144d907cdc4eb4bd229a0bca705c952352302772f8269d9648`.

Candidate text remains tokenizer data. Hostile candidate strings, cache invalidation, artifact/checkpoint compatibility, feature-flag isolation, and candidate-order permutation are directly tested.

## K. Promotion decision

**KEEP AS RESEARCH**

The artifact demonstrates real unseen-schema gains, but blind Recall@8 is only 34.97%, robustness fails under irrelevant/long state content, conditional scorer accuracy drops on harder blind shortlists, and SGD licensing needs review. The production release remains unchanged.

## L. Next bottleneck

**Noise-robust query encoding and truncation.** Retrieval misses remain 1,934 of 2,161 Phase-10 blind errors, and irrelevant/long state content causes catastrophic Recall@8 loss. Solve that single bottleneck before another retriever-capacity or scorer experiment.
