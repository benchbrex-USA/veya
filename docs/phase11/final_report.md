# Phase 11 Final Report

## A. Core result

No: the development-selected noise-trained retriever did not become robust enough to irrelevant or long state content. RB1 regressed full clean schema-held-out Recall@8 from 43.10% to 39.80%, robust mean Recall@8 from 13.35% to 11.10%, and matched frozen-scorer end-to-end accuracy from 43.50% to 36.50%.

The experiment did produce two credible dense leads. RB2, the exactly-once second hard-negative refresh, improved robust mean Recall@8 by 5.05 points versus Phase 10 (paired 95% CI +3.15 to +7.00) while keeping clean Recall@8 within 0.37 points. Generic semantic-field extraction improved JSON-noise Recall@8 from 11.00% to 38.50% (delta +27.50 points, CI +20.50 to +34.50). Both are post-selection evidence and require a newly frozen confirmation split.

## B. Clean performance

CLINC150 train was used as a whole schema held out from retriever training. It is development evidence, not a newly pristine blind set.

| Retriever | Recall@8 | MRR | E2E | Conditional accuracy |
|---|---:|---:|---:|---:|
| RB0 Phase 10 | 43.10% | 0.2707 | 43.50% | 97.75% |
| RB1 + noise | 39.80% | 0.2451 | 36.50% | 93.59% |
| RB1I + invariance | 41.63% | 0.2616 | 44.00% | 95.65% |
| RB2 + second hard negatives | 42.73% | 0.2763 | 44.50% | 96.74% |

The clean-quality floor was Phase-10 minus one point (42.10%). RB1 and RB1I fail it; RB2 passes it.

## C. Noise robustness

Recall@8 on the deterministic 200-row schema-held-out noise matrix:

| Retriever | Clean | Short prefix | Short suffix | Long prefix | Long suffix | Middle noise |
|---|---:|---:|---:|---:|---:|---:|
| RB0 Phase 10 | 44.00% | 15.00% | 14.00% | 6.50% | 5.50% | 11.00% |
| RB1 + noise | 40.50% | 15.00% | 10.50% | 8.00% | 9.00% | 8.50% |
| RB1I + invariance | 42.00% | 20.50% | 12.00% | 7.00% | 7.50% | 11.00% |
| RB2 + second hard negatives | 42.00% | 28.00% | 23.00% | 9.50% | 10.00% | 19.00% |

RB2's mean over all ten noisy conditions was 18.40%, versus 13.35% for RB0. Absolute quality remains weak.

## D. Length robustness

The 96-token single-view path and adaptive path were measured on all 2,200 noisy sample records.

| Token-length bucket | n | Single-view Recall@8 | Adaptive Recall@8 |
|---|---:|---:|---:|
| 0-32 | 1,121 | 15.88% | 15.88% |
| 33-64 | 479 | 15.45% | 15.45% |
| 257+ | 600 | 8.50% | 7.83% |

Long-prefix inputs dropped 558 tokens on average and removed the useful span from the full right-truncated view in 100% of records. Adaptive segmentation put the useful span fully inside at least one view in 100% of long-prefix, long-suffix, and repeated-text records, but max/mean/RRF fusion still failed to convert that coverage into aggregate retrieval gains. The remaining problem is therefore not only span coverage; representation and score comparability across noisy views are limiting.

## E. Training ablation

| Ablation | Full clean Recall@8 | Noisy mean Recall@8 | Delta vs RB0, paired 95% CI |
|---|---:|---:|---:|
| RB0 baseline | 43.10% | 13.35% | control |
| RB1 + noise augmentation | 39.80% | 11.10% | -2.25 points, [-4.00, -0.45] |
| RB2 + second hard-negative refresh | 42.73% | 18.40% | +5.05 points, [+3.15, +7.00] |
| RB1I + invariance loss λ=0.05 | 41.63% | 12.15% | -1.20 points, [-3.05, +0.75] |

The refresh mined 37,080 negatives from 4,635 TRAIN rows. It rejected 540 normalized duplicates and 491 base-embedding near-duplicates at cosine ≥0.92. Its lineage digest is `69bacf1f827fbf189256c27a92f7dac74a609ff9426399bd0edfd341f770b63b`.

## F. Query strategy

Quality is mean Recall@8 over clean, short prefix/suffix, long prefix/suffix, and middle insertion.

| Strategy | Recall@8 | Mean views/query | Decision |
|---|---:|---:|---|
| Q0 single | 15.25% | 1.00 | Pareto choice |
| Q1 first + last | 15.00% | 1.33 | reject |
| Q2 first + middle + last | 15.00% | 1.64 | reject |
| Q3 four-view | 15.00% | 1.98 | reject |
| Q4 adaptive | 15.00% | 1.94 | reject |

For adaptive views, mean fusion matched max exactly; RRF was 0.08 points worse. Neither interval excluded zero. Generic structured extraction is the exception: on JSON noise it recovered 38.50% Recall@8 versus 11.00% for all-fields single-view encoding.

## G. Dense plateau verdict

**DENSE STILL IMPROVING**

RB2 and generic semantic-field extraction both materially improved held-out robustness. The Phase-11 rule therefore prohibits declaring a dense plateau even though noise-only training, invariance loss, generic multi-view segmentation, and fusion changes failed.

## H. Late interaction

Not executed. The late-interaction gate did not open because at least one dense intervention materially improved held-out robustness. No top-32/64 MaxSim path, token cache, or late-interaction latency claim was created.

## I. End-to-end

Matched n=200, K=8, frozen base decision head, and frozen temperature 1.971127355603397:

| Retriever | Recall@8 | Conditional accuracy | E2E | Mean scorer margin on hits |
|---|---:|---:|---:|---:|
| RB0 Phase 10 | 44.50% | 97.75% | 43.50% | 0.7422 |
| RB1 + noise | 39.00% | 93.59% | 36.50% | 0.7602 |
| RB1I + invariance | 46.00% | 95.65% | 44.00% | 0.7387 |
| RB2 + second hard negatives | 46.00% | 96.74% | 44.50% | 0.7825 |

RB1's paired E2E delta versus RB0 was -7.0 points, CI [-12.5, -1.0]. RB2's +1.0 point interval [-4.0, +6.0] includes zero.

## J. Latency

Final CPU run, n=30 mixed clean/long-prefix/long-suffix queries, 150 cached candidates:

| Strategy | p50 | p95 | p99 | Throughput | Mean views | Cache memory |
|---|---:|---:|---:|---:|---:|---:|
| Q0 single | 194.6 ms | 212.3 ms | 218.6 ms | 5.72 q/s | 1.00 | 75 KiB |
| Q1 two-view | 272.6 ms | 289.4 ms | 295.1 ms | 4.42 q/s | 1.67 | 75 KiB |
| Q2 three-view | 353.9 ms | 372.5 ms | 396.0 ms | 3.54 q/s | 2.30 | 75 KiB |
| Q3 four-view | 509.0 ms | 564.1 ms | 598.8 ms | 2.52 q/s | 2.97 | 75 KiB |
| Q4 adaptive | 438.8 ms | 464.8 ms | 467.1 ms | 2.94 q/s | 2.90 | 75 KiB |

The shared projection is 512 KiB. Projected candidate caches are 75 KiB at 150, 500 KiB at 1,000, and 4.88 MiB at 10,000 candidates. Exact ranking is not the cost driver; repeated frozen-encoder calls are.

## K. Artifact

- Retriever ID: `phase11-noise-robust-k8-v1`
- Development-selected variant: `RB1_noise`
- Artifact SHA-256: `197b896558e1d53a931b6d62f913c79c17795efff2f164313cc9c95ac8d21d4f`
- Frozen backbone checkpoint: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
- Noise version: `phase11-generic-noise-v1`
- Hard-negative version: `phase11-second-refresh-v1` (evaluated, not selected into RB1)
- K: 8; shared 1024→128 projection; naturalized-label candidates; state-only control

Fresh double-load produced zero query/candidate embedding delta, zero score delta, zero rank or top-8 changes, and zero score-map changes over 32 candidate permutations.

## L. External evaluation integrity

No consumed blind or external test was opened: MASSIVE, HWU64, CLINC150 validation/test, and Banking77 test were not used. Evidence is from the frozen CLINC150 train partition as a whole schema excluded from training, split hash `8d56603620ae95aa7ab5ce312bab6775f22821d830b66dc510bfc4325c9cd996`. It is not claimed as pristine blind evidence. No new external source was introduced, so no new overlap result exists.

Production manifest hash `c7febd238afa79144d907cdc4eb4bd229a0bca705c952352302772f8269d9648`, K, scorer, calibration, router cutoff, automation-disabled policy, and serving default remain unchanged.

## M. Promotion decision

**KEEP AS RESEARCH**

RB1 fails clean, robustness, E2E, and latency gates. RB2 and structured extraction are promising post-selection findings, not a releasable winner. The repository-wide regression gate is additionally blocked by an unrelated concurrent Phase-12 Python 3.11 syntax error.

## N. Next bottleneck

**representation/backbone capacity**

The encoder loses useful content under right truncation, but even views that fully retain the useful span remain weak under noise. Candidate caches and exact ranking are already cheap; stacking more view heuristics is not the next move.
