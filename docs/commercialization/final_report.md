# Veya Commercialization Program — Gate Report

## A. Commercial verdict

**RESEARCH — do not promote to Developer Preview, Paid Beta, or GA.**

This increment materially improved retrieval robustness and reproduced a clean
unseen-schema gain on a second external dataset. Absolute quality remains far
below a credible commercial route. Production release
`0.3.5-audit-baseline.1`, its manifest, scorer, calibration, automation policy,
and default retriever remain unchanged.

## B. Model

### One-shot HWU64, 3,000 rows, 60 candidates

| Route | Clean Recall@8 | Suffix Recall@8 | Prefix Recall@8 |
|---|---:|---:|---:|
| Audited current | 27.30% | 10.87% | 11.33% |
| Phase 10, full state | 34.07% | 14.23% | 18.67% |
| Phase 10, max view | **34.07%** | **27.83%** | **29.77%** |

Phase 10 improved clean Recall@8 by **6.77 points** over current (paired
bootstrap 95% CI **+4.8 to +8.8**). The frozen max-view policy improved suffix
Recall@8 by **13.60 points** (CI **+12.0 to +15.3**) and prefix Recall@8 by
**11.10 points** (CI **+9.4 to +12.8**) versus Phase-10 full-state retrieval.

### Frozen-scorer subset, 500 rows

| Route / condition | Recall@8 | Conditional accuracy | End-to-end | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Current / clean | 28.60% | 79.72% | 22.80% | 19.912 | 0.986 | 0.260 |
| Phase 10 / clean | 35.40% | 76.84% | 27.20% | 18.122 | 0.961 | 0.257 |
| Phase 10 / suffix | 12.60% | 49.21% | 6.20% | 24.346 | 1.125 | 0.254 |
| Max view / suffix | 29.00% | 53.79% | 15.60% | 20.015 | 1.031 | 0.212 |
| Phase 10 / prefix | 18.00% | 62.22% | 11.20% | 22.879 | 1.128 | 0.313 |
| Max view / prefix | 31.60% | 67.09% | 21.20% | 19.205 | 0.994 | 0.252 |

NLL treats a retrieval miss as zero target probability. ECE uses the frozen
temperature descriptively; it is not evidence that this route is calibrated.
Risk-coverage was not refit because the route failed the quality gate.

### Development robustness

On the original frozen 300-row sample, max-view retrieval preserved clean
Recall@8 at 42.67%, raised short-suffix recall from 14.00% to 40.67%, long-suffix
recall from 5.33% to 41.33%, and prefix recall from 15.67% to 41.33%.

## C. Performance

Local Apple ARM CPU, cached 150-candidate vectors:

| Query path | p50 | p95 | p99 |
|---|---:|---:|---:|
| Phase-10 single full view | 70.0 ms | 77.5 ms | 78.6 ms |
| Frozen max-view policy | 108.2 ms | 129.8 ms | 135.0 ms |

The robust path costs approximately 54% p50 latency on the tested suffix
workload. Exact candidate scoring is still negligible; encoding extra query
views is the cost. Throughput, production-instance memory, availability, cold
start, and fully loaded cost per 1K decisions are **not measured**. The scorer's
uncached clean p50 was about 138 ms on the same machine, but stage measurements
must not be presented as a production SLO.

## D. Competitor benchmark

No external competitor claim was run. There are no credentials, frozen
customer-representative workload, same-region network setup, or current price
ledger. The audited Veya retriever is the only fair internal baseline. COM-G2
remains ready, not complete.

## E. Product

- Added explicit experimental mode `phase10_domain_general_multiview`.
- Added bounded deterministic views and first/max/mean/RRF fusion.
- Frozen winner: `max`, at most four views, at most 4,096 characters per view.
- Existing research API/gateway remains available; no commercial API maturity
  promotion occurred.
- Python/TypeScript commercial SDKs, idempotency, org/project primitives, and
  customer model pinning are not implemented as commercial contracts.

## F. Revenue infrastructure

Metering, pricing, billing, reconciliation, spend controls, and margin evidence
are not implemented. Building them remains gated by model quality and a real
workload cost model.

## G. Customer evidence

None. Support routing and operations exception classification are working wedge
hypotheses only. There are no design-partner baselines, shadow results, ROI
measurements, or customer references.

## H. Moat

| Moat | Evidence | Status |
|---|---|---|
| Quality | Two external blind-like schemas show ~34% Phase-10 Recall@8 | Real gain, insufficient absolute quality |
| Reliability | Aggregate NLL/Brier/ECE recorded; forced-answer behavior exposed | Not validated |
| Systems | Exact search and candidate caching are cheap; robust query latency measured | Research only |
| Data | Versioned multi-schema lineage and hard-negative corpus | Licensing blocks commercial use |
| Evaluation | Frozen development IDs, pre-download blind freeze, aggregate reports | Public data; four exact overlaps |
| Workflow | Workload contract drafted | No customer validation |
| Trust | Immutable manifest, artifact hashes, opt-in flag, rollback path | No tenancy/enterprise program |

## I. Security / enterprise

Query and candidate strings remain tokenizer-only untrusted data. View counts
and lengths are bounded, candidate caches remain versioned, and the feature is
opt-in. Existing server security tests remain applicable. Organization
isolation, scoped key storage/rotation, audit logs, SSO/RBAC, retention/deletion,
private networking, and formal security review are outstanding.

## J. Regressions

- Robust query p50 increased from 70.0 ms to 108.2 ms on the local CPU test.
- Clean quality is unchanged by max-view fusion, not improved.
- Noisy blind recall remains 4.3–6.2 points below clean recall.
- The harder clean Phase-10 shortlist reduced conditional scorer accuracy from
  79.72% to 76.84% on the HWU64 subset.
- HWU64 contains four exact text overlaps with Phase-10 data; this prevents a
  contamination-free claim. Base-model pretraining exposure is also unknown.
- The frozen source filter admits exact status `IRR` because the pre-download
  protocol followed the upstream `IRR_` wording literally; 53 such rows occur
  in the 3,000-row sample.

## K. Biggest remaining bottleneck

**Absolute unseen-schema retrieval quality.**

Noise handling is materially better, but clean Recall@8 remains 34.07% on
HWU64 and 34.97% on MASSIVE. Retrieval still misses roughly two thirds of
targets before the scorer can act.

## L. Next graph

1. Train deterministic prefix/suffix/structured-noise augmentation on TRAIN
   data only.
2. Run a second hard-negative refresh with duplicate/equivalence audits.
3. Re-test licensed authoritative candidate semantics.
4. Freeze and evaluate on schema-held-out development.
5. If dense retrieval does not materially raise clean and noisy Recall@8,
   declare the plateau and compare one bounded late-interaction reranker over
   dense top-32/top-64.
6. Identify another blind source only after architecture and thresholds freeze.

