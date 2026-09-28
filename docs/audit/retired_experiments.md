# Retired and research-only experiment registry

These paths must not be reintroduced without a materially new hypothesis and clean validation data.

| Approach | Status | Evidence | Why it is not served |
|---|---|---|---|
| Frozen cosine shortlist | **RETIRED** | Phase 4 route comparisons | Learned K=8 projection dominated on the matched Banking77 development slice |
| K=16 shortlist + packed rerank | **RETIRED** | `artifacts/experiments/phase4/banking77_routes.json` | Lower matched end-to-end accuracy than K=8 |
| K=32 shortlist + packed rerank | **RETIRED** | Phase 4/6 results | Packed reranker degraded as candidate count increased |
| Frozen candidate-set scorer | **RETIRED** | `artifacts/decisions/candidate_set.json` | 0.531 matched accuracy versus 0.688 for K=8 packed rerank |
| Head-only listwise reranker | **RETIRED** | `artifacts/experiments/phase6/train_decision.json` | Paired 95% interval included zero |
| Pairwise auxiliary head | **RETIRED** | `artifacts/experiments/phase6/train_decision_pairwise.json` | Paired 95% interval did not establish a positive gain |
| Mean-pooled interaction scorer | **RETIRED** | `artifacts/experiments/phase6/interaction_decision.json` | Conditional accuracy collapsed to 0.224 |
| Top-layer packed-head adaptation | **RETIRED AS TESTED** | `artifacts/experiments/phase7/full_run.json`, `parameter_update_audit.json` | Parameters changed, but 0/597 validation decisions improved |
| Packed semantic enrichment (D1/D2) | **RETIRED** | `artifacts/experiments/phase8/eval.json` | Conditional accuracy fell from 0.697 to 0.325/0.332 and order sensitivity increased |
| Ordinal Score head | **RESEARCH ONLY** | `artifacts/decisions/score_architecture.json` | No suitable ordinal research table exists; artifacts marked production are refused |
| Phase-9 independent hypothesis scorer | **RESEARCH ONLY** | `artifacts/experiments/phase9/report.json`, `clean_load.json` | Reproducible development gain, but release evidence is exposed and adversarial sensitivity remains |

The highest-value next experiment is not another reranker. Fresh blind conditional accuracy is 98.03% once the target reaches K=8; retrieval Recall@8 is only 35.6%.
