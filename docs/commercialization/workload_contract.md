# Commercial Workload Contract

The initial wedge is **high-volume structured routing**: support routing and
operations exception classification. This is a hypothesis, not customer
validation.

Each future pilot must freeze the following before evaluation:

```yaml
workflow_id:
owner:
decision_unit:
candidate_schema_version:
candidate_count_distribution:
state_shape:
state_length_distribution:
language:
monthly_volume:
latency_budget_ms:
error_budget:
manual_review_cost:
false_positive_cost:
false_negative_cost:
current_system:
current_quality:
current_latency:
current_cost_per_1k:
data_permission:
retention_policy:
success_metric:
shadow_only: true
```

No workflow may be described as commercially validated until its owner,
volume, baseline, error budget, and measured business outcome are present.
Synthetic and public benchmark workloads remain internal evidence only.

## Benchmark layers

1. **Public:** reproducible datasets; development exposure is labeled.
2. **Private:** rotating, aggregate-only workflow cases; no per-case tuning.
3. **Customer shadow:** permissioned traffic that cannot affect production
   decisions during evaluation.

All systems in a comparison receive the same state, candidate semantics,
schema, examples, and success criterion. External results additionally record
provider, model/version, date, region, request parameters, wall latency, and
the price source used. Missing external credentials block that adapter only;
they do not block Veya research.

