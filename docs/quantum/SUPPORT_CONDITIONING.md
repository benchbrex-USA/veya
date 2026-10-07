# Research note: Veya support-example conditioning

Status: **not implemented**. This note does not change the production encoder, the decision heads, or the release manifest. It is a proposal for a later benchmark.

## Question

High-cardinality Choice already retrieves a shortlist and then reranks. A separate question is whether a frozen Veya encoder can condition on a few verified calibration demonstrations without becoming autoregressive and without writing those demonstrations into the option string. Phase 8 showed that packing extra label text into the option string hurt accuracy. This proposal keeps the demonstrations out of the packed option text.

## Sketch

The query and each support item would be encoded separately.

```text
h_query = Encoder(query state)

h_support_i = Encoder(
    support state +
    support question +
    support answer
)

h_context = attention(
    query = h_query,
    keys = support embeddings,
    values = support embeddings
)

h_final = h_query + gate * h_context
```

`h_final` would then enter the existing choice, score, and noul heads. The gate would be a learned scalar or vector, initialized so the support path starts as a no-op. The encoder would stay frozen in the first experiment. No new token decoder would be added.

Support items would be the same bounded, project-scoped records the quantum retrieval path already isolates: experiment family, device type, verified diagnosis, verified next action, and outcome. K would stay in 0..8. A support item from another project would not be readable.

## What would have to be measured

Compare, on a grouped holdout (unseen device or unseen experiment family, not a random row split):

- packed Veya with no support
- packed Veya with retrieved support through this fusion
- the Ising-then-Veya pipeline in `ARCHITECTURE.md`

Report accuracy, Brier, ECE, defer rate, and selective risk. Do not promote the fusion on a development-exposed slice. Do not train on the frozen blind split. A gain that appears only when the true label is forced into the support set is not a result.

## Non-goals

- Replacing ModernBERT.
- Feeding Ising prose in as a support answer unless a human has verified it.
- Using generative confidence as the gate.
- Shipping this inside the audited Choice route before a benchmark exists.
