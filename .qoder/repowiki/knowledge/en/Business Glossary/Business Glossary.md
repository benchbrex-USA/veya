---
kind: business_term
name: Business Glossary
category: business_term
scope:
    - '**'
---

### RLCD
- Definition：Reinforcement Learning via Calibrated Distillation — the policy-gradient training method that combines a soft cross-entropy distillation loss with a reward term built from strictly proper scoring rules (log score, spherical score, ranked probability score), using GRPO-style group-mean advantage normalization.

### System 1
- Definition：The non-autoregressive decision paradigm Veya implements: answering typed questions in a single encoder forward pass rather than generating text token-by-token, yielding ~33 ms per question on a T4.

### choice
- Definition：One of Veya's three decision primitives: pick one label from N labeled options and return probabilities over all options plus a confidence score.

### score
- Definition：One of Veya's decision primitives: output an expected level on an ordinal rubric together with a full distribution and confidence; the served path uses categorical scoring, while the separate ordinal head is research-only.

### noul
- Definition：One of Veya's decision primitives: output a calibrated P(true) in [0, 1] for a yes/no proposition (used for phishing/spam/jailbreak/churn detection).

### typed decisions
- Definition：The benchmark dataset and fine-tuning target consisting of ~1,200 cases expanded to ~6,000 typed decisions, where each row carries a gold teacher probability distribution used as soft targets.
- Aliases：LocalLLaMA/typed-decisions

### shortlist
- Definition：The learned K=8 retrieval stage that narrows high-cardinality choice questions (20+ options) before the packed rerank, intended to fix token-budget collapse across domains.
- Aliases：K=8 route、learned shortlist

### Phase 9 / Phase 10 / Phase 11 / Phase 12
- Definition：Named research stages in `research/veyabench/` covering hypothesis scoring, multi-schema contrastive retriever training, noise robustness, backbone capacity, and context experiments respectively.
- Aliases：phase9、phase10、phase11、phase12

### release-candidate claim
- Definition：The audited production-readiness assertion that the README explicitly disavows: the English Choice path is a research release, automation is disabled, and base checkpoints are near chance zero-shot.

### automated confidence gating
- Definition：The proposed production workflow that routes low-confidence answers to human triage and high-confidence ones to automated action; currently disabled because frozen thresholds observed unacceptable risk on the sealed blind set.
- Aliases：confidence gating、auto-routing
