# Expert routing

Status: **routing policy, not a tuned production system**. No profile in `experts/registry.py` has `benchbrex_tuned: true`. No training sweep was executed. No checkpoint was promoted.

## Default path

Veya Core owns bounded choice, score, noul, defer, and the final policy gate. A general text task that is not one of those primitives selects Nemotron 3.5 Lightning, then GPT-OSS-20B, then defer. Private text stops at GPT-OSS-20B and does not fall through to a hosted model.

| Request | Selected expert | Fallback |
|---|---|---|
| Bounded decision | Veya Core | none |
| Ordinary text | Nemotron 3.5 Lightning | GPT-OSS-20B, defer |
| Private text | GPT-OSS-20B | defer |
| Fast image or multimodal | DeepSeek V4.1 Flash | GLM-5.3-Flash, defer |
| Deep multimodal | GLM-5.3-Flash | DeepSeek, defer |
| Max multimodal and frontier budget | Kimi-K3 | GLM, DeepSeek, defer |
| Max text planning and frontier budget | Nemotron Ultra | Lightning, GPT-OSS, defer |
| Quantum calibration plot | Ising 1.5 31B | defer |
| Safety classification | Safety Guard 8B v3 | defer |

Kimi and Ultra are not on the ordinary fallback chain. A general vision model is not a fallback for Ising. The safety model does not judge scientific or business truth.

Sampling values are the provider starting points from the task brief: Lightning and DeepSeek at temperature 1.0 and top-p 0.95, GPT-OSS at temperature 1.0 and top-p 1.0, Kimi at temperature 1.0 with top-p 0.95 or 1.0 on a max-tier escalation, Ultra at temperature 1.0 and top-p 0.95, Ising at temperature 0.2. They are not BenchBrex optima.

`POST /api/experts/route` returns this decision. It does not call a NIM.

## What was not run

The sealed CLINC150 aggregate is read from `artifacts/release/blind_clinc150_val_result.json` and labeled previously measured. Recall@8 on that file is 0.356, conditional accuracy is 0.980337, and end-to-end accuracy is 0.349. Recall@1, Recall@4, and Recall@16 are absent from that aggregate.

Phase 10 remains research. The report's cited suffix collapse, 42.67% to 14.00% to 5.33% Recall@8, is still the promotion blocker. The sweep planner in `experts/sweep.py` builds a staged grid and refuses a blind split, a K other than 8, a production-checkpoint write, and a single-seed result. It does not start `veya/train.py`.

The reference trainer still has `act_weight` 0. The loss term is not a supervised defer label. This change leaves that weight at 0.

Teacher rows require a human-accepted label stored separately from the teacher output. Disagreement raises review priority and keeps automation on defer.
