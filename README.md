<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/benchbrex-USA/veya/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/benchbrex-USA/veya/main/assets/logo-lockup.png" alt="Veya" width="330" />
  </picture>
</p>

**Non-autoregressive typed-decision research system.** The audited release path serves English Choice; Noul, categorical Score, the multilingual checkpoints, ordinal Score, and Phase 9 remain research-only.

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3?usp=sharing)
[![PyPI version](https://img.shields.io/pypi/v/veya.svg)](https://pypi.org/project/veya/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-veyadynamics%2Fveya-blue)](https://huggingface.co/veyadynamics/veya)
[![Multilingual](https://img.shields.io/badge/%F0%9F%A4%97%20Model-veya--multilingual-blue)](https://huggingface.co/veyadynamics/veya-multilingual)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-veya--demo-orange)](https://huggingface.co/spaces/veyadynamics/veya-demo)
[![Dev.to Article](https://img.shields.io/badge/dev.to-Read%20Article-0A0A0A?logo=devdotto&logoColor=white)](https://dev.to/BenchBrex/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
[![beguardit.com](https://img.shields.io/badge/beguardit.com-BenchBrex-FFDD00)](https://beguardit.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/benchbrex-USA/veya/main/assets/veya_vs_jev_full.png" alt="Veya versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>

## Audited release status

The Phase 1–9 forensic audit does **not** support a release-candidate or global production-ready claim. Current status:

| Scope | Status |
|---|---|
| English Choice | **RESEARCH RELEASE** |
| English Noul / categorical Score | **RESEARCH ONLY** |
| Multilingual Choice / Noul / Score | **RESEARCH ONLY** |
| Ordinal Score | **RESEARCH ONLY, not served** |
| Phase-9 hypothesis scorer | **RESEARCH ONLY, not served** |
| Automatic execution | **DISABLED** |

On the sealed cross-domain CLINC150 blind set (3,000 examples, 150 labels), the K=8 route reached 35.6% Recall@8, 98.03% conditional accuracy when the target was retrieved, and 34.9% end-to-end accuracy. The bottleneck is retrieval. End-to-end ECE was 0.267, so probabilities must not be treated as a production risk guarantee. See the [full audited verdict](docs/audit/release_verdict.md) and [Phase 1–9 matrix](docs/audit/phase_1_9_matrix.md).

Phase 10 trained an **experimental, non-served** multi-schema retriever. On the untouched MASSIVE English blind test it improved Recall@8 from 28.51% to 34.97% and end-to-end accuracy from 22.93% to 27.34%, but failed the robustness gate: irrelevant or long suffixes sharply reduced retrieval. The decision is **KEEP AS RESEARCH**; the audited production manifest still uses the original retriever. See the [Phase-10 report](docs/phase10/final_report.md).

The benchmark sections below are historical capability measurements, often on training-mix or development-exposed data. They are retained for reproducibility, not release certification.

Veya's base API evaluates typed questions (`choice`, `score`, `noul`) without text generation. Historical T4 measurements were 33 ms for one question and 7.2 ms/question batched. The audited high-cardinality route is multi-stage: learned retrieval followed by a packed rerank.

Three checkpoints, and a `Router` that picks between them per request:

| | encoder | params | context | audited status |
|---|---|---:|---:|---|
| [`veya`](https://huggingface.co/veyadynamics/veya) | ModernBERT-large | 421M | 512 | English Choice research release |
| [`veya-multilingual`](https://huggingface.co/veyadynamics/veya-multilingual) | mmBERT-base | 322M | 1024 | research only; release calibration unvalidated |
| [`veya-typed-decisions`](https://huggingface.co/veyadynamics/veya-typed-decisions) | ModernBERT-large | 421M | 1024 | research only; benchmark-specialized |

---

## Installation

```bash
pip install veya
```

Python 3.10 or newer. The dependencies set that floor: `huggingface_hub` 1.x, `transformers` 5.x and `torch` 2.14 all require 3.10.

---

## Quickstart: Route Mode (Experimental)

Veya ships three checkpoints. The built-in `Router` detects scripts/languages and dispatches between them, but the multilingual and typed-decisions checkpoints are not audited production routes. Use this API for research/evaluation, not as an implied multilingual guarantee.

```python
import veya
from veya import Router

# Preload checkpoints (high memory use; latency depends on hardware)
router = Router(preload=True)

# 1. State payload
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to veya (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # -> billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to veya-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi["answers"]["department"]["choice"])  # -> billing (confidence: 0.86)
print("Routing    :", res_hi["routing"]["model"])                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
```

Every result carries full routing metadata explaining why the choice was made:

```python
res_hi["routing"]
# {
#   'model': 'multilingual',
#   'repo': 'veyadynamics/veya/multilingual',
#   'reason': 'non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it'
# }
```

Inspect a routing decision without running any forward pass:

```python
router.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason
# "Latin script but language looks like 'de', not English"
```

### Why Route: The Evidence

On a shared benchmark (17,416 questions, one T4 GPU, identical questions per model):

| Benchmark / Task | English (`veya`) | Multilingual (`veya-multilingual`) | `Router` (Routed) |
|---|---|---|---|
| MASSIVE intent, English | **0.783** | 0.657 | **0.783** |
| MASSIVE intent, 13 other languages | 0.306 | **0.451** | **0.451** |
| XNLI, English | **0.860** | 0.843 | **0.860** |
| XNLI, 14 other languages | 0.521 | **0.731** | **0.731** |
| Languages usable (>3x random) | 23 / 51 | 45 / 51 | **45 / 51** |
| Latency, 1 question (T4 GPU) | 39.5 ms | **32.8 ms** | **32.8 ms** |
| Latency, 10 questions batched | 158.6 ms | **72.3 ms** | **72.3 ms** |

The English checkpoint collapses on non-Latin scripts (Khmer scores **0.000 accuracy at 0.952 confidence**). Because the model stays confident while being wrong, confidence gating cannot save you. `Router` detects the script in <0.5 ms pure Python before the forward pass.

### Production Preload & Memory

A cold checkpoint build costs seconds; language detection costs microseconds. At the default `max_loaded=1`, traffic that alternates languages rebuilds a model on *every* request (measured at a 7.4 s median reload on CPU and 10.3 s on T4).

For a server or production app, preload:

```python
# Every checkpoint resident in memory; language flips cost detection only (<1 ms)
router = Router(preload=True)
router = Router(preload=True, device="cuda")

# Or preload only the specific checkpoints you serve:
router.preload(["english", "multilingual"])

# If your app already built an agent, attach it to avoid duplicate VRAM:
router.attach("english", existing_agent)

# Manage resident memory (default keeps 1 hot, LRU eviction)
router = Router(max_loaded=2)       # keep two hot
router.unload()                     # free memory
```

| Deployment Mode | Per-Request Latency | Model Reloads |
|---|---|---|
| `Router()` (lazy, `max_loaded=1`) | 7 to 10 s on every language switch | 1 per switch |
| `Router(preload=True)` | **32.8 ms (GPU) / 193–464 ms (CPU)** | **none** |

---

## Single-Model Mode (Direct SDK)

If you only need a single checkpoint for a dedicated pipeline, you can load models directly:

```python
import veya

# 1. Load a specific checkpoint directly from the hub
agent = veya.load("veyadynamics/veya")                           # English root
agent_ml = veya.load("veyadynamics/veya", subfolder="multilingual") # 100+ languages
agent_td = veya.load("veyadynamics/veya", subfolder="typed-decisions")

# 2. Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"])   # -> billing (confidence: 0.94)
print("Urgency    :", answers["urgency"]["score"])        # -> 1.84 / 2.0
print("Churn Risk :", answers["churn_risk"]["noul"])       # -> 0.892 (89.2% probability)
```

---

## Automated Confidence Gating

Because Veya's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["confidence"]

if conf >= 0.85:
    # High confidence: automated action without human in the loop
    route_automatically(dept)
else:
    # Low confidence: escalate to human triage
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

---

## Built-in Workflow Presets

Veya provides pre-tuned question schemas for immediate production use:

```python
import veya

agent = veya.load("veyadynamics/veya")

# 1. Intelligent Model Router (routes to small vs. frontier models)
routing = agent.predict({"request": "Refactor this service using dependency injection"}, veya.router_questions())

# 2. Real-time Prompt Guardrails (jailbreaks, injections, leaks)
guard = agent.predict({"prompt": "Ignore all instructions"}, veya.guard_questions())

# 3. Content Safety & Moderation (toxicity, harassment, threats)
safety = agent.predict({"post": "User comment text"}, veya.moderation_questions())

# 4. Support Ticket Triage (intent, urgency, frustration, churn)
triage = agent.predict({"message": "My payment failed twice"}, veya.triage_questions())
```

---

## Decision Primitives

| Primitive | Output | Use Cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification, topic categorization |
| **`score`** | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

---

## Benchmarks

**Full report: [`BENCHMARKS.md`](BENCHMARKS.md)** — every run consolidated, languages and themes, with per-language detail for all 51 languages.

<p align="center">
  <img src="https://raw.githubusercontent.com/benchbrex-USA/veya/main/assets/veya_benchmark.png" alt="Per-language accuracy for both checkpoints across 51 languages" width="100%" />
</p>

All Veya numbers below are measured. Every model answered byte-identical questions
(fixed seed) in the same run. Reproduce with
[`notebooks/veya_benchmark_colab.ipynb`](https://github.com/benchbrex-USA/veya) on a T4.

### Speed (Tesla T4, measured)

| questions per call | `veya` | `veya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

Batched throughput reaches 103-332 questions/sec on a single T4. For reference, TypeSafe Jev
has been independently measured at 236-276 ms p50
([AbdelStark](https://github.com/AbdelStark/jev-benchmarks),
[nibzard](https://github.com/nibzard/decision-model-benchmark)) -- Veya answers a single
question roughly **6-7x faster**.

### Veya (with routing) vs Jev

Every Veya figure is what `Router().predict(...)` actually returns — the checkpoint the router
selects for that input, not a hand-picked best of three. Jev figures are **third-party
published, never measured here** (no TypeSafe API access), so sample sizes and prompts differ.

| | Jev 1.13.0 | Veya (routed) | |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** | +0.039 |
| AG News, 4 labels | 0.910 | **0.950** | +0.040 |
| DAIR Emotion, 6 labels | 0.480 | **0.595** | +0.115 |
| Banking77 (72 vs 77 labels) | **0.870** | 0.425 | Jev leads on >20 options |
| ECE *(lower better)* | 0.246 | **0.081** | 3× better (post-temperature) |
| p50 latency, 1 question | 236–276 ms | **32.8 ms** | 7.8× faster |
| Languages usable | *no published benchmark* | **45 of 51** | — |
| Weights | closed API | **Apache 2.0** | — |
| Cost | $0.042 / 1M tokens | **$0 self-hosted** | — |

On DAIR Emotion, Jev assigned **zero probability to the true label on 16% of examples** — a hard
failure for anything branching on confidence.

#### Where Jev leads

* **High-cardinality label spaces (20+ options):** The audited release routes 20+ candidates through the learned K=8 shortlist and keeps packed Choice through 19. This fixes packed token-budget collapse but does not solve cross-domain retrieval: sealed CLINC150 Recall@8 was only 35.6%, despite 98.03% conditional accuracy after retrieval. The public `predict_shortlist` helper remains available for caller-supplied retrieval, but its quality is the caller's responsibility.
* **Soft distribution matching:** On typed-decisions, while Veya achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
* **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). Veya achieves its 0.081 ECE after domain temperature fitting.

Full detail, including every workflow and all 51 languages: **[`BENCHMARKS.md`](BENCHMARKS.md)**.

### typed-decisions, measured on all three checkpoints

400 cases, 2,000 decisions, four workflows.

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`veya-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `veya` | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `veya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher self-agreement ceiling* | *0.735* | | | | |
| *per-question majority class* | *0.461* | | | | |
| *random guess* | *0.318* | | | | |

The fine-tuned checkpoint beats Jev by 3.9 points and clears the teacher ceiling, with 2.4x
better Brier and 1.6x better score MAE. It wins on all four workflows: invoice processing
0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730.
By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

Two places it still trails Jev: **soft accuracy** (0.471 vs 0.580 — its argmax is better but
its distributions match the teacher less well) and **ECE** (0.213 vs 0.144), which temperature
fitting addresses.

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461).
All of the capability on this benchmark comes from fine-tuning.

### Multilingual (51 languages, MASSIVE intent, 20 options, random = 0.050)

| | `veya` | `veya-multilingual` |
|---|---|---|
| English | **0.783** | 0.657 |
| 13 other languages | 0.306 | **0.451** |
| XNLI, English | **0.860** | 0.843 |
| XNLI, 14 other languages | 0.521 | **0.731** |

Across all 51 languages the English checkpoint macro-averages **0.227** with macro ECE
**0.733**, and only 23 of 51 languages clear 3x random. Khmer scores **0.000 at 95.2%
confidence**. This is why [`Router`](#model-routing-three-checkpoints-one-call) exists: the
model's own confidence gives no warning, so the routing decision has to be made before the
forward pass.

### English tasks

| task | `veya` | `veya-multilingual` | note |
|---|---|---|---|
| AG News | **0.947** | 0.937 | in training mix |
| BoolQ | **0.830** | 0.787 | in training mix |
| DAIR Emotion | **0.573** | 0.513 | held out |
| prompt-injections | **0.698** | 0.578 | held out, n=116 |
| SST-5 (ordinal) | 0.372 | 0.282 | held out |

### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`veya`) and
**0.314 -> 0.106** (`veya-multilingual`). `veya-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

### Honest limits

* **Not a release candidate:** the audited English Choice path is a research release. Noul, Score, multilingual routes, and Phase 9 are research-only.
* **Automation is disabled:** frozen 2%/5%/10% development thresholds observed 5.23%/9.48%/14.25% risk on the sealed blind set.
* **High-N retrieval is domain-specific:** Recall@8 was 35.6% on sealed CLINC150, making end-to-end accuracy 34.9% even though conditional reranking accuracy was 98.03%.
* **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. Veya is a fast base to
  specialise, not a zero-shot decision engine.
* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget (`max_len - head_max_len`):
  * `veya` (English) defaults to 512 context (`head_max_len = 192`, ~320 tokens for state).
  * `veya-multilingual` and `veya-typed-decisions` default to 1,024 context (`head_max_len = 256`, ~768 tokens for state; mmBERT-base encoder supports up to 8,192 with RoPE).
  At default settings, a 77-option question like Banking77 allocates only `(256 - 16) // 77` ≈ 3–4 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise `agent.cfg["head_max_len"] = 512` and `agent.cfg["max_len"] = 1024` (or up to 2048 / 4096 / 8192) so every option has enough tokens to remain distinct.
  2. Or shortlist with embeddings and run one forward pass on the top `k` labels (`predict_shortlist`, example below). `predict` and `system_one` still score every criterion they are given.
  3. Or split the label set yourself into a coarse question and a fine question.

```python
import veya

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which banking intent is this?",
        "criteria": {
            "card_arrival": "where is my card",
            "transfer_fee": "fee charged on a transfer",
            # ...the rest of a large label set
        },
    }
}
result = veya.predict_shortlist(
    agent,
    {"text": "I was charged twice for a transfer"},
    questions,
    embed_fn=veya.embed_fn_from_agent(agent),  # or any callable: texts -> (n, dim)
    k=20,
)
result["shortlist"]["intent"]["labels"]  # the top 20 labels sent to the model
```

`embed_fn(texts)` returns one vector per string. `embed_fn_from_agent` mean-pools the encoder already loaded on the agent; the decision head runs in the following `predict` / `system_one` call. Probabilities on a shortlisted choice are over those `k` labels. When `k` is at least the number of labels, the original question is passed through and `embed_fn` is not called.

[Issue #102](https://github.com/benchbrex-USA/veya/issues/102) reports that a top-20 zero-shot shortlist moved a BANKING77 run from 54.3% to 60.8% on the reporter's setup. Those figures are the reporter's; this repository has not remeasured them.

* Served `score` behavior is categorical. The separate ordinal head is **RESEARCH ONLY** and is not loaded by the audited manifest.
* `veya` collapses outside English; `veya-multilingual` is weaker on English. Route, or pick
  deliberately.

---

## Live Demo & Resources

* **Hugging Face Model:** [veyadynamics/veya](https://huggingface.co/veyadynamics/veya)
* **Interactive Web Demo:** [veyadynamics/veya-demo](https://huggingface.co/spaces/veyadynamics/veya-demo)
* **Engineering Writeup:** [Read the full story on Dev.to](https://dev.to/BenchBrex/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

---

## Fine-Tuning

Fine-tune Veya on your own domain data. The training math lives in one place --
[`veya/train.py`](veya/train.py) -- so the notebook, the reference single-process
trainer, and the shipped runtime cannot drift apart:

* **[`notebooks/veya_finetune_typed_decisions_2xT4_kaggle.ipynb`](notebooks/veya_finetune_typed_decisions_2xT4_kaggle.ipynb)**
  runs the whole loop on Kaggle's free 2xT4 GPUs (DDP via `torchrun`): build the
  dataset, train with RLCD (proper-scoring-rule rewards, GRPO-style policy
  gradient + soft cross-entropy distillation), fit calibration temperatures,
  evaluate, and push the result to the Hub. It imports the reward, the
  calibration fit, and the held-out split from `veya.train`.
* **`veya.train.run_finetune`** is the tested single-process reference path
  (CPU / MPS / one GPU) using the identical objective.

Three guarantees are baked into `veya/train.py`:

* **Calibration matches serving.** `fit_temperature_soft` clamps to
  `[TEMP_MIN, TEMP_MAX]` -- the same band `veya.common.clamp_temperature` applies
  at load -- so the temperature the trainer reports is the one the runtime uses.
* **Calibration is held out.** `split_items` fits temperature on rows the model
  never trained on, so the reported ECE is honest.
* **Reproducible.** `set_global_seed` seeds `random`, `numpy`, and `torch`
  (CPU + CUDA) per rank.

The objective exposes the **CE-only vs CE+PG ablation** as config flags
(`rl_weight` / `ce_weight`), and `act_weight` decides whether the defer/escalate
head receives gradient (default `0.0`, matching the historical recipe). Streaming
the HF `typed-decisions` dataset needs the extra: `pip install 'veya[train]'`.

To run the **CE-only vs CE+PG ablation** on a single device, flip the weights and
compare the held-out calibration each run reports:

```python
from veya.train import TrainConfig, load_typed_decisions_items, run_finetune

items = load_typed_decisions_items("models/production")  # needs `veya[train]`

# CE-only (pure distillation): rl_weight=0
run_finetune("models/production", "out/ce_only", items,
             TrainConfig(rl_weight=0.0, ce_weight=1.0))

# CE + policy gradient (full RLCD): both non-zero
run_finetune("models/production", "out/ce_pg", items,
             TrainConfig(rl_weight=1.0, ce_weight=1.0))

# Train the defer/escalate head too: act_weight > 0
run_finetune("models/production", "out/ce_pg_act", items,
             TrainConfig(rl_weight=1.0, ce_weight=1.0, act_weight=0.1))
```

For the multi-GPU recipe the Kaggle notebook wraps the same `compute_rcld_loss`
and `fit_temperature_soft` in `torchrun --nproc_per_node=2`, so the objective is
identical to the single-process path above.

Fine-tuning is where most of the value is. On the typed-decisions benchmark the base
checkpoints score near chance zero-shot (0.36 and 0.35 against a 0.318 random baseline),
while the fine-tuned checkpoint reaches **0.766** on the same 2,000 decisions -- above
TypeSafe Jev's published 0.727 and above the 0.735 teacher self-agreement ceiling. Treat Veya
as a fast base to specialise, not as a zero-shot decision engine.

Runtime scales with epochs and world size: the shipped checkpoint records a
1-epoch, single-GPU run at ~2 hours, while the full 4-epoch 2xT4 recipe over
~30k questions takes roughly 4-5 hours.

---

## Support the Project

If Veya helps your research or products, consider supporting independent research:

<p align="left">
  <a href="https://beguardit.com" target="_blank">beguardit.com</a>
</p>

---

## License

Apache 2.0. Developed by Veya Dynamics.
