# Veya — Master Reference

> **Fast, non-autoregressive System 1 decision engine with calibrated probabilities.**

| Field | Value |
|---|---|
| **Name** | `veya` |
| **Version** | `0.3.5` (`__version__`) |
| **License** | Proprietary (all rights reserved) |
| **Author** | Veya Dynamics |
| **Python** | `>=3.10` (classifiers: 3.10 / 3.11 / 3.12 / 3.13) |
| **Build** | `setuptools>=61.0` (backend `setuptools.build_meta`) |

**Keywords:** `decision-model` · `rlcd` · `calibration` · `system-one` · `routing` · `guardrails` · `moderation` · `triage`
**Links:** Homepage — `https://huggingface.co/veyadynamics/veya` · Demo Space — `https://huggingface.co/spaces/veyadynamics/veya-demo`

---

## Table of Contents

- [3. Project Overview](#3-project-overview)
- [4. Installation](#4-installation)
- [5. Core Library Reference (`veya/`)](#5-core-library-reference-veya)
  - [5.1 Public API Surface](#51-public-api-surface-veya__init__py) · [5.2 Core Architecture](#52-core-architecture--commonpy) · [5.3 Agent Runtime](#53-agent-runtime--agentpy) · [5.4 Calibration System](#54-calibration-system--calibrationpy) · [5.5 Email Utilities](#55-email-utilities--emailpy) · [5.6 Language Detection](#56-language-detection--langpy) · [5.7 Manifest Builder](#57-manifest-builder--manifestpy) · [5.8 Preset Question Schemas](#58-preset-question-schemas--presetspy) · [5.9 Release Bundle](#59-release-bundle--release_bundlepy) · [5.10 Multi-Checkpoint Router](#510-multi-checkpoint-router--routerpy) · [5.11 Selective Automation](#511-selective-automation--selectivepy) · [5.12 Model Server](#512-model-server--serverpy) · [5.13 Embedding Shortlist](#513-embedding-shortlist--shortlistpy) · [5.14 Shortlist Artifact](#514-shortlist-artifact--shortlist_artifactpy) · [5.15 K=8 Packed Reranker](#515-k8-packed-reranker--rerankerpy) · [5.16 RLCD Training](#516-rlcd-training--trainpy)
- [6. Experimental Modules (`veya/experimental/`)](#6-experimental-modules-veyaexperimental)
- [7. Web Gateway (`web/`)](#7-web-gateway-web)
  - [7.1 Gateway Server](#71-gateway-server--webserverpy) · [7.2 Model Server Entry](#72-model-server-entry--webmodel_serverpy) · [7.3 Request/Response Contracts](#73-requestresponse-contracts--webcontractspy)
- [8. Models and Production Artifacts](#8-models-and-production-artifacts)
  - [8.1 Production Models](#81-production-models-modelsproduction) · [8.2 Release Artifacts](#82-release-artifacts-artifactsrelease) · [8.3 Decision Records](#83-decision-records-artifactsdecisions) · [8.4 Experiment Phases](#84-experiment-phases-artifactsexperiments) · [8.5 Commercialization Artifacts](#85-commercialization-artifacts-artifactscommercialization)
- [9. Architecture](#9-architecture)
  - [9.1 System Architecture](#91-system-architecture) · [9.2 Two-Process Serving Model](#92-two-process-serving-model) · [9.3 Fail-Closed Integrity](#93-fail-closed-integrity) · [9.4 Selective Automation](#94-selective-automation) · [9.5 Concurrency Model](#95-concurrency-model)
- [10. Infrastructure](#10-infrastructure)
  - [10.1 Docker](#101-docker) · [10.2 Kubernetes](#102-kubernetes) · [10.3 CI/CD](#103-cicd)
- [11. Testing](#11-testing)
- [12. Research Module (`research/`)](#12-research-module-research)
- [13. Benchmarks and Evaluation Results](#13-benchmarks-and-evaluation-results)
  - [13.1 Typed-Decisions](#131-typed-decisions-benchmark-400-cases-2000-decisions) · [13.2 Public Datasets](#132-public-datasets) · [13.3 Speed](#133-speed-tesla-t4) · [13.4 Calibration](#134-calibration) · [13.5 Multilingual](#135-multilingual-51-language-massive-sweep) · [13.6 Application Workflows](#136-application-workflows-400-cases-each) · [13.7 Known Limitations](#137-known-limitations)
- [14. Operational Runbook (Summary)](#14-operational-runbook-summary)
- [15. Production Readiness](#15-production-readiness)
- [16. Shadow Deployment](#16-shadow-deployment)
- [17. Prospective Blind Workflow](#17-prospective-blind-workflow)
- [18. Documentation Index](#18-documentation-index)

---

## 3. Project Overview

**Description** (from `pyproject.toml`): *“Fast, non-autoregressive System 1 decision engine with calibrated probabilities.”*

Veya is a bidirectional-encoder decision model: one parallel forward pass answers typed questions about a state (text/JSON) with calibrated probabilities and a defer/answer action — no autoregressive generation. It ships as a PyPI package (`veya`), a two-process audited serving stack (gateway + model service), and a research evaluation module (`research/veyabench`).

### Model checkpoints

| Checkpoint | Backbone | Params | Context |
|---|---|---|---|
| `veya` (English) | ModernBERT-large (`answerdotai/ModernBERT-large`) | 421M | 512 |
| `veya-multilingual` | mmBERT-base | 322M | 1024 |
| `veya-typed-decisions` | ModernBERT-large (fine-tuned) | 421M | 1024 |

### Decision primitives

| Primitive | qtype id | Semantics |
|---|---|---|
| `choice` | 0 | Classification over 2–100 candidates |
| `score` | 1 | Ordinal rubric (ordered levels; served Score is categorical) |
| `noul` | 2 | Calibrated `P(true)` for a statement |
| `systemone` | — | Composite: one state + up to 8 mixed-type questions in a single pass |

### Audited release path

The **English Choice** primitive is the audited **Research Release** (packed ≤ 19 candidates; K=8 shortlist + rerank above 19). All other primitives, languages, and experimental heads are **research-only**. See `docs/audit/release_verdict.md`.

---

## 4. Installation

```bash
pip install veya
```

**Required dependencies** (upper bounds cap the next major of each dep; patch/minor upgrades flow via Dependabot):

| Package | Constraint |
|---|---|
| `torch` | `>=2.0.0,<3` |
| `transformers` | `>=4.48.0,<6` |
| `safetensors` | `>=0.4.0,<1` |
| `huggingface_hub` | `>=0.20.0,<2` |
| `numpy` | `>=1.20.0,<3` |

**Optional extras:** `pip install 'veya[train]'` adds `datasets>=2.14.0,<4` (the reference trainer streams the HF typed-decisions dataset; runtime does not need it).

**Packaged modules:** `veya`, `veya.experimental`, `web` (with `web/static/*` package data). `assets/`, `research/`, and `notebooks/` are repository-only and excluded from auto-discovery.

---

## 5. Core Library Reference (`veya/`)

### 5.1 Public API Surface (`veya/__init__.py`)

`__version__ = "0.3.5"`. Exports (`__all__`):

| Kind | Symbols |
|---|---|
| Classes | `Agent`, `RLAgent` (alias), `Router`, `RouteDecision` |
| Constants | `DEFAULT_MODELS`, `QTYPES`, `QTYPE_NAMES`, `__version__` |
| Functions | `load`, `shortlist_choice`, `predict_shortlist`, `embed_fn_from_agent`, `detect_language`, `detect_script`, `is_english`, `clean_email_body`, `email_questions`, `email_state`, `guard_questions`, `moderation_questions`, `router_questions`, `triage_questions`, `proper_reward`, `td_lambda_targets`, `ece_score`, `confidence_from_probs`, `render_options` |

`detect_language` is `veya.lang.analyse` re-exported. Checkpoints are downloaded from HF Hub (`veyadynamics/veya` bundle repo).

### 5.2 Core Architecture — `common.py`

**Constants:** `QTYPES = {"choice": 0, "score": 1, "noul": 2}`; `QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}`; `TEMP_MIN = 0.5`; `TEMP_MAX = 5.0`.

**`DecisionModel(nn.Module)`** — bidirectional transformer encoder + typed decision head:

| Component | Definition |
|---|---|
| `encoder` | Backbone (`AutoModel`, SDPA attention) |
| `head` | `nn.TransformerEncoder` over `head_layers` (default 2) `TransformerEncoderLayer(d, nhead=d//64, 4d, dropout=0.1, batch_first=True, norm_first=True)` |
| `type_emb` | `nn.Embedding(3, d)` — primitive-type embedding added to hidden states |
| `scorer` | `LayerNorm(d) -> Linear(d, d) -> GELU -> Linear(d, 1)` — per-option logit at marker positions |
| `act_head` | `Linear(d + 4, 256) -> GELU -> Linear(256, n_act)` — defer/answer action head |
| `temperature` | Registered buffer of 3 per-primitive temperatures |

`forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False) -> (logits, act_logits)`. Option logits are gathered at `[MASK]` marker positions, masked with `-1e4` outside the marker mask. Act features: top-1 probability, top1–top2 margin, normalized entropy over `k` options, and `k/255`, concatenated with the pooled `[CLS]` hidden state.

**Functions:**

| Function | Purpose |
|---|---|
| `build_model(cfg, encoder_dir=None)` | Instantiate `DecisionModel` from config (`head_layers`, `act_costs`) |
| `build_sequence(tok, state, q, ...)` | Tokenize as `[CLS] <type> instructions [SEP] [MASK]opt0 [MASK]opt1 … [SEP] state [SEP]` (defaults `max_len=512`, `head_max_len=192`, `option_token_cap=48`; supports `option_order` permutation and left/right state truncation) |
| `proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0)` | Strictly proper scoring rule: log score + spherical + ranked probability score (RPS subtracted for `score` qtype) |
| `td_lambda_targets(...)` | TD(λ) targets, `gamma=0.99`, `lam=0.95` |
| `ece_score(...)` | Expected calibration error, `n_bins=15` |
| `safe_automation_coverage(...)` | Coverage under the selective-automation safety rule |
| `confidence_from_probs(probs)` | Confidence from a probability vector |
| `collate_items(...)` | Batch collation for inference |
| `clamp_temperature(t, lo=TEMP_MIN, hi=TEMP_MAX)` | Clamp fitted temperatures |
| `render_options(q)` | Render option texts in label-index order; `noul` is always `[false, true]` |
| `serialize_state(state)` | str passthrough / compact JSON for dict/list |
| `state_has_content(state)` | Non-empty supported inference input |

### 5.3 Agent Runtime — `agent.py`

**`Agent`** (aliased `RLAgent`):

- `__init__(model_id_or_path, device=None, token=None, subfolder=None)` — downloads from HF Hub, loads `rl_agent_config.json` + `model.safetensors` via `safetensors`, verifies architecture, applies calibration, and falls back across devices (CUDA → CPU, MPS → CPU).
- `system_one(state, questions)` / `predict(state, questions)` — a single parallel forward pass; returns per-question answers with probabilities, confidence, a defer/answer action, and token usage.
- `_verify_compatibility()` — checks required config keys and required weight prefixes (`encoder.`, `type_emb.`, `scorer.`, `act_head.`).
- `_fix_tokenizer_config()` — patches `tokenizer_config.json` for cross-version tokenizer compatibility.

**`load(model_id, **kwargs) -> Agent`** — canonical constructor.

### 5.4 Calibration System — `calibration.py`

- `FORMAT = "veya-calibration-v1"`; `MIN_OBSERVATIONS = 32`.
- Cardinality buckets: `"2"`, `"3-5"`, `"6-10"`, `"11-20"`, `"21-50"`, `"51+"`.

| Function | Purpose |
|---|---|
| `fit_temperature(...)` | Grid search over 48 log-spaced temperatures in `[TEMP_MIN, TEMP_MAX]` |
| `fit_groups(...)` | Fit per (language, primitive, bucket) groups |
| `build_artifact(...)` | Assemble the calibration artifact |
| `resolve_temperature(...)` | Fallback chain: language+bucket → bucket → primitive → global |
| `dump_calibration` / `load_calibration` | Artifact I/O |
| `apply_calibration` | Apply temperatures to logits |

**Validation:** rejects fitting on the test split, wrong checkpoint hash, and wrong `scorer_id` — the artifact is bound to the exact producing checkpoint.

### 5.5 Email Utilities — `email.py`

- `clean_email_body(text)` — strips quoted history, signatures, and disclaimers; caps at 3000 characters.
- `email_state(...)` — builds the email state payload.
- `email_questions()` — preset questions: `category`, `is_spam`, `is_phishing`, `urgency`, `needs_reply`.

### 5.6 Language Detection — `lang.py`

Dependency-free. Covers **24 non-Latin script ranges**, **8 stopword languages** (`en`, `fr`, `de`, `es`, `pt`, `it`, `nl`, `ro`), and diacritics detection. `NON_EN_DIACRITIC_RATE = 0.02`.

| Function | Purpose |
|---|---|
| `detect_script(text)` | Primary Unicode script |
| `script_profile(text)` | Per-script character profile |
| `latin_profile(text)` | Latin-script statistics |
| `guess_latin_language(text)` | Stopword/diacritic language guess |
| `analyse(text)` | Full analysis result |
| `is_english(text)` | English gate |
| `defer_reason(...)` | Why a non-English input should defer |
| `reads_non_latin(...)` | Non-Latin script heuristic |

Language routing happens **before** the forward pass — the English checkpoint does not degrade gracefully outside English (see [13.5](#135-multilingual-51-language-massive-sweep)).

### 5.7 Manifest Builder — `manifest.py`

`build_manifest(model_dir, git_commit)` — creates a manifest with `model_id`, `architecture_version`, `backbone`, `checkpoint_hash` (SHA-256), `calibration_artifact`, `supported_languages`, `max_state_tokens`, `max_candidates_packed`, `decision_primitives`, `git_commit`, `created_at`.

### 5.8 Preset Question Schemas — `presets.py`

| Function | Questions |
|---|---|
| `triage_questions()` | `intent`, `is_urgent`, `frustration`, `refund_requested`, `churn_risk` |
| `email_questions()` | `category`, `is_spam`, `is_phishing`, `urgency`, `needs_reply` |
| `guard_questions()` | `jailbreak`, `prompt_injection`, `sensitive_data`, `harm_severity`, `topic` |
| `moderation_questions()` | `toxic`, `harassment`, `threat`, `spam`, `severity` |
| `router_questions()` | `difficulty`, `domain`, `needs_tools`, `is_sensitive` |

### 5.9 Release Bundle — `release_bundle.py`

Constants: `ROUTER_FORMAT = "veya-choice-router-v1"`; `INSTRUCTIONS = "Which label best fits the text?"`.

**`ReleaseBundle`** — wraps agent + router + shortlist + calibration + selective policy:

- `score_choice(...)` — routes between the packed path (≤ 19 candidates) and the shortlist path (> 19).
- `public_choice(...)` — full response including selective automation (defer/answer).
- `answer_questions(...)` — routes each question type to its primitive handler.

**Functions:** `load_bundle`, `load_release_manifest` (format `"veya-release-manifest-v2"`; recomputes SHA-256 of every pinned file at startup; path-traversal prevention), `load_router`, `plan_for`, `raw_choice_logits`, `recommend_packed_cutoff`.

A hash mismatch raises before the service binds — the audited bundle never loads over a modified runtime file.

### 5.10 Multi-Checkpoint Router — `router.py`

Constants: `BUNDLE_REPO = "veyadynamics/veya"`; `DEFAULT_MODELS` = `english`, `multilingual`, `typed-decisions`; `STANDALONE_MODELS`.

**`Router`** — `__init__(models, device, token, max_loaded=1, default="english", auto_task_detection=False, preload=False)`. LRU eviction guarded by an `RLock`, with inference performed **outside** the lock. `RouteDecision` is a dict subclass.

**Route precedence:** explicit model > explicit task > detected workflow (opt-in) > explicit language > detected script/language > default.

**Typed-decision workflows** (detected when `auto_task_detection=True`): `agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents`.

### 5.11 Selective Automation — `selective.py`

| Function | Purpose |
|---|---|
| `score_probabilities` | Selective scores: `max_prob`, `neg_entropy`, `margin` |
| `select_threshold` / `apply_threshold` | Threshold selection and application |
| `wilson_interval` | Wilson confidence interval |
| `risk_coverage_curve` | Risk–coverage sweep |
| `apply_report` | Apply a policy to an evaluation report |
| `load_policy` | Load + validate a policy artifact |

Policy format `"veya-selective-policy-v1"`; validates `route`, `checkpoint_hash`, `shortlist_sha256`, `scorer_id`, and `chosen_on=validation`. Thresholds are chosen on validation data and applied unchanged — never re-selected on data that showed a breach.

### 5.12 Model Server — `server.py`

`MAX_REQUEST_BYTES = 1MB`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /version` | Release metadata |
| `POST /v1/choice` | Packed / shortlist choice |
| `POST /v1/noul` | Calibrated P(true) |
| `POST /v1/score` | Ordinal rubric score (categorical) |
| `POST /v1/systemone` | One state + dict of questions |
| `POST /v1/choice-hypothesis` | Hypothesis-conditioned choice |
| `POST /v1/choice-compare` | Experimental comparison |

Architecture: stdlib `ThreadingHTTPServer`; inference serialized by a process-wide `_INFER_LOCK`; `serve(agent, host="127.0.0.1", port=8080, bundle, hypothesis)`.

### 5.13 Embedding Shortlist — `shortlist.py`

Constants: `DEFAULT_SHORTLIST_K=20`; `MAX_CHOICE_CANDIDATES=1024`; `PACKED_MARKER_CEILING=100`; `PRODUCTION_SHORTLIST_ABOVE=19`; `PRODUCTION_SHORTLIST_K=8`.

| Function | Purpose |
|---|---|
| `shortlist_choice(...)` | Cosine-similarity candidate ranking |
| `predict_shortlist(...)` | Shortlist + rerank inference |
| `embed_fn_from_agent(...)` | Mean-pooled embeddings with attention-mask exclusion |
| `_criteria_items(...)` | Unicode NFKC normalization + duplicate detection |

### 5.14 Shortlist Artifact — `shortlist_artifact.py`

Constants: `ROUTE = "shortlist_k8_packed_rerank"`; `FORMAT = "veya-shortlist-v1"`. A bias-free linear projection with dot-product scoring in a **128-dimensional learned space**.

Functions: `save_shortlist`, `load_shortlist`, `save_route_calibration`, `load_route_calibration`. Every load validates `checkpoint_hash`, `shortlist_sha256`, `fitting_split`, and `scorer_id`.

### 5.15 K=8 Packed Reranker — `reranker.py`

`FORMAT = "veya-k8-reranker-v1"`.

**`K8Reranker(nn.Module)`** — clones `head`/`type_emb`/`scorer` from the production model. Methods: `clone_from_agent`, `choice_logits` (agent encoder + reranker head), `save_reranker`, `load_reranker`.

### 5.16 RLCD Training — `train.py`

**`TrainConfig`** defaults:

| Field | Value | Field | Value |
|---|---|---|---|
| `epochs` | 4 | `w_sph` | 0.75 |
| `micro_batch` | 8 | `w_rps` | 1.0 |
| `grad_accum` | 4 | `rl_weight` | 1.0 |
| `group_size` | 4 | `ce_weight` | 1.0 |
| `lr_encoder` | 2.5e-5 | `act_weight` | 0.0 |
| `lr_head` | 1e-4 | `grad_clip` | 1.0 |
| `weight_decay` | 0.01 | `seed` | 42 |
| `sigma_start` | 0.4 | `max_len` | 1024 |
| `sigma_end` | 0.1 | `head_max_len` | 256 |

**Functions:**

| Function | Purpose |
|---|---|
| `compute_rcld_loss(...)` | GRPO-style policy gradient + soft cross-entropy + optional act head |
| `run_finetune(...)` | AdamW, `CosineAnnealingLR`, `GradScaler` |
| `build_training_item(...)` | Training item construction |
| `collate_train_batch(...)` | Batch collation |
| `split_items(...)` | Train/calibration split |
| `fit_temperature_soft(...)` | Soft-label temperature fitting |

**Outputs:** `model.safetensors` + `rl_agent_config.json`.

---

## 6. Experimental Modules (`veya/experimental/`)

Module docstring: *“Experimental heads. Present and tested. Not used by `Agent.predict`.”*

| Module | Contents |
|---|---|
| `candidate_set.py` | **CandidateSetHead** — state-once candidate scorer; scores each candidate independently |
| `ordinal.py` | **OrdinalHead** — cumulative threshold model with softplus chain, ordered class probabilities |
| `top8.py` | **Top8Interaction** — 1-layer transformer cross-attention between query and 8 candidates, `dim=1024`, `width=256` |
| `ordinal_artifact.py` | Ordinal artifact — `FORMAT = "veya-ordinal-score-v1"`; enforces `status="research"`, `production=False` |
| `ordinal_data.py` | **OrdinalDataUnavailable** exception; `load_ordinal_table` from CSV (`text`, `label`), requires ≥ 32 rows |
| `domain_retriever.py` | Phase-10/11 domain-general candidate retrieval — `FORMAT = "veya-domain-retriever-v1"`; modes `MODE_CURRENT`, `MODE_PHASE10`, `MODE_PHASE10_MULTIVIEW`, `MODE_PHASE11`; query fusions `first`, `max`, `mean`, `rrf`; Phase-11 view policies `single`, `first_last`, `first_middle_last`, `four_view`, `adaptive`; classes `ProjectionPair(nn.Module)`, `DomainRetriever`; functions `build_query_views`, `build_phase11_query_views`, `fuse_query_scores`, `naturalize_label`, `save/load_domain_retriever`, `configure_experimental_retriever` |
| `hypothesis.py` | Phase-9 state–hypothesis compatibility scoring — `FORMAT = "veya-hypothesis-scorer-v1"`, `TEMPLATE_VERSION = "phase9-h0-v1"`, `MAX_LENGTH = 160`; classes `CompatibilityHead(nn.Module)`, `IsolatedAdaptedEncoder(nn.Module)`, `HypothesisScorer`; functions `hypothesis_text`, `pair_segments`, `encode_pairs`, `compatibility_logits`, `score_groups`, `load_compatible_scorer`, `validate_phase9_record` |

---

## 7. Web Gateway (`web/`)

### 7.1 Gateway Server — `web/server.py`

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `VEYA_MODEL_URL` | `http://127.0.0.1:8080` | Model service URL |
| `VEYA_WEB_PORT` | `3000` | Gateway port |
| `VEYA_WEB_HOST` | `127.0.0.1` | Bind address |
| `VEYA_PUBLIC_ORIGIN` | unset | Required for public bind (canonical HTTPS origin) |
| `VEYA_SHADOW_MODE` | unset | Shadow telemetry flag |
| `VEYA_REQUEST_TIMEOUT` | `30` | Per-request inference ceiling (seconds) |

**Rate limiting:** `WINDOW_S=60`, `CAPACITY=12` weighted units per IP (weight `1 + ceil((candidates-19)/20)`, capped at 12). **Concurrency:** `MAX_INFLIGHT=2` via `BoundedSemaphore` — the 3rd concurrent inference is refused with `503 model_busy`. **Size limits:** `MAX_REQUEST_BYTES=32KB`, `MAX_RESPONSE_BYTES=512KB`, `INFER_TIMEOUT_S=30`.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness with component latching |
| GET | `/startupz` | Startup probe |
| GET | `/robots.txt` | Robots exclusion |
| GET | `/sitemap.xml` | Sitemap |
| GET | `/api/benchmark` | Audited benchmark aggregate |
| GET | `/api/health` | Model service health proxy |
| GET | `/api/version` | Release metadata |
| GET | `PAGES` dict | Website pages from static files |
| GET | `/static/*` | Static file serving |
| POST | `/api/infer` | Inference with validation, rate limiting, CORS |

**Security headers:** `X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options`, HSTS, `Permissions-Policy`, CSP. Cross-origin model requests are rejected.

**Graceful shutdown:** `SIGTERM`/`SIGINT` → stop admitting work, drain in-flight requests for `SHUTDOWN_DRAIN_S=30`s, then hard exit. While draining, `/healthz` still answers 200; all non-probe routes return `503 shutting_down`.

**Readiness components** (monotonic latching — they cannot flap back): `model_loaded`, `policy_loaded`, `calibration_loaded`, `shortlist_loaded`, `hashes_valid`, `startup_failed` (sticky fail-closed).

**Error codes:** `VALIDATION_ERROR`, `UNSUPPORTED_LANGUAGE`, `OVERLOADED`, `RATE_LIMITED`, `TIMEOUT`, `INTERNAL_ERROR`, `SHUTTING_DOWN`, `NOT_FOUND`.

**Telemetry fields** (one JSON line per request, logger `veya.web`): `request_id`, `primitive`, `candidate_count`, `route`, `deferred`, `status`, `latency_ms`, `model_version`, `confidence`, `max_probability`, `automation`, `config_hash`, `shadow` (plus `error_code`, `timestamp`). Never logs request text or candidate labels.

### 7.2 Model Server Entry — `web/model_server.py`

Loads the release manifest from `artifacts/release/manifest.json` via `load_release_manifest()` and starts the model service on `127.0.0.1:8080` (never published).

### 7.3 Request/Response Contracts — `web/contracts.py`

Constants: `Primitive = Literal["choice", "noul", "score", "systemone"]`; `MAX_CHOICE=100`; `MAX_QUESTIONS=8`; `MAX_STATE_BYTES=12000`.

- Exceptions: `RequestError`, `UpstreamContractError`.
- `validate()` — primitive type, state, instructions, criteria with strict bounds (choice 2–100 candidates; systemone 1–8 questions).
- `validate_response()` — whitelist of model output fields; strips `max_probability` from the client payload (it is captured server-side for telemetry first).

---

## 8. Models and Production Artifacts

### 8.1 Production Models (`models/production/`)

| File | Purpose |
|---|---|
| `model.safetensors` (~803 MB) | Production checkpoint weights |
| `rl_agent_config.json` | Config: ModernBERT-large, `head_layers=2`, `max_len=512`, `head_max_len=192`, `amp_dtype=bf16`, 7313 training updates |
| `choice_router.json` | `packed_max_candidates=19`, `hierarchical_k=8`, `embed_max_length=96` |
| `shortlist_k8.pt` + `.json` | K=8 shortlist projection artifact |
| `calibration.json` | Per-cardinality temperature buckets (packed choice) |
| `calibration_shortlist_k8.json` | Hierarchical route temperature (1.971127355603397) |
| `calibration_noul.json` | Noul calibration |
| `calibration_score.json` | Score calibration |
| `calibration_multilingual.json` | Multilingual calibration |
| `calibration_hypothesis_hce2.json` | Hypothesis scorer calibration |
| `encoder/` | Encoder sub-directory |
| `tokenizer/` | Tokenizer files |
| `multilingual/` | mmBERT-base multilingual subfolder |
| `typed-decisions/` | Fine-tuned typed-decisions subfolder |

**`rl_agent_config.json` key values:**

- `temperature` (per primitive): `[1.637, 1.251, 1.983]` (exact: `1.6369030475616455`, `1.2514300346374512`, `1.983399510383606`)
- `temperature_by_options`: `choice:2 = 1.906`, `choice:3-5 = 1.760`, `choice:6-10 = 1.000`, `choice:11+ = 0.101`, `score:3-5 = 1.251`, `noul:2 = 1.983`
- Training: 7313 updates, 1 epoch, 1.96 hours, `fine_tuned_from_checkpoint = true`, `act_costs = {"escalate": 0.5}`, `cost_wrong_act = 3.0`

### 8.2 Release Artifacts (`artifacts/release/`)

20 items:

| File | Purpose |
|---|---|
| `manifest.json` | Release manifest v2 — SHA-256 pins for all 15 `veya/*.py` runtime files, checkpoint, tokenizer, router, shortlist, calibrations, selective policy |
| `selective_policy.json` | Selective automation policy (`operating_epsilon: "disabled"`) |
| `production_scope.json` | Machine-readable production scope contract |
| `production_readiness_matrix.json` | G0–G24 readiness matrix |
| `runtime_audit.json` | Runtime performance audit (second-order pin: `manifest_sha256` of the manifest file itself) |
| `shadow_schema.json` | Shadow record schema, sample-size requirements, acceptance criteria |
| `activation_procedure.json` | Pre-registered selective-activation procedure |
| `prospective_blind_config.template.json` | G6/G8 pre-registration template |
| `gap_audit.json` | Structured gap audit |
| `legal_review_packet.json` | 6 flagged datasets with exact legal questions |
| `license_inventory.json` | License and third-party inventory |
| `release_metadata.json` | Reproducible release metadata (build provenance) |
| `sbom.json` | Software bill of materials |
| `system_map.json` | System map |
| `phase_audit.json` | Phase audit record |
| `route_cutoff_audit.json` | Packed/shortlist route cutoff audit |
| `blind_clinc150_val_result.json` | Sealed CLINC150 validation blind result |
| `blind_clinc150_test_failed_run.json` | Recorded failed blind run (fail-closed evidence) |
| `manifest-phase5.json` | Historical phase-5 manifest |
| `manifest-phase9.json` | Historical phase-9 manifest |

### 8.3 Decision Records (`artifacts/decisions/`)

| File | Decision |
|---|---|
| `automation_policy.json` | Selective automation policy parameters |
| `choice_routing.json` | Packed vs shortlist routing decision |
| `calibration.json` | Calibration approach decision |
| `candidate_semantics.json` | Candidate representation semantics |
| `candidate_set.json` | Candidate-set head decision (research-only) |
| `commercialization.json` | Commercialization track decisions |
| `domain_general_retrieval.json` | Phase-10/11 domain-general retrieval decision |
| `encoder_adaptation.json` | Encoder adaptation strategy decision |
| `hypothesis_scoring.json` | Phase-9 hypothesis scoring decision |
| `k8_reranker.json` | K=8 reranker decision |
| `noise_robust_retrieval.json` | Noise-robust retrieval decision |
| `score_architecture.json` | Score head architecture decision |

### 8.4 Experiment Phases (`artifacts/experiments/`)

Phase directories `phase3` through `phase12` (including `phase11b`) plus `retired.json`, which catalogs **8 retired experiments**:

1. K16/K32 production rerank
2. Frozen cosine retrieval
3. Candidate-set scorer
4. Head-only listwise reranker
5. Pairwise head addon
6. Mean-pooled top-8 scorer
7. Phase-7 top-layer on short labels
8. Packed semantic enrichment

### 8.5 Commercialization Artifacts (`artifacts/commercialization/`)

| File | Purpose |
|---|---|
| `blind_hwu64_result.json` | HWU64 blind evaluation result (frozen; pins the pre-re-mint manifest `c7febd23…`) |
| `graph.json` | Commercialization capability graph |
| `query_policy_freeze.json` | Query policy freeze record |
| `query_robustness.json` | Query robustness evaluation |

---

## 9. Architecture

### 9.1 System Architecture

```
Browser ──> VeyaClient ──POST /api/infer──> Web Gateway (web/server.py, port 8000/3000)
                                            │  rate limiting · CORS · validation · static site
                                            ▼
                              Model Service (veya/server.py, 127.0.0.1:8080, never published)
                                            │  _INFER_LOCK serializes forward passes
                                            ▼
                              load_release_manifest()  ── SHA-256 verified, fail-closed
                                            ▼
                       ReleaseBundle (Agent + Router + Shortlist + Calibration + Policy)
```

### 9.2 Two-Process Serving Model

- **Gateway** (`web/server.py`): rate limiting, CORS, request validation, static website serving, telemetry, graceful shutdown.
- **Model service** (`veya/server.py` via `web/model_server.py`): checkpoint ownership, hash verification, inference.
- Both are stdlib `ThreadingHTTPServer` processes (one container, model service on loopback only).
- Inference is serialized by a single lock; horizontal scaling is via the K8s HPA (2–8 replicas) — scale out, not up.

### 9.3 Fail-Closed Integrity

Every runtime file (`veya/*.py`, 15 files), model artifact, calibration, shortlist, and policy is SHA-256 pinned in `artifacts/release/manifest.json`. A mismatch prevents startup (`load_release_manifest` raises before binding). `runtime_audit.json` additionally pins the SHA-256 of the manifest file itself. The readiness probe latches sticky-fail: once `startup_failed` is observed, the process never reports ready again.

### 9.4 Selective Automation

Currently **DISABLED** — every decision defers (`operating_epsilon: "disabled"`; packed route reports `automation: null`, shortlist route `automation: "defer"`). Frozen reference thresholds: 0.02 → `0.9796500892257723`, 0.05 → `0.9541679372957224`, 0.10 → `0.896261827143334`. Activation requires a prospective blind evaluation PASS (G6/G8 gates) on target-domain data, per `activation_procedure.json` and `docs/OPERATIONS.md`.

### 9.5 Concurrency Model

From `docs/CONCURRENCY.md`:

- Two processes, one HTTP hop.
- `_INFER_LOCK`: a single `threading.Lock` serializes forward passes.
- `MAX_INFLIGHT=2` via `BoundedSemaphore` (3rd concurrent request → `503 model_busy`).
- Rate limiting: 12 weighted units per 60s per IP.
- Graceful shutdown: 30s drain.
- Performance baseline (CPU, `runtime_audit.json`): p50 ≈ 105 ms small packed choice at ≈ 9.5 req/s; ≈ 2030 MB serving RSS; choice with 77 candidates ≈ 163 ms / 6.1 req/s; noul ≈ 76 ms / 11.7 req/s; score ≈ 76 ms / 12.9 req/s; systemone (3 questions) ≈ 215 ms / 4.5 req/s.

---

## 10. Infrastructure

### 10.1 Docker

**`Dockerfile`** — multi-stage build on `python:3.11-slim`:

- Stage 1 (builder): `pip install --no-cache-dir -r requirements.lock` (uv-compiled pins).
- Stage 2 (production): non-root user `veya` (no login shell), copies `veya/`, `web/`, `models/production/`, `artifacts/release/`; pre-compiled bytecode (`python -m compileall`); `EXPOSE 8000`.
- `HEALTHCHECK` on `/readyz` (interval 30s, timeout 10s, **start-period 120s**, retries 3).
- ENV defaults: `VEYA_SHADOW_MODE=0`, `VEYA_LOG_LEVEL=INFO`, `VEYA_MODEL_DIR=/app/models/production`, `VEYA_WEB_HOST=0.0.0.0`, `VEYA_WEB_PORT=8000`, `VEYA_PUBLIC_ORIGIN=https://localhost`.
- `CMD` runs the model service in the background and `exec`s the gateway as PID 1 (so it receives stop signals).

**`docker-compose.yml`** — single service `veya-server`: ports `8000:8000`; memory reservation 4G / limit 6G; read-only bind mounts (`models/production`, `artifacts/release`); healthcheck with 120s start period; `restart: unless-stopped`.

### 10.2 Kubernetes (`k8s/`)

| Manifest | Contents |
|---|---|
| `deployment.yaml` | 2 replicas; `RollingUpdate` (`maxSurge=1`, `maxUnavailable=0`); `terminationGracePeriodSeconds=45`; security: `runAsNonRoot`, `seccompProfile: RuntimeDefault`, `readOnlyRootFilesystem`, capabilities drop `ALL`; resources: requests `cpu=1`/`mem=4Gi`, limits `cpu=2`/`mem=6Gi`; probes: liveness `/healthz` (10s), readiness `/readyz` (5s), startup `/startupz` (5s × 24 = 120s); `preStop sleep 5`; pod anti-affinity |
| `service.yaml` | `ClusterIP`, port 80 → container 8000 |
| `hpa.yaml` | 2–8 replicas, 70% CPU target; scale-up 2 pods/60s (no stabilization); scale-down 300s stabilization |
| `configmap.yaml` | `public-origin` and `shadow-mode` keys |
| `kustomization.yaml` | Base with all 4 resources |
| `shadow-overlay/` | `namePrefix: shadow-`, `shadow-mode="1"`, 1 replica, HPA 1–2, disjoint `shadow-veya` selectors |

### 10.3 CI/CD (`.github/workflows/`)

| Workflow | Trigger | Jobs / gates |
|---|---|---|
| `ci.yml` | PR + push to `main`; Python 3.10/3.11/3.13 matrix | `test`, `lint` (ruff + mypy on `veya/`), `build`, `pr-gate` (artifact integrity + security), `main-gate` (integration + reliability) |
| `release.yml` | `v*` tags | Version verify, PyPI **Trusted Publisher**, GitHub Release (owner `benchbrex-USA/veya`) |
| `security.yml` | PR / push / weekly schedule | 4 jobs: `secrets` (gitleaks), `deps` (pip-audit), `codeql` (Python `security-extended`), `supply_chain` (no pickle / `torch.load` / `eval` / `exec` / `subprocess` in `veya/`) |
| `release-gate.yml` | `v*` tags | Full test suite, artifact integrity, license inventory, freeze records, Docker build |

---

## 11. Testing

**Suite status:** full `tests/` run — **442 passed / 1 failed / 3 skipped** (6 subtests passed). The single failure is the known non-blocking commercialization freeze cross-reference (the frozen blind record legitimately pins the pre-re-mint manifest `c7febd23…`). Focused 11-suite gate (artifact integrity, release metadata, telemetry `max_probability`, shadow deployment, blind verdict, prospective-blind CLI, activation procedure, security hardening, health probes, reliability hardening, concurrency) — **169 passed, 0 failed**.

The suite contains **50 test files** (48 pytest modules + 2 Node.js `.mjs` suites):

### Core library & training

| Test file | Coverage |
|---|---|
| `test_criteria.py` | `render_options` / criteria rendering (dict/list rubrics, noul defaults) |
| `test_decision_model.py` | `DecisionModel` forward, marker gathering, act features, single-option edge |
| `test_train.py` | RLCD loss, `TrainConfig`, finetune loop, collation, splits |
| `test_manifest.py` | `build_manifest` fields and checkpoint hashing |
| `test_download.py` | HF Hub download and tokenizer config patching |
| `test_email.py` | Email cleaning, state building, preset questions |
| `test_router.py` | Route precedence, LRU eviction, typed-decision workflows |
| `test_shortlist.py` | Cosine shortlist, embeddings, NFKC/duplicate handling |
| `test_shortlist_artifact.py` | Shortlist save/load validation (hash, split, scorer) |
| `test_reranker.py` | `K8Reranker` clone, logits, save/load |
| `test_top8.py` | `Top8Interaction` experimental head |
| `test_phase2.py` | Phase-2 packed-choice baseline |
| `test_phase6.py` | Phase-6 interaction head experiments |
| `test_calibration_report.py` | ECE / Brier / reliability / risk-coverage reporting |
| `test_packaging.py` | Package metadata, version, module layout |

### Selective prediction & evaluation integrity

| Test file | Coverage |
|---|---|
| `test_selective_gates.py` | G1–G4 gate semantics (fail-closed) |
| `test_selective_extra.py` | `veya/selective.py` scoring, Wilson intervals, risk-coverage |
| `test_sample_size.py` | Wilson-inversion sample-size requirements (189/957 cliff) |
| `test_contamination.py` | Train/test overlap detection |
| `test_surface_noise.py` | Surface-noise perturbation suites |
| `test_paraphrase_noise.py` | Paraphrase-noise suites |

### Serving, gateway & reliability

| Test file | Coverage |
|---|---|
| `test_web_gateway.py` | Gateway endpoints, validation, CORS, headers, telemetry |
| `test_server_security.py` | Model-service security |
| `test_security_hardening.py` | Transport security headers, HSTS, audit logging |
| `test_model_server_e2e.py` | Model service end-to-end |
| `test_local_e2e.py` | Full local stack end-to-end |
| `test_concurrency.py` | `_INFER_LOCK` serialization, `MAX_INFLIGHT` guarantees |
| `test_reliability.py` | Timeouts, overload 503, structured errors |
| `test_reliability_hardening.py` | Graceful shutdown, drain, retry headers |
| `test_health_probes.py` | `/healthz`, `/readyz`, `/startupz`, sticky fail-closed |
| `test_drift_monitor.py` | Drift reports and triggers |

### Release, audit & certification

| Test file | Coverage |
|---|---|
| `test_artifact_integrity.py` | 31 strict manifest/audit hash tests (15/15 runtime files) |
| `test_release_bundle.py` | `load_release_manifest` verification, path traversal prevention |
| `test_release_metadata.py` | Release metadata assembly and provenance |
| `test_freeze_record.py` | `veya-freeze-record-v1` minting/validation |
| `test_blind_verdict.py` | Verdict orchestrator, gate ordering, split isolation |
| `test_prospective_blind_cli.py` | CLI end-to-end: PASS/FAIL/exit-2, tamper detection |
| `test_activation_procedure.py` | 28 activation-procedure tests |
| `test_shadow_deployment.py` | 13 shadow-mode tests (header, schema, flatten, sample-size cliff) |
| `test_telemetry_max_probability.py` | 15 server-side capture tests |
| `test_commercialization.py` | Frozen commercialization records, tamper regression |
| `test_rival_benchmark.py` | Rival harness (byte-identical cases, `metrics()` parity) |

### Research-phase modules

| Test file | Coverage |
|---|---|
| `test_phase9_hypothesis.py` | Hypothesis scorer (compatibility, calibration) |
| `test_phase10_retriever.py` | Domain retriever (modes, fusions, view policies) |
| `test_phase11_noise.py` | Phase-11 noise families |
| `test_phase11_protocol.py` | Phase-11 protocol registration |
| `test_phase12_context.py` | Phase-12 context audits |
| `test_phase12_sparse_scorer.py` | Sparse scorer comparisons |

### Web assets (Node.js)

| Test file | Coverage |
|---|---|
| `test_model_client.mjs` | Browser `VeyaClient` (`web/static/model-client.mjs`) |
| `test_site_content.mjs` | Website content integrity |

---

## 12. Research Module (`research/`)

### Scripts (`research/scripts/`)

| Script | Purpose |
|---|---|
| `bench_apps.py` | Application workflow suites (6 themes + public datasets, all 3 checkpoints) |
| `bench_latency.py` | Latency / throughput measurement |
| `bench_local.py` | Local benchmark harness |
| `bench_rival.py` | Rival harness — runs rival models on byte-identical cases (`--rival all --n 400`) |
| `build_benchmark_nb.py` | Builds the Colab benchmark notebook |
| `make_plots.py` | Benchmark plots |
| `make_rival_plot.py` | Rival comparison figure (`assets/veya_vs_rival.png`) |
| `scan_dependency_licenses.py` | License scanning for the inventory |
| `veya_benchmark_colab.ipynb` | Published Colab benchmark notebook |

### VeyaBench (`research/veyabench/`)

| Module | Purpose |
|---|---|
| `blind_verdict.py` | P1–P6 verdict orchestrator (`veya-blind-verdict-v1`) with drift/gates/contamination annexes |
| `run_prospective_blind.py` | One-command G6/G8 CLI (exit 0/1/2) |
| `selective_gates.py` | G1–G4 selective gate reports |
| `drift_monitor.py` | `veya-drift-report-v1` reports and fail-closed triggers |
| `contamination.py` | Contamination / train-test overlap screens |
| `calibration_report.py` | ECE, Brier, reliability, risk-coverage |
| `freeze_record.py` | `veya-freeze-record-v1` freeze minting and validation |
| `release_metadata.py` | Release metadata assembler (`assemble_release_metadata`) |
| `sample_size.py` | Wilson-interval sample-size requirements (`required_n`) |
| `surface_noise.py` | Surface noise perturbation conditions |
| `paraphrase_noise.py` | Paraphrase perturbation conditions |
| `build_sbom.py` | SBOM construction |
| `protocol.py` | Shared evaluation protocol |
| `write_baseline.py` | Baseline writing |
| `run_runtime_audit.py` | Runtime audit (loads the model; regenerates `runtime_audit.json`) |
| `run_release_eval.py` | Release evaluation suite |
| `run_blind_clinc150.py` | Sealed CLINC150 blind evaluation |
| `run_commercial_blind_hwu64.py` | Commercialization blind HWU64 run |
| `run_commercial_query_robustness.py` | Query robustness evaluation |
| `run_ordinal_eval.py` | Ordinal (research) evaluation |
| `run_phase3_eval.py`, `run_phase3_followup.py` | Phase-3 evaluation and follow-up |
| `run_phase4_banking.py` | Phase-4 Banking77 |
| `run_phase5_release.py` | Phase-5 release |
| `run_phase6_baseline.py`, `run_phase6_final.py`, `run_phase6_interaction.py`, `run_phase6_train.py` | Phase-6 interaction experiments |
| `run_phase7_adapt.py`, `run_phase7_update_audit.py` | Phase-7 adaptation |
| `run_phase8_eval.py`, `build_phase8_semantics.py` | Phase-8 semantics |
| `run_phase9_hypothesis.py`, `run_phase9_integrate.py` | Phase-9 hypothesis scoring |
| `run_phase10_train.py`, `run_phase10_validate.py`, `run_phase10_blind_massive.py`, `run_phase10_diagnostics.py`, `phase10_data.py` | Phase-10 domain-general retrieval |
| `run_phase11_train.py`, `run_phase11_validate.py`, `run_phase11b_backbone_capacity.py`, `phase11_noise.py`, `phase11_protocol.py` | Phase-11 noise robustness |
| `run_phase12_candidate_access.py`, `run_phase12_context_audit.py`, `run_phase12_context_baselines.py`, `run_phase12_g11_scorer.py`, `run_phase12_sgd_baseline.py`, `run_phase12_sparse_compare.py`, `run_phase12_sparse_scorer.py`, `phase12_context.py`, `phase12_sgd_inventory.py`, `freeze_phase12_context.py`, `prepare_phase12_label_review.py` | Phase-12 SGD context work |
| `rivals/` | Rival adapters: `base.py`, `nli.py` (`nli-bart`, `facebook/bart-large-mnli`), `st.py` (`st-centroid`, `all-MiniLM-L6-v2`), `jev.py` (dormant; activates only with `TYPESAFE_API_KEY` + `typesafe_sdk`, otherwise recorded as *not measured*) |

### Configs (`research/configs/`)

| Config | Status |
|---|---|
| `candidate_head_micro.json` | `SCAFFOLDED` |
| `ordinal_micro.json` | `IMPLEMENTED_NOT_RESEARCH_VALIDATED` |

### Results (`research/results/`)

`cpu_51_language_sweep.json` (MASSIVE all 51 languages, CPU), `t4_colab_benchmark.json` (T4 Colab run), `app_benchmark.json` (application workflows), `rival_benchmark.json` (rival harness), `post_fix_baseline.json`, `release_checklist.json`.

---

## 13. Benchmarks and Evaluation Results

> **Audit note (2026-09-23):** These are historical capability measurements, not release certification. Some datasets were in the training mix (Banking77 / AG News evaluation reuse found by the Phase 1–9 audit). Jev figures are third-party published, never measured here — treat as indicative. Full data: `research/results/`, narrative: [`BENCHMARKS.md`](BENCHMARKS.md).

### 13.1 Typed-Decisions Benchmark (400 cases, 2000 decisions)

| Model | Accuracy | Soft Acc | Brier | ECE | Score MAE |
|---|---|---|---|---|---|
| `veya-typed-decisions` | **0.766** | 0.471 | 0.061 | 0.213 | 0.242 |
| `veya` | 0.361 | 0.332 | 0.316 | 0.175 | 0.694 |
| `veya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |

Reference rows: teacher ceiling 0.735 · majority class 0.461 · random guess 0.318. The base checkpoints sit below the majority-class baseline — all capability on this benchmark comes from fine-tuning.

| Workflow | `veya-typed-decisions` |
|---|---|
| Agent trace observability | 0.730 |
| Customer service | 0.764 |
| Invoice processing | 0.804 |
| Security incidents | 0.766 |

### 13.2 Public Datasets

| Dataset | `veya` | `veya-multilingual` | `veya-typed-decisions` | Jev (published) |
|---|---|---|---|---|
| AG News (4 labels) | 0.950 | 0.930 | **0.953** | 0.910 |
| DAIR Emotion (6 labels) | 0.595 | 0.530 | **0.600** | 0.480 |
| Banking77 (77 labels) | 0.425 | 0.425 | **0.492** | 0.870 |

Historical packed Banking77 is a clear loss from the shared option-token budget; the audited release keeps packed Choice through 19 candidates and routes 20+ through K=8 retrieval + rerank. That route remains domain-specific: sealed CLINC150 Recall@8 = 35.6%, conditional accuracy = 98.03%, end-to-end accuracy = 34.9%.

**Rival harness** (controlled run, macOS CPU, seed 13, N=40/suite, 2026-09-29; byte-identical `bench_apps` cases through the same `metrics()` pipeline): `veya/typed-decisions` beats `st-centroid` on 8/10 suites — macro accuracy **0.721 vs 0.509**, macro ECE 0.166 vs 0.168. The rival wins banking77 (0.475 vs 0.325) and moderation (0.525 vs 0.500). `veya/english` and `veya/multilingual` macro accuracies: 0.719 and 0.696. `nli-bart` not run; `jev-typesafe` not measured.

### 13.3 Speed (Tesla T4)

| Questions per call | `veya` | `veya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms | **72.3 ms** |
| 50 | 771.3 ms | **337.4 ms** |

103–332 questions/sec batched. Jev independently measured at 236–276 ms p50 — Veya answers one question roughly **6–7× faster**.

### 13.4 Calibration

| Checkpoint | As shipped | Temperature refit |
|---|---|---|
| `veya` | 0.466 | **0.081** |
| `veya-multilingual` | 0.314 | **0.106** |
| *Jev (published)* | — | *0.246* |

Both ship over-confident; `veya-multilingual` ships with no fitted temperatures at all. Refitting one temperature per (question type, option count) on held-out data is the single highest-value fix available.

### 13.5 Multilingual (51-language MASSIVE sweep)

Intent, 20 options (random = 0.050):

| Metric | `veya` | `veya-multilingual` |
|---|---|---|
| Macro accuracy | 0.2269 | **0.3661** |
| Macro ECE (lower better) | 0.7331 | **0.3869** |
| Languages clearing 3× random | 23 / 51 | **45 / 51** |

| Task | `veya` | `veya-multilingual` |
|---|---|---|
| MASSIVE intent — English | **0.783** | 0.657 |
| MASSIVE intent — other languages | 0.306 | **0.451** |
| MASSIVE scenario — English | **0.603** | 0.560 |
| MASSIVE scenario — other languages | 0.281 | **0.439** |
| XNLI — English | **0.860** | 0.843 |
| XNLI — other languages | 0.521 | **0.731** |

English MASSIVE intent per checkpoint: `veya` **0.820** / multilingual 0.680. The English checkpoint does not degrade gracefully outside English — it collapses and stays confident: **Khmer 0.000 accuracy at 0.952 confidence**; mean confidence never drops below 0.885 at any accuracy level, so confidence gating cannot catch it — which is why routing happens *before* the forward pass.

### 13.6 Application Workflows (400 cases each)

| Theme | `veya` | `veya-multilingual` | `veya-typed-decisions` | Data |
|---|---|---|---|---|
| Email spam | **0.993** | **0.993** | 0.958 | in training |
| Phishing | 0.980 | **0.993** | 0.940 | in training |
| LLM guardrails (jailbreak) | 0.708 | 0.755 | **0.762** | held out |
| Moderation (toxicity) | **0.530** | 0.525 | **0.530** | held out |
| RAG passage relevance | 0.625 | **0.657** | 0.625 | in training |
| Support triage (10-way queue) | 0.502 | **0.522** | 0.505 | in training |
| Model routing (domain) | 0.639 | 0.123 | **0.659** | held out |

Email spam and phishing score ~0.99 with ECE ≈ 0.01 but were in the training mix — not generalization evidence. Moderation on held-out toxic-chat is 0.530 with macro-F1 0.400. Guardrails at 0.708–0.762 is the honest jailbreak-detection number (deepset prompt-injections measured 0.698 separately).

### 13.7 Known Limitations

- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400) — barely above chance on a balanced split.
- **Treat 20+ candidate Choice as retrieval-dependent** — the audited K=8 retriever does not generalize reliably across schemas.
- **Both checkpoints ship over-confident** — fit temperatures on your own data.
- **`veya` collapses outside English; `veya-multilingual` is weaker on English** — route. (Served Score is categorical; the ordinal implementation is research-only and not loaded by the audited manifest.)

---

## 14. Operational Runbook (Summary)

Seven procedures in `docs/RUNBOOK.md`:

| # | Procedure | Essentials |
|---|---|---|
| 1 | **Drift Alert Triggered** | `drift_monitor.drift_report` / `drift_triggers`; alert types `risk_breach` (Wilson 95% UCB > ε), `config_mismatch` (>1 `config_hash` per period), `coverage_drop` (< 0.8 × first-period baseline), `distribution_shift` (KS test p < 0.01 on 10-bin confidence histograms) |
| 2 | **Risk Exceeds Epsilon** | Tighten the selective threshold (0.02/0.05/0.10 buckets) or keep `operating_epsilon: "disabled"`; investigate with G1–G4 gate reports; never re-select a threshold on the data that showed the breach |
| 3 | **Health Probe Failing** | `curl /readyz` → triage by latched component (`model_loaded`, `policy_loaded`, `calibration_loaded`, `shortlist_loaded`, `hashes_valid`, `startup_failed`); a manifest mismatch kills the model service before it binds |
| 4 | **High Latency / Overload** | Check HPA first (70% CPU, 2–8 replicas); grep `model_busy` (3rd concurrent request → 503, `Retry-After: 5`); compare against the `runtime_audit.json` baseline; scale **out** only |
| 5 | **Checkpoint Update** | Follow `docs/CHANGE_PROTOCOL.md` → update `models/production/` → re-mint every affected SHA-256 pin in `manifest.json` → regenerate metadata (`python -m research.veyabench.release_metadata`) → verify (`tests/test_artifact_integrity.py` + `load_release_manifest`) → rebuild and redeploy |
| 6 | **Corrupted Artifact Recovery** | Identify the failing hash in model-service logs; `shasum -a 256` the file; restore from a known-good release tag; restart (readiness is sticky) |
| 7 | **Graceful Shutdown Issues** | Budget: `terminationGracePeriodSeconds` (45) > preStop (5s) + drain (30s); raise `SHUTDOWN_DRAIN_S` if long in-flight work must finish; `docker compose stop --timeout 45` |

---

## 15. Production Readiness

**Status: `IMPLEMENTATION_READY_PRODUCTION_CERTIFICATION_BLOCKED`** (matrix format `veya-production-readiness-matrix-v1`, release `0.3.5-audit-baseline.1`, generated 2026-09-30).

- **PASS: 20/25 · BLOCKED_EXTERNAL: 5/25 (G6, G8, G15, G20, G24) · FAIL: 0/25 · NOT_IN_SCOPE: 0/25**
- Manifest SHA-256: `cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de`; all **15 runtime files match, 0 drift** (31 strict integrity tests).
- Blocked on exactly three external inputs: **target-domain labeled data** (G6/G8), **legal disposition** for 6 flagged datasets (G15), and **live shadow traffic + ground truth** (G20).

| G-Node | Area | Status |
|---|---|---|
| G0 | Governance | PASS |
| G1 | Scope contract | PASS |
| G2 | Structured gap audit | PASS |
| G3 | Evaluation integrity (contamination) | PASS |
| G4 | Calibration reporting | PASS |
| G5 | Selective safety (ε guarantee, per-slice gates) | PASS |
| G6 | Prospective evaluation harness | **BLOCKED_EXTERNAL** |
| G7 | Robustness (surface + paraphrase noise) | PASS |
| G8 | Blind prospective verdict (P1–P6) | **BLOCKED_EXTERNAL** |
| G9 | API contracts | PASS |
| G10 | Concurrency | PASS |
| G11 | Performance audit | PASS |
| G12 | Reliability | PASS |
| G13 | Security | PASS |
| G14 | Supply chain (lock, SBOM, hash pins) | PASS |
| G15 | Licensing | **BLOCKED_EXTERNAL** |
| G16 | Observability (telemetry, request IDs) | PASS |
| G17 | Drift monitoring | PASS |
| G18 | Health endpoints | PASS |
| G19 | Containerization & orchestration | PASS |
| G20 | Live-traffic shadow evaluation | **BLOCKED_EXTERNAL** |
| G21 | Tiered CI/CD gates | PASS |
| G22 | Reproducible release metadata | PASS |
| G23 | Operator documentation | PASS |
| G24 | Final readiness verdict | **BLOCKED_EXTERNAL** |

**Supported scope (v1):** English; choice/noul/score-categorical/systemone; ≤ 100 candidates (packed ≤ 19, shortlist-K8 above 19); selective automation with defer; target-domain deployment. Unsupported: multilingual, moderation, arbitrary schemas, unrestricted zero-shot, ordinal score, autonomous operation without deferral.

---

## 16. Shadow Deployment

Operational procedure for live-shadow evaluation (gate G20). Status: **READY in-repository; BLOCKED_EXTERNAL on live traffic and cluster access.** A shadow deployment runs a full copy of the production stack on mirrored traffic, records telemetry for offline evaluation, never serves clients, and never affects production responses.

**Six steps** (`docs/SHADOW_DEPLOYMENT.md`):

1. **Deploy the shadow instance** — `kubectl apply -k k8s/shadow-overlay/` (namePrefix `shadow-`, `shadow-mode="1"`, 1 replica, HPA 1–2, disjoint selectors).
2. **Configure traffic routing** — Istio/Envoy traffic mirroring (`mirrorPercentage`, start ~5%) or application-level request duplication; strip/allowlist the `Origin` header; mind the per-IP rate budget (12 weight/minute) and `MAX_INFLIGHT=2`.
3. **Verify shadow health** — `/readyz` 200 with `X-Veya-Model-Hash` / `X-Veya-Policy-Hash` join headers; cold start ~34s (2.2 GB checkpoint), startup budget 120s.
4. **Monitor collection** — ship `'"shadow":true'` telemetry lines off the pod before teardown.
5. **Analyze results** — assemble records (telemetry + release hashes + served prediction + ground truth), flatten `slice_metadata` for `drift_monitor`, run drift triggers, selective gates at a candidate operating point, and the full `veya-shadow-evaluation-v1` verdict.
6. **Teardown** — preserve telemetry, then `kubectl delete -k k8s/shadow-overlay/` (production resources untouched).

**Sample size** (`sample_size.py`, Wilson inversion, ε=0.02, confidence 0.95): **189 accepted decisions** (zero accepted errors) clear the risk gate — 188 do not (fail-closed); **957 total** records at `c_min = 593/3000` (0.1977). With up to 3 observed accepted errors: 437 accepted / 2211 total.

---

## 17. Prospective Blind Workflow

Pre-registered blind evaluation for gates G6/G8. Status: **BLOCKED_EXTERNAL** — implemented, tested end-to-end with synthetic data, waiting on a target-domain labeled dataset (Banking77/CLINC150 are the controlled training side and are screened out by P1).

**CLI:**

```bash
python -m research.veyabench.run_prospective_blind \
  --config artifacts/release/prospective_blind_config.json \
  --data /path/to/target-domain-labeled.jsonl \
  --output artifacts/release/prospective_blind_result.json
```

| Exit code | Meaning |
|---|---|
| 0 | PASS — all six primary gates hold |
| 1 | FAIL — documented failure (one-line summary + `primary_gates`) |
| 2 | Validation/runtime error — nothing evaluated, no result written |

**Six primary gates (P1–P6):**

| Gate | Criterion |
|---|---|
| P1 data validity | Contamination screens flag ≤ tolerance (default 0) blind texts vs the controlled training side |
| P2 selective safety | Clean-slice accepted-risk UCB95 ≤ frozen ε (G1) |
| P3 useful coverage | Clean-slice coverage ≥ frozen `c_min` |
| P4 shift robustness | G2 (per-shift-slice risk ≤ ε) AND G3 (shift coverage ≥ retention × clean) AND G4 (no slice below collapse floor) |
| P5 calibration transfer | Freeze record pins the temperature artifact; frozen temperature applied, never refit |
| P6 operational viability | Caller-supplied operational measurements with no failures |

Failure precedence P1 > P2 > P3 > P4 > P5 > P6. Template ε=0.10, `c_min`=0.80 are pre-registration decisions fixed before unblinding. Provenance is verified before any statistic: config identity, release manifest hash, frozen selective policy (byte-exact copy with only `operating_epsilon` switched to a frozen threshold), frozen temperature (1.971127355603397), dataset SHA-256, and training-side corpus for contamination screens. Calibration/test split isolation fails closed on path or content-hash collisions. Output is written once with exclusive creation (a second run against the same output refuses to overwrite).

---

## 18. Documentation Index

### Root

| File | Description |
|---|---|
| `README.md` | Project overview and quickstart |
| `BENCHMARKS.md` | Benchmark tables, audit note, stated limits |
| `MASTER.md` | This document — master reference |
| `pyproject.toml` / `setup.py` / `requirements.lock` | Packaging and pinned dependencies |

### `docs/`

| File | Description |
|---|---|
| `CHANGE_PROTOCOL.md` | Procedure for changing manifest-pinned files (re-mint rules) |
| `CONCURRENCY.md` | Concurrency model and documented guarantees |
| `OPERATIONS.md` | Production deployment, configuration, model update and activation procedures |
| `PRODUCTION_READINESS.md` | G0–G24 readiness matrix (evidence-recomputed) |
| `PRODUCTION_SCOPE.md` | Production scope narrative (supported / unsupported) |
| `PROSPECTIVE_BLIND_WORKFLOW.md` | G6/G8 prospective blind evaluation workflow |
| `RUNBOOK.md` | Incident response — 7 procedures |
| `SHADOW_DEPLOYMENT.md` | G20 shadow deployment procedure (6 steps + verified dry run) |
| `WEBSITE_ARCHITECTURE.md` | Website architecture |
| `WEBSITE_REDESIGN_AUDIT.md` | Website redesign audit |
| `audit/release_verdict.md` | Phase 1–9 audit verdict (research-release scope) |
| `audit/phase_1_9_matrix.md` | Phase 1–9 audit matrix |
| `audit/retired_experiments.md` | Retired experiment records |
| `audit/data_lineage.json` | Data lineage record |
| `audit/blind_clinc150_freeze.json` | Sealed CLINC150 freeze |
| `audit/blind_clinc150_val_freeze.json` | Sealed CLINC150 validation freeze |
| `audit/data/` | External audit data (SGD and others) |
| `commercialization/SCOPING_PLAN.md` | Commercialization scoping plan |
| `commercialization/architecture_decision.md` | Commercialization architecture decision |
| `commercialization/workload_contract.md` | Workload contract |
| `commercialization/final_report.md` | Final report |
| `commercialization/verification_report.md` | Verification report |
| `commercialization/blind_hwu64_freeze.json` | HWU64 blind freeze |
| `commercialization/workflows/` | Workflow documents |
| `phase10/architecture_decision.md` | Phase-10 architecture decision |
| `phase10/blind_massive_freeze.json` | Phase-10 blind MASSIVE freeze |
| `phase10/final_report.md` | Phase-10 final report |
| `phase10/verification_report.md` | Phase-10 verification report |
| `phase11/architecture_decision.md` | Phase-11 architecture decision |
| `phase11/final_report.md` | Phase-11 final report |
| `phase11/verification_report.md` | Phase-11 verification report |
| `superpowers/plans/`, `superpowers/specs/` | Internal planning specs |

### `web/` and `research/`

| File | Description |
|---|---|
| `web/README.md` | Serving stack runbook (model service first, then gateway) |
| `web/EXECUTION_GRAPH.md` | Gateway execution graph |
| `web/PRELAUNCH_AUDIT.md` | Prelaunch audit |
| `web/VERIFICATION_REPORT.md` | Verification report |
| `research/README.md` | Research module overview |

### Key artifact documents

Release evidence lives in `artifacts/release/` (see [8.2](#82-release-artifacts-artifactsrelease)) and `artifacts/decisions/` (see [8.3](#83-decision-records-artifactsdecisions)). Historical evaluation reports and baseline locks referencing the pre-re-mint manifest (`c7febd23…`) are frozen evidence of the mint that produced them and are never rewritten to match the current manifest.

---

*End of master reference. Generated from repository evidence: `pyproject.toml`, `veya/`, `web/`, `models/production/`, `artifacts/`, `docs/`, `k8s/`, `.github/workflows/`, `tests/`, `research/`, and `BENCHMARKS.md`.*
