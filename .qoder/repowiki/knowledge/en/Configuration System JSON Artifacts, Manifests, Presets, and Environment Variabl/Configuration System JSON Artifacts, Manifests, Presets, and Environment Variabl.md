---
kind: configuration_system
name: 'Configuration System: JSON Artifacts, Manifests, Presets, and Environment Variables'
category: configuration_system
scope:
    - '**'
source_files:
    - veya/release_bundle.py
    - veya/manifest.py
    - veya/presets.py
    - veya/server.py
    - veya/__init__.py
    - pyproject.toml
    - models/production/rl_agent_config.json
---

## Overview

Veya has no centralized configuration framework (no Pydantic settings, no `.env` loader, no YAML config). Runtime configuration is split across three orthogonal mechanisms:

1. **JSON artifacts** — frozen model/router/shortlist/calibration metadata loaded from disk at startup.
2. **Release manifests** — a signed manifest format that binds every artifact by SHA-256 hash before loading.
3. **Python presets + environment variables** — in-process defaults for question schemas and research harness knobs.

There is no application-level `config.py`, no `settings.toml`, and no runtime env-var parsing in the production path (`veya/server.py`).

## Key Files and Packages

| File | Role |
|---|---|
| `veya/release_bundle.py` | Loads and validates release bundles; defines `veya-release-manifest-v2` format; enforces file hashes and cross-component consistency |
| `veya/manifest.py` | Builds a `build_manifest` dict from a checkpoint directory (`rl_agent_config.json`, `model.safetensors`) |
| `veya/presets.py` | In-memory question-schema factories (`triage_questions`, `email_questions`, `guard_questions`, `moderation_questions`, `router_questions`) |
| `veya/server.py` | HTTP server; reads `agent.cfg` (from the loaded checkpoint) but does not parse env vars |
| `veya/__init__.py` | Exposes preset functions and `__version__ = "0.3.5"` |
| `pyproject.toml` | Declares package version, Python >=3.10, and dependencies (`torch>=2.0.0`, `transformers>=4.48.0`, `safetensors>=0.4.0`, `huggingface_hub>=0.20.0`, `numpy>=1.20.0`) |
| `models/production/rl_agent_config.json` | The per-checkpoint encoder/tokenizer config consumed by `manifest.py` and `Agent.load` |
| `research/scripts/*.py` | Research harness scripts that set `USE_TF`, `USE_TORCH`, `TOKENIZERS_PARALLELISM`, `BENCH_N`, `VEYA_ORDINAL_CSV`, `VEYA_ORDINAL_LABELS` via `os.environ.setdefault` / `os.environ.get` |

## Architecture and Conventions

### Release Manifest Format

The authoritative production configuration entry point is `load_release_manifest` in `veya/release_bundle.py`. It expects a JSON manifest with this structure (enforced at load time):

```json
{
  "format": "veya-release-manifest-v2",
  "base_checkpoint": "...",
  "base_checkpoint_hash": "sha256",
  "model_config": "...",
  "model_config_hash": "sha256",
  "tokenizer_files": {"path": "sha256", ...},
  "routing": {
    "choice_router": "...",
    "choice_router_hash": "sha256"
  },
  "choice": {
    "shortlist_artifact": "...",
    "shortlist_hash": "sha256",
    "shortlist_k": 8,
    "packed_max_candidates": 32
  },
  "calibration": {
    "packed_choice": "...",
    "packed_choice_hash": "sha256",
    "hierarchical_choice": "...",
    "hierarchical_choice_hash": "sha256",
    "hierarchical_temperature": float
  },
  "selective_policy": "...",
  "selective_policy_hash": "sha256",
  "runtime_files": {"path": "sha256", ...}
}
```

On load, the function:
- Resolves every relative path against `root` and rejects paths escaping it (`resolve_manifest_path` raises `ValueError`).
- Computes SHA-256 of every referenced file and compares against the manifest's expected hash.
- Verifies `base_checkpoint_hash`, shortlist/checkpoint binding, route calibration K, router cutoff, and hierarchical temperature match the loaded bundle.
- Loads the selective policy bound to `(ROUTE, checkpoint_hash, shortlist_sha256, reranker_id)`.

### Checkpoint Manifest

`veya/manifest.build_manifest(model_dir, git_commit)` produces a lightweight descriptor from a checkpoint directory:

- Reads `rl_agent_config.json` inside the model dir.
- Hashes `model.safetensors` via streaming SHA-256.
- Derives `supported_languages` from whether the encoder name contains `mmbert`.
- Emits `architecture_version` from `veya.__version__`.
- Lists `decision_primitives: ["choice", "score", "noul"]`.

This is used by the build/release pipeline rather than at inference time.

### Question Schema Presets

`veya/presets.py` provides factory functions returning question dicts conforming to the internal schema validated by `Agent._to_internal`:

- `type`: one of `"choice"`, `"score"`, `"noul"`.
- `instructions`: string prompt describing the task.
- `criteria`: optional — either a `{label: description}` map (for `choice`) or a list of ordered labels (for `score`).

These are pure Python objects; they are not read from files and cannot be overridden at runtime through configuration.

### Server Configuration

`veya/server.py` is stdlib-only (`http.server.BaseHTTPRequestHandler`, `ThreadingHTTPServer`). It takes configuration exclusively through function arguments:

- `serve(agent, host="127.0.0.1", port=8080, bundle=None, hypothesis=None)`
- `make_handler(agent, bundle=None, hypothesis=None)`

Runtime behavior is controlled by the `agent.cfg` dict (loaded from the checkpoint's `rl_agent_config.json`) and by the attached `ReleaseBundle`/`Hypothesis` objects. There is no env-var reading in the server.

### Environment Variables

Environment variables exist only in the research harness under `research/scripts/` and `research/veyabench/`; they are never imported by the `veya` package itself. Observed variables:

| Variable | Default | Used By |
|---|---:|---|
| `USE_TF` | `"0"` | Benchmarks disable TensorFlow abseil runtime |
| `USE_TORCH` | `"1"` | Benchmarks force torch backend |
| `TOKENIZERS_PARALLELISM` | `"false"` | Prevents tokenizer parallelism deadlocks |
| `BENCH_N` | `"400"` | Number of benchmark iterations |
| `VEYA_ORDINAL_CSV` | `""` | Path to ordinal evaluation CSV |
| `VEYA_ORDINAL_LABELS` | `""` | Ordinal label order |

All are set via `os.environ.setdefault(...)` so they can be overridden by the caller process.

## Conventions and Constraints

- **Manifests are immutable**: `load_release_manifest` refuses any file whose SHA-256 differs from the manifest. This is enforced by `verified_path` raising `ValueError("... hash does not match the release manifest")`.
- **Path traversal is blocked**: `resolve_manifest_path` computes `os.path.commonpath((root_path, resolved))` and raises `ValueError("manifest path escapes the release root")` if the resolved path is outside `root`.
- **Cross-component consistency is enforced**: mismatched `hierarchical_k`, `packed_max_candidates`, `route_calibration.temperature`, and shortlist/checkpoint bindings all raise `ValueError` during bundle load.
- **Question schemas are validated in-process**: `_validate_request` calls `Agent._to_internal(question)` on every incoming request, rejecting malformed `instructions`/`criteria` fields.
- **No `.env` or YAML config exists**: the codebase contains no `python-dotenv`, `pydantic-settings`, `omegaconf`, or similar library usage; configuration is passed as function arguments or loaded from JSON.
- **Version is declared in two places**: `pyproject.toml` (`version = "0.3.5"`) and `veya/__init__.py` (`__version__ = "0.3.5"`); the server echoes `__version__` in `/version` responses.
- **Production model config lives alongside weights**: `models/production/rl_agent_config.json` is the canonical source for encoder/tokenizer hyperparameters consumed by `manifest.py` and `Agent.load`.