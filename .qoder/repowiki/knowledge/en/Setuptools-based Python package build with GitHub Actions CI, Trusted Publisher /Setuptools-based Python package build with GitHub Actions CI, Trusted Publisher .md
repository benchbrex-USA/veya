---
kind: build_system
name: Setuptools-based Python package build with GitHub Actions CI, Trusted Publisher release, and supply-chain security gates
category: build_system
scope:
    - '**'
source_files:
    - pyproject.toml
    - setup.py
    - .github/workflows/ci.yml
    - .github/workflows/release.yml
    - .github/workflows/security.yml
    - .github/dependabot.yml
    - tests/test_packaging.py
---

## Build System Overview

Veya is a pure-Python distribution built with **setuptools** (PEP 517 backend) driven by `pyproject.toml`. There are no Makefiles, Dockerfiles, or shell build scripts — the entire build pipeline lives in GitHub Actions.

### Packaging metadata

- `pyproject.toml` declares the package (`name = "veya"`, `version = "0.3.5"`), `requires-python = ">=3.10"`, dependencies (`torch>=2.0.0`, `transformers>=4.48.0`, `safetensors>=0.4.0`, `huggingface_hub>=0.20.0`, `numpy>=1.20.0`), classifiers for Python 3.10–3.13, and a proprietary license.
- `setup.py` is a one-line shim that calls `setuptools.setup()` so legacy tooling still works; all metadata lives in `pyproject.toml` only.
- `tool.setuptools.packages` explicitly lists `veya`, `veya.experimental`, `web` because `assets/`, `research/`, and `notebooks/` sit at repo root and would otherwise confuse auto-discovery.
- `web/static/*` is included as package data via `tool.setuptools.package-data`.

### CI pipeline (`.github/workflows/ci.yml`)

Three jobs run on every PR and on pushes to `main`:

1. **test** — matrix over Python 3.10, 3.11, 3.13 on `ubuntu-latest`. Installs CPU-only torch (`https://download.pytorch.org/whl/cpu`) to keep jobs under a minute, installs the package editable (`pip install -e .`), runs an import smoke test, then executes individual test files: `test_router.py`, `test_criteria.py`, `test_download.py`, `test_shortlist.py`, `test_decision_model.py`, `test_packaging.py`, `test_email.py`.
2. **lint** — runs `ruff check veya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120` and byte-compiles every module under `veya/` and `tests/` with `python -m compileall`.
3. **build** — installs `build` + `twine`, runs `python -m build`, validates the dist with `twine check dist/*`, and on tag pushes asserts that the version string extracted from `pyproject.toml` equals the tag name stripped of its leading `v`.

Environment variables set globally: `USE_TF=0`, `USE_TORCH=1`, `TOKENIZERS_PARALLELISM=false` (the comment explains TF's abseil runtime can deadlock model construction).

### Release pipeline (`.github/workflows/release.yml`)

Triggered by pushing tags matching `v*` (or manually via `workflow_dispatch`). Three jobs:

- **build** — same install/test/build steps as CI, plus a second version-match assertion against the tag.
- **pypi** — publishes via `pypa/gh-action-pypi-publish@release/v1` using PyPI's **Trusted Publisher** flow (OIDC token, no API key). The workflow requires a pre-configured Trusted Publisher for `benchbrex-USA/veya` on environment `pypi`; `skip-existing: true` makes reruns safe after manual uploads.
- **github_release** — creates or updates a GitHub Release named after the tag and attaches the `dist/` artifacts.

### Security & supply-chain pipeline (`.github/workflows/security.yml`)

Runs on PRs, pushes to `main`, and weekly on a cron schedule:

- **secrets** — two-pass gitleaks scan (commit diff + full working tree) to catch secrets leaked in earlier commits.
- **deps** — parses `dependencies = [...]` from `pyproject.toml` into a temporary `requirements-audit.txt` and runs `pip-audit --strict --desc` against it.
- **codeql** — CodeQL analysis for Python with `security-extended` queries.
- **supply_chain** — static grep forbids unsafe deserialization patterns (`pickle`, `torch.load(`, `joblib.load(`, `yaml.load(`) and dynamic execution (`os.system(`, `subprocess.`, `shell=True`, `eval(`, `exec(`) anywhere under `veya/`.

### Dependency management

- `dependabot.yml` opens weekly PRs for both `pip` and `github-actions` ecosystems, capped at 5 open PRs and labeled `dependencies` / `ci`.

### Packaging tests (enforced invariant)

`tests/test_packaging.py` is itself a build-time contract:

- Asserts `pyproject.toml` declares `requires-python` and advertises specific Python versions via classifiers.
- No classifier may advertise a Python version below the declared floor.
- `transformers>=` must be at least `4.48` (required by ModernBERT support used by checkpoints).
- `setup.py` must not duplicate `python_requires`, `install_requires`, or `classifiers`.
- CI workflows must not use a Python version below `requires-python`.

The file is executed directly by CI (`python tests/test_packaging.py`) and also has a pytest entry point `test_recorded_checks_pass`.

### Artifacts produced

- `dist/` wheel/sdist archives uploaded as CI artifacts and published to PyPI.
- Frozen ML artifacts live outside the build system: `models/production/` holds the shipped safetensors models and tokenizer configs; `artifacts/decisions/`, `artifacts/commercialization/`, and `artifacts/experiments/phaseN/` hold experiment graphs, calibration JSON, and `.pt`/`.npz` research outputs. These are committed alongside source code rather than generated at build time.