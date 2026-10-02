---
kind: dependency_management
name: Python Dependency Management via pyproject.toml, pip, and Dependabot
category: dependency_management
scope:
    - '**'
source_files:
    - pyproject.toml
    - setup.py
    - .github/dependabot.yml
    - .github/workflows/ci.yml
    - .github/workflows/release.yml
    - tests/test_packaging.py
---

## Approach

Veya is a Python package (`veya`) distributed on PyPI. Dependencies are declared exclusively in `pyproject.toml` using PEP 621 metadata; there is no `requirements.txt`, `Pipfile`, Poetry lockfile, or vendored third-party source tree. The build backend is `setuptools.build_meta` (with `setuptools>=61.0`). A minimal `setup.py` shim exists only for tooling that still invokes `setup()` directly — all real metadata lives in `pyproject.toml`.

The runtime dependency surface is intentionally small:

- `torch>=2.0.0`
- `transformers>=4.48.0`
- `safetensors>=0.4.0`
- `huggingface_hub>=0.20.0`
- `numpy>=1.20.0`

All constraints are lower bounds (`>=`); no upper bounds or exact pins are used in the manifest. There is no private PyPI index configured anywhere in the repo — CI installs from the default PyPI plus the CPU-only torch wheel index.

## Key Files

- `pyproject.toml` — single source of truth for package name, version (`0.3.5`), `requires-python = ">=3.10"`, classifiers, dependencies, and setuptools package discovery.
- `setup.py` — one-line shim that delegates to setuptools; explicitly documented as a compatibility layer.
- `.github/dependabot.yml` — schedules weekly updates for both `pip` and `github-actions` ecosystems.
- `.github/workflows/ci.yml` — installs deps with `pip install -e .` after installing CPU-only torch via `--index-url https://download.pytorch.org/whl/cpu`; sets `USE_TF=0`, `USE_TORCH=1`, `TOKENIZERS_PARALLELISM=false` to avoid TF/abseil deadlocks during import.
- `.github/workflows/release.yml` — builds with `python -m build`, validates with `twine check dist/*`, publishes to PyPI via trusted publishing (`pypa/gh-action-pypi-publish@release/v1`) using GitHub OIDC against the `pypi` environment (no API token).
- `tests/test_packaging.py` — runtime assertions that enforce packaging conventions (see Conventions below).

## Architecture and Conventions

**Single manifest.** All package metadata is centralized in `pyproject.toml`. The test suite parses it with a regex (avoiding `tomllib`, which is not available on Python 3.10) and cross-checks multiple sources.

**Lower-bound pinning.** Every dependency uses a `>=` floor rather than an exact version. The comment in `test_packaging.py` explains the rationale: "Checkpoints run on answerdotai/ModernBERT-large, which transformers only knows from 4.48", so the `transformers>=4.48.0` floor is treated as a hard requirement.

**No lockfile.** There is no `poetry.lock`, `uv.lock`, or equivalent. Reproducibility relies on the CI matrix pinning specific Python versions (3.10, 3.11, 3.13) and the explicit lower bounds in `pyproject.toml`. Development uses a local `.venv`.

**CPU-only torch in CI.** Both CI and release workflows install torch from the CPU-only wheel index (`https://download.pytorch.org/whl/cpu`) to keep jobs fast and avoid CUDA wheels.

**Trusted publishing.** The release workflow uses GitHub's OIDC-based trusted publisher for PyPI (documented in the workflow comments). No secret tokens are stored.

**Dependabot coverage.** Two ecosystems are tracked: `pip` (weekly, max 5 open PRs, labeled `dependencies`) and `github-actions` (weekly, labeled `dependencies, ci`). The comment in `.github/dependabot.yml` states the security rationale: "Actions are the other half of the supply chain: a compromised action runs with the workflow's token. Keep them current, and prefer pinning to a SHA for third-party ones."

## Conventions and Constraints

Enforced by tests (`tests/test_packaging.py`, invoked from CI):

1. **Metadata lives in `pyproject.toml` only.** `setup.py` must not contain `python_requires`, `install_requires`, or `classifiers`; the test asserts these fields are absent and documents the rule: "metadata belongs in pyproject.toml only".
2. **CI Python versions must not fall below `requires-python`.** The test extracts all quoted `X.Y` strings from `.github/workflows/ci.yml` and rejects any version below the `requires-python` floor (currently 3.10).
3. **Classifiers must not advertise Python versions below the declared floor.** Any `Programming Language :: Python :: X.Y` classifier below `requires-python` is flagged.
4. **`transformers` floor must be ≥ 4.48.** The test asserts this because the frozen checkpoints require ModernBERT support introduced in that version.
5. **Release tag must match `pyproject.toml` version.** Both `ci.yml` and `release.yml` extract the version string from `pyproject.toml` and compare it against the git tag (`refs/tags/v*`); a mismatch aborts the build/publish.
6. **Built distributions must pass `twine check`.** Executed in both CI and release workflows before artifacts are uploaded.
7. **PyPI publish requires a Trusted Publisher environment.** The release workflow requires the `pypi` GitHub environment with OIDC permissions; without it, publishing fails (documented in the workflow comments).

Not enforced but observed:

- No private registry or `PYPI_URL` / `PIP_INDEX_URL` configuration is present.
- No `requirements.txt` files exist at the repository root; only `data/external/sgd/sgd_x/requirements.txt` appears under a vendored dataset generator and is unrelated to the `veya` package.
- No vendoring of third-party Python packages into the tree.