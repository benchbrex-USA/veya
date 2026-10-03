# Veya serving stack for a Hugging Face Space (Docker SDK, free cpu-basic).
#
# The runtime tree mirrors the repository layout exactly: the release manifest
# (artifacts/release/manifest.json) resolves every artifact path relative to the
# release root and pins SHA-256 for veya/*.py, the checkpoint, model config,
# tokenizer, router, shortlist, both calibrations, and the selective policy
# (veya/release_bundle.py: resolve_manifest_path / load_release_manifest).
# Copying those sources byte-for-byte keeps the audited load gate intact, so no
# pinned file is edited here. deploy/hf-space/assemble.py re-verifies every
# pinned digest against the staged bytes before anything is uploaded.
#
# Two deliberate differences from the repository Dockerfile:
#
#   1. The checkpoint is not in the image. models/production/**/*.safetensors is
#      gitignored and the weights live in the private veyadynamics/veya model
#      repo; space/fetch_checkpoint.py downloads that one file at boot from a
#      pinned revision and asserts the manifest digest. A public Space must
#      never republish a private research release as LFS blobs.
#   2. The gateway listens on 7860 instead of 8000 because the runtime probes
#      app_port from the Space README frontmatter.

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0

WORKDIR /app

# requirements.lock is uv-compiled (uv pip compile pyproject.toml --all-extras)
# and pins torch==2.14.0 with no nvidia-* entries, because it was resolved on
# macOS where the torch wheel carries no CUDA dependencies. Installing that lock
# as-is on Linux would pull the 554 MB CUDA wheel plus ~3 GB of nvidia runtime
# packages. So the CPU build of the *same pinned version* goes in first from
# the PyTorch CPU index (torch-2.14.0+cpu, 196 MB); pip then sees torch==2.14.0
# already satisfied and installs the rest of the lock verbatim. PEP 440: a
# specifier without a local segment matches a wheel that has one.
COPY requirements.lock ./requirements.lock
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.lock \
 && python -c "import torch, transformers, huggingface_hub, safetensors, numpy; \
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('huggingface_hub', huggingface_hub.__version__)"

# Serving code. space/ holds the two Space-only entry points; they live outside
# web/ so the published package (pyproject packages = ["veya", "veya.experimental",
# "web"]) and its packaging tests are untouched by deployment scaffolding.
COPY veya/ ./veya/
COPY web/ ./web/
COPY space/ ./space/
COPY models/production/ ./models/production/
COPY artifacts/release/ ./artifacts/release/

# Pre-compile bytecode, then hand the tree to uid 1000 (the Hugging Face runtime
# identity). Ownership matters: fetch_checkpoint writes model.safetensors into
# /app/models/production and veya/agent.py rewrites tokenizer_config.json in
# place, so a read-only /app would fail the boot on a clean container.
RUN python -m compileall -q veya web space \
 && chown -R 1000:1000 /app

USER 1000

# Kept off the pinned-path set and out of the release root's verified subtree,
# but still inside /app: the checkpoint lands here at boot.
ENV PYTHONPATH=/app \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VEYA_DEVICE=cpu \
    VEYA_SHADOW_MODE=0 \
    VEYA_LOG_LEVEL=INFO \
    VEYA_MODEL_PORT=8080 \
    VEYA_MODEL_URL=http://127.0.0.1:8080 \
    VEYA_WEB_HOST=0.0.0.0 \
    VEYA_WEB_PORT=7860

EXPOSE 7860

# No HEALTHCHECK: the runtime probes app_port itself, and a container-level
# health failure during the ~4 minute cold start would only add noise. Liveness
# is /healthz, audited-release readiness is /readyz (503 until every manifest
# hash verifies and the model service publishes its release bindings).
CMD ["python", "space/boot.py"]
