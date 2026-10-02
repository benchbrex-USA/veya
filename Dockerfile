# Veya serving stack — multi-stage build.
#
# The runtime tree mirrors the repository layout exactly: the release manifest
# (artifacts/release/manifest.json) resolves every artifact path relative to
# the repository root and pins SHA-256 hashes for veya/*.py, the checkpoint,
# tokenizer, router, shortlist, calibrations, and the selective policy
# (veya/release_bundle.py: resolve_manifest_path / load_release_manifest).
# Copying the sources byte-for-byte therefore keeps the audited load gate
# intact; a changed pinned file must be re-minted per docs/CHANGE_PROTOCOL.md
# before the model service will start.

# ---- Stage 1: builder — resolve the pinned dependency set once ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Metadata first: the heavy torch install layer stays cached while sources change.
COPY requirements.lock pyproject.toml setup.py ./

# requirements.lock is uv-compiled (uv pip compile pyproject.toml --all-extras);
# pip honors the exact pins.
RUN pip install --no-cache-dir -r requirements.lock

# ---- Stage 2: production — slim runtime with only the serving tree ----
FROM python:3.11-slim AS production

# Non-root serving identity: no login shell, no home writes.
RUN groupadd -r veya && useradd -r -g veya -d /app -s /sbin/nologin veya

WORKDIR /app

# Installed packages from the builder. Both stages share the same base image,
# so the interpreter and the site-packages layout match.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Production code only. web/ carries both serving entry points: model_server.py
# starts the audited model service, server.py is the gateway and website.
# models/production/ is the base checkpoint named by the manifest;
# artifacts/release/ holds the manifest, the selective policy, and the
# benchmark evidence the website republishes.
COPY veya/ ./veya/
COPY web/ ./web/
COPY models/production/ ./models/production/
COPY artifacts/release/ ./artifacts/release/

# Pre-compile bytecode so the non-root user never needs to write __pycache__.
RUN python -m compileall -q veya web

USER veya

# The gateway listens here. The model service binds 127.0.0.1:8080 inside the
# container only and is never published (web/README.md: do not expose the
# model service directly).
EXPOSE 8000

# Defaults. A port-published container must bind a non-loopback address, and
# the gateway only permits that together with a canonical HTTPS origin
# ("Public binding requires VEYA_PUBLIC_ORIGIN", web/server.py); override
# VEYA_PUBLIC_ORIGIN for any real deployment. VEYA_MODEL_URL already defaults
# to the in-container model service (http://127.0.0.1:8080) and VEYA_DEVICE
# to cpu. VEYA_LOG_LEVEL and VEYA_MODEL_DIR are provisioned for future use.
ENV VEYA_SHADOW_MODE=0 \
    VEYA_LOG_LEVEL=INFO \
    VEYA_MODEL_DIR=/app/models/production \
    VEYA_WEB_HOST=0.0.0.0 \
    VEYA_WEB_PORT=8000 \
    VEYA_PUBLIC_ORIGIN=https://localhost \
    PYTHONUNBUFFERED=1

# /readyz is 200 only after the model service verified and latched the audited
# release (manifest, checkpoint, shortlist, calibration); /healthz stays
# liveness-only. CPU load of the production tree can exceed a minute, hence
# the longer start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=8)"]

# Both serving processes share this container because the model service binds
# 127.0.0.1:8080 only. The gateway runs as PID 1 (exec) and receives stop
# signals; the model service exits with the container. Mirrors the runbook in
# web/README.md: model service first, then the gateway.
CMD ["sh", "-c", "python -m web.model_server & exec python -m web.server"]
