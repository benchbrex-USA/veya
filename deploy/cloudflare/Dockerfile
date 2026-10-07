# Cloudflare Container image for the verified production Veya release.
# Build context is the repository root. The checkpoint is not copied into the
# image. Boot fetches it from the private R2 hostname and verifies the hash.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.lock ./requirements.lock
# The lock was resolved on macOS, where torch has no CUDA wheels. Install the
# same pinned version from the CPU index first so Linux does not pull NVIDIA.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0 \
 && pip install --no-cache-dir -r requirements.lock \
 && python - <<'PY'
import torch
import transformers
import huggingface_hub
import safetensors
import numpy
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("safetensors", safetensors.__version__)
print("numpy", numpy.__version__)
assert not torch.cuda.is_available()
PY

FROM python:3.11-slim AS production

RUN groupadd -r veya && useradd -r -g veya -d /app -s /sbin/nologin veya
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY veya/ ./veya/
COPY web/__init__.py web/server.py web/contracts.py web/model_server.py ./web/
COPY web/static/ ./web/static/
RUN test -f /app/web/static/site-config.json
COPY artifacts/release/ ./artifacts/release/
COPY deploy/cloudflare/fetch_release.py deploy/cloudflare/supervise.py deploy/cloudflare/release-pins.json ./boot/
COPY models/production/ /tmp/model-src/
RUN mkdir -p /app/models/production \
    && cd /tmp/model-src \
    && find . -type d -name .cache -prune -o -type f ! -name '*.safetensors' -print \
    | while read -r file; do \
        mkdir -p "/app/models/production/$(dirname "$file")"; \
        cp "$file" "/app/models/production/$file"; \
      done \
    && rm -rf /tmp/model-src \
    && find /app -name '*.safetensors' -delete \
    && python -m compileall -q veya web boot \
    && chown -R veya:veya /app
USER veya
EXPOSE 8000
ENV VEYA_SHADOW_MODE=0 \
    VEYA_LOG_LEVEL=INFO \
    VEYA_MODEL_DIR=/app/models/production \
    VEYA_MODEL_PATH=/app/models/production/model.safetensors \
    VEYA_WEB_HOST=0.0.0.0 \
    VEYA_WEB_PORT=8000 \
    VEYA_PUBLIC_ORIGIN=https://localhost \
    VEYA_MODEL_URL=http://127.0.0.1:8080 \
    VEYA_DEVICE=cpu \
    VEYA_RELEASE_ID=0.3.5-audit-baseline.1 \
    VEYA_CHECKPOINT_URL=http://veya-r2.internal/releases/0.3.5-audit-baseline.1/model.safetensors \
    VEYA_CHECKPOINT_SHA256=891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c \
    VEYA_CHECKPOINT_BYTES=842609210 \
    VEYA_MANIFEST_SHA256=cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de \
    PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=8)"]
CMD ["python", "/app/boot/supervise.py"]
