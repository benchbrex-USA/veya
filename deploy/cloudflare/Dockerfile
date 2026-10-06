# Cloudflare Container image for the verified production Veya release.
# Build context is the repository root. The checkpoint is not copied into the
# image. Boot fetches it from the private R2 hostname and verifies the hash.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.lock pyproject.toml setup.py ./
RUN pip install --no-cache-dir -r requirements.lock

FROM python:3.11-slim AS production

RUN groupadd -r veya && useradd -r -g veya -d /app -s /sbin/nologin veya
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY veya/ ./veya/
COPY experts/ ./experts/
COPY quantum/ ./quantum/
COPY web/__init__.py web/server.py web/contracts.py web/model_server.py ./web/
COPY artifacts/release/ ./artifacts/release/
COPY deploy/cloudflare/fetch_release.py ./boot/fetch_release.py
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
    && python -m compileall -q veya experts quantum web boot \
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
    PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=8)"]
CMD ["sh", "-c", "python /app/boot/fetch_release.py && python -m web.model_server & exec python -m web.server"]
