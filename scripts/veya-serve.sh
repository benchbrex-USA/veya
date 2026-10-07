#!/bin/bash
# Auto-start Veya model server + gateway when the external drive is mounted.
# Installed as a macOS LaunchAgent that watches for the drive.

VEYA_ROOT="/Volumes/Parth Patel/Veya"
LOG_DIR="/tmp"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

if [ ! -d "$VEYA_ROOT/veya" ]; then
  echo "$(date): Drive not mounted at $VEYA_ROOT, exiting." >> "$LOG_DIR/veya-serve.log"
  exit 0
fi

already_running() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

if already_running 8080 && already_running 3000; then
  echo "$(date): Already running on :8080 and :3000, nothing to do." >> "$LOG_DIR/veya-serve.log"
  exit 0
fi

echo "$(date): Starting Veya serving stack..." >> "$LOG_DIR/veya-serve.log"

cd "$VEYA_ROOT"

if ! already_running 8080; then
  nohup "$PYTHON" -m web.model_server >> "$LOG_DIR/veya-model.log" 2>&1 &
  echo "$(date): model_server pid=$!" >> "$LOG_DIR/veya-serve.log"
fi

if ! already_running 3000; then
  VEYA_WEB_HOST=127.0.0.1 \
  VEYA_WEB_PORT=3000 \
  VEYA_MODEL_URL=http://127.0.0.1:8080 \
  VEYA_DEVICE=cpu \
  VEYA_PUBLIC_ORIGIN=https://veya.veya-8d0.workers.dev \
  VEYA_ALLOWED_ORIGINS="https://veya.veya-8d0.workers.dev" \
  nohup "$PYTHON" -m web.server >> "$LOG_DIR/veya-gateway.log" 2>&1 &
  echo "$(date): gateway pid=$!" >> "$LOG_DIR/veya-serve.log"
fi

for i in $(seq 1 36); do
  R=$(curl -s --max-time 5 http://127.0.0.1:3000/readyz 2>/dev/null)
  if echo "$R" | grep -q '"ready"'; then
    echo "$(date): READY after ~$((i*5))s" >> "$LOG_DIR/veya-serve.log"
    exit 0
  fi
  sleep 5
done

echo "$(date): Timed out waiting for readyz" >> "$LOG_DIR/veya-serve.log"
exit 1
