"""Boot the Veya Space in the order the Hugging Face runtime probe expects.

Three constraints collide here, and they rule out the obvious order:

1. The runtime probes ``app_port`` shortly after the container starts. A
   process that spends its first four minutes downloading 804 MB of weights
   looks like a failed Space, so the gateway has to bind first. ``web.server``
   answers ``/healthz`` without touching the model, which is the same liveness
   contract the audited deployment documents in docs/RUNBOOK.md.
2. ``web.model_server`` refuses to start until every manifest-pinned hash
   verifies, and then needs ~80 s more to load ModernBERT-large on CPU
   (artifacts/release/runtime_audit.json: 90.8 s cold start, 1638.9 MB RSS).
3. ``/readyz`` must stay 503 until that load latches. ``web.server`` re-probes
   the model service every READINESS_TTL_S and only sets ``startup_failed`` when
   the service answers /health but cannot present valid release bindings, so a
   gateway that starts before its backend keeps retrying rather than poisoning
   itself (web/server.py: _refresh_readiness).

Hence: bind the gateway, fetch and verify the checkpoint, load the model, then
supervise both children and exit if either dies so the runtime restarts the
container instead of serving a half-live Space.

The model service binds 127.0.0.1 inside the container and is never published,
which keeps web/README.md's "do not expose the model service directly" true
even though the Space itself is public.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "space" / "fetch_checkpoint.py"
MODEL_PORT = int(os.environ.get("VEYA_MODEL_PORT", "8080"))

# Cold start budget: 804 MB download on Space-internal bandwidth, ~11 s to hash
# it, ~79 s to load it, plus the first /version probe. 900 s is generous
# without letting a wedged container sit on the free tier forever.
READY_TIMEOUT_S = float(os.environ.get("VEYA_READY_TIMEOUT_S", "900"))
GATEWAY_TIMEOUT_S = 90.0


def log(message: str) -> None:
    print(f"[boot] {message}", flush=True)


def web_port() -> int:
    """Hugging Face owns the port choice; the repo convention is the fallback."""
    for name in ("PORT", "VEYA_WEB_PORT"):
        value = os.environ.get(name)
        if value:
            return int(value)
    return 7860


def public_origin() -> str:
    """Canonical HTTPS origin for the gateway's public-binding guard.

    web/server.py raises "Public binding requires VEYA_PUBLIC_ORIGIN" for any
    non-loopback VEYA_WEB_HOST, and SPACE_HOST is injected by the runtime, so
    deriving it keeps the value from being a hand-copied string that can drift
    from the actual Space name.
    """
    value = os.environ.get("VEYA_PUBLIC_ORIGIN", "").rstrip("/")
    if value:
        return value
    host = os.environ.get("SPACE_HOST", "").strip()
    return f"https://{host}" if host else ""


def answered(url: str, timeout: float, ok: tuple[int, ...], label: str) -> bool:
    deadline = time.perf_counter() + timeout
    last = "no response"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                if response.status in ok:
                    return True
                last = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            if exc.code in ok:
                return True
            last = f"HTTP {exc.code}"
        except Exception as exc:
            last = f"{type(exc).__name__}"
        time.sleep(1.0)
    log(f"{label} never answered within {timeout:.0f}s (last: {last})")
    return False


def main() -> int:
    port = web_port()
    origin = public_origin()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "VEYA_WEB_HOST": os.environ.get("VEYA_WEB_HOST", "0.0.0.0"),
            "VEYA_WEB_PORT": str(port),
            "VEYA_MODEL_PORT": str(MODEL_PORT),
            "VEYA_MODEL_URL": os.environ.get("VEYA_MODEL_URL", f"http://127.0.0.1:{MODEL_PORT}"),
            "VEYA_DEVICE": os.environ.get("VEYA_DEVICE", "cpu"),
        }
    )
    if origin:
        env["VEYA_PUBLIC_ORIGIN"] = origin
    log(f"gateway {env['VEYA_WEB_HOST']}:{port} origin={origin or '<unset>'} model={env['VEYA_MODEL_URL']}")

    children: dict[str, subprocess.Popen] = {}

    def shutdown(code: int) -> int:
        for name, proc in children.items():
            if proc.poll() is None:
                log(f"stopping {name} (pid {proc.pid})")
                proc.terminate()
        deadline = time.perf_counter() + 15
        for proc in children.values():
            while proc.poll() is None and time.perf_counter() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                proc.kill()
        return code

    def on_signal(signum, _frame):
        log(f"received signal {signum}")
        raise SystemExit(shutdown(0))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    started = time.perf_counter()

    children["gateway"] = subprocess.Popen(
        [sys.executable, "-m", "web.server"], cwd=str(ROOT), env=env
    )
    if not answered(f"http://127.0.0.1:{port}/healthz", GATEWAY_TIMEOUT_S, (200,), "gateway /healthz"):
        return shutdown(1)
    log(f"gateway live in {time.perf_counter() - started:.1f}s; /readyz stays 503 until the model latches")

    fetch = subprocess.run([sys.executable, str(FETCH)], cwd=str(ROOT), env=env)
    if fetch.returncode != 0:
        log("checkpoint is unavailable or unverified; not starting the model service")
        return shutdown(fetch.returncode)

    children["model"] = subprocess.Popen(
        [sys.executable, "-m", "web.model_server"], cwd=str(ROOT), env=env
    )
    if not answered(
        f"http://127.0.0.1:{MODEL_PORT}/health", READY_TIMEOUT_S, (200,), "model service /health"
    ):
        return shutdown(1)
    if not answered(f"http://127.0.0.1:{port}/readyz", 120.0, (200,), "gateway /readyz"):
        return shutdown(1)
    log(f"audited release latched in {time.perf_counter() - started:.1f}s")

    while True:
        for name, proc in children.items():
            code = proc.poll()
            if code is not None:
                log(f"{name} exited with {code}; exiting so the runtime restarts the Space")
                return shutdown(code or 1)
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
