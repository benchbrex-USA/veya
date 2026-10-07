"""Start Veya only after the checkpoint verifies.

Fetch runs to completion first. The model server and the gateway are children.
SIGTERM and SIGINT stop both. If either child exits, the other is stopped and
this process exits with that child's status. A failed fetch never leaves the
gateway running.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

PIN_PATH = Path("/app/boot/release-pins.json")
PIN_KEYS = (
    "VEYA_RELEASE_ID",
    "VEYA_CHECKPOINT_SHA256",
    "VEYA_CHECKPOINT_BYTES",
    "VEYA_MANIFEST_SHA256",
)


def pins_match(environ: Mapping[str, str], pins: Mapping[str, str]) -> None:
    """Fail closed when the injected environment disagrees with the image pins."""
    for key in PIN_KEYS:
        expected = str(pins[key])
        actual = str(environ.get(key, ""))
        if actual != expected:
            raise SystemExit(f"release pin mismatch for {key}")


def stop_children(children: Sequence[subprocess.Popen[bytes]], timeout: float = 20) -> None:
    for proc in children:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + timeout
    for proc in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def supervise(
    fetch_argv: Sequence[str],
    child_argvs: Sequence[Sequence[str]],
    environ: Mapping[str, str] | None = None,
    pins: Mapping[str, str] | None = None,
) -> int:
    env = os.environ if environ is None else environ
    if pins is not None:
        pins_match(env, pins)
    completed = subprocess.run(list(fetch_argv), env=dict(env), check=False)
    if completed.returncode != 0:
        return completed.returncode or 1
    run_env = dict(env)
    children = [subprocess.Popen(list(argv), env=run_env) for argv in child_argvs]

    def handle(signum: int, _frame: object) -> None:
        stop_children(children)
        raise SystemExit(128 + signum)

    previous = {
        sig: signal.signal(sig, handle) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        while True:
            for proc in children:
                code = proc.poll()
                if code is not None:
                    stop_children([other for other in children if other is not proc])
                    return code
            time.sleep(0.2)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    pins = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    code = supervise(
        [sys.executable, "/app/boot/fetch_release.py"],
        [
            [sys.executable, "-m", "web.model_server"],
            [sys.executable, "-m", "web.server"],
        ],
        pins=pins,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
