"""Process-isolated Veya Foundry bridge.

External research sources frequently require mutually incompatible Python,
CUDA, tokenizer, and runtime versions. Veya therefore communicates with each
source through a JSON-lines subprocess contract instead of importing foreign
packages into the main process.
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .contracts import VeyaRequest, VeyaResponse, request_to_dict, response_from_dict
from .expert import VeyaExpert


@dataclass(frozen=True)
class VeyaBridgeConfig:
    expert_id: str
    command: tuple[str, ...]
    cwd: Optional[Path] = None
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.expert_id.strip():
            raise ValueError("expert_id must not be empty")
        if not self.command:
            raise ValueError("command must not be empty")


class VeyaProcessExpert(VeyaExpert):
    """Run one Foundry expert behind a persistent JSON-lines process."""

    def __init__(self, config: VeyaBridgeConfig):
        self.config = config
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def expert_id(self) -> str:
        return self.config.expert_id

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = None
        if self.config.environment is not None:
            import os

            env = dict(os.environ)
            env.update(self.config.environment)
        self._proc = subprocess.Popen(
            self.config.command,
            cwd=None if self.config.cwd is None else str(self.config.cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._proc

    def predict(self, request: VeyaRequest) -> VeyaResponse:
        with self._lock:
            proc = self._ensure_process()
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(json.dumps(request_to_dict(request), ensure_ascii=False) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                stderr = ""
                if proc.stderr is not None:
                    stderr = proc.stderr.read()
                raise RuntimeError(
                    f"Veya expert {self.expert_id!r} exited without a response: {stderr[-2000:]}"
                )
            payload = json.loads(line)
            response = response_from_dict(payload)
            if response.expert_id != self.expert_id:
                raise RuntimeError(
                    f"Veya expert id mismatch: expected {self.expert_id!r}, got {response.expert_id!r}"
                )
            return response

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
