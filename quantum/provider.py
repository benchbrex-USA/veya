"""OpenAI-compatible NIM client.

The caller cannot choose the URL or the model. Both come from the operator
config and the allowlist. Exceptions carry a short code and no response body.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import time
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from typing import Callable
from urllib.parse import urlsplit

from quantum.config import REGISTRY, QuantumConfig

Sleep = Callable[[float], None]


class ProviderError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    request_id: str | None
    external_model_score: dict[str, object] | None
    latency_s: float


class ExternalModelProvider:
    """Small surface used by the quantum pipeline. Tests supply fakes."""

    def analyze_text(self, model: str, text: str, *, max_tokens: int = 1024) -> ProviderResult:
        raise ProviderError("unavailable")

    def analyze_multimodal(
        self,
        model: str,
        text: str,
        images: list[tuple[str, str]],
        *,
        max_tokens: int = 1024,
    ) -> ProviderResult:
        raise ProviderError("unavailable")

    def health(self) -> dict[str, object]:
        raise ProviderError("unavailable")

    def capabilities(self, model: str) -> dict[str, object]:
        spec = REGISTRY.get(model)
        if spec is None:
            raise ProviderError("model_not_allowed")
        return {"model": model, **spec}


class NimProvider(ExternalModelProvider):
    def __init__(self, config: QuantumConfig, sleep: Sleep | None = None):
        self.config = config
        self._sleep = sleep or time.sleep

    def capabilities(self, model: str) -> dict[str, object]:
        if model not in REGISTRY:
            raise ProviderError("model_not_allowed")
        spec = dict(REGISTRY[model])
        spec["model"] = model
        spec["configured"] = model in self.config.allowed_models() and self.config.enabled
        return spec

    def health(self) -> dict[str, object]:
        self._require_enabled()
        status, payload = self._request("GET", self._path("models"), None)
        if status != 200 or not isinstance(payload, dict):
            raise ProviderError("unavailable")
        return {"status": "ok", "policy": self.config.policy}

    def analyze_text(self, model: str, text: str, *, max_tokens: int = 1024) -> ProviderResult:
        self._require_model(model, image=False)
        return self._chat(model, [{"role": "user", "content": text}], max_tokens)

    def analyze_multimodal(
        self,
        model: str,
        text: str,
        images: list[tuple[str, str]],
        *,
        max_tokens: int = 1024,
    ) -> ProviderResult:
        self._require_model(model, image=True)
        parts: list[dict[str, object]] = []
        for mime, encoded in images:
            parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, encoded)}})
        parts.append({"type": "text", "text": text})
        return self._chat(model, [{"role": "user", "content": parts}], max_tokens)

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise ProviderError("disabled")

    def _require_model(self, model: str, *, image: bool) -> None:
        self._require_enabled()
        spec = REGISTRY.get(model)
        if spec is None or model not in self.config.allowed_models():
            raise ProviderError("model_not_allowed")
        if image and not spec.get("image"):
            raise ProviderError("image_not_supported")

    def _chat(self, model: str, messages: list[dict[str, object]], max_tokens: int) -> ProviderResult:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": min(max_tokens, self.config.max_tokens),
            "stream": False,
        }
        started = time.monotonic()
        status, payload = self._request("POST", self._path("chat/completions"), body)
        if status != 200 or not isinstance(payload, dict):
            raise ProviderError("invalid_response")
        text, request_id, score = _completion(payload)
        return ProviderResult(
            text=text,
            model=model,
            request_id=request_id,
            external_model_score=score,
            latency_s=time.monotonic() - started,
        )

    def _path(self, leaf: str) -> str:
        parts = urlsplit(self.config.base_url)
        base = parts.path.rstrip("/")
        if base.endswith("/v1"):
            path = base + "/" + leaf
        else:
            path = (base + "/v1/" + leaf) if base else ("/v1/" + leaf)
        return path

    def _request(self, method: str, path: str, body: dict[str, object] | None) -> tuple[int, object]:
        attempts = self.config.max_retries + 1
        last = ProviderError("unavailable")
        for attempt in range(attempts):
            try:
                status, payload = self._once(method, path, body)
            except ProviderError as exc:
                last = exc
                if exc.code in {"rate_limited", "overloaded"} and attempt + 1 < attempts:
                    self._sleep(min(1.0, 0.2 * (attempt + 1)))
                    continue
                raise
            return status, payload
        raise last

    def _once(self, method: str, path: str, body: dict[str, object] | None) -> tuple[int, object]:
        parts = urlsplit(self.config.base_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        timeout = self.config.read_timeout_s
        conn: HTTPConnection
        if parts.scheme == "https":
            conn = HTTPSConnection(host, port, timeout=timeout, context=_tls_context())
        else:
            conn = HTTPConnection(host, port, timeout=timeout)
        raw = b""
        if body is not None:
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Veya-quantum/0.3.5",
        }
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        try:
            conn.request(method, path, body=raw if body is not None else None, headers=headers)
            response = conn.getresponse()
            status = response.status
            if status == 429:
                response.read(256)
                raise ProviderError("rate_limited")
            if status == 503:
                response.read(256)
                raise ProviderError("overloaded")
            if status >= 400:
                response.read(256)
                raise ProviderError("upstream_error")
            declared = response.getheader("Content-Length")
            if declared and declared.isdigit() and int(declared) > self.config.max_response_bytes:
                raise ProviderError("response_too_large")
            data = response.read(self.config.max_response_bytes + 1)
        except ProviderError:
            raise
        except (socket.timeout, TimeoutError):
            raise ProviderError("timeout") from None
        except (OSError, HTTPException, UnicodeError):
            raise ProviderError("unavailable") from None
        finally:
            conn.close()
        if len(data) > self.config.max_response_bytes:
            raise ProviderError("response_too_large")
        try:
            return status, json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ProviderError("invalid_response") from None


def _tls_context() -> ssl.SSLContext:
    """Verify hosted certificates. Use a known CA file when the interpreter has none."""
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and os.path.isfile(configured):
        context.load_verify_locations(cafile=configured)
        return context
    default = ssl.get_default_verify_paths().openssl_cafile
    if default and os.path.isfile(default):
        return context
    for candidate in ("/etc/ssl/cert.pem", "/private/etc/ssl/cert.pem"):
        if os.path.isfile(candidate):
            context.load_verify_locations(cafile=candidate)
            return context
    try:
        import certifi
    except ImportError:
        return context
    bundle = certifi.where()
    if os.path.isfile(bundle):
        context.load_verify_locations(cafile=bundle)
    return context


def _completion(payload: dict[str, object]) -> tuple[str, str | None, dict[str, object] | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderError("invalid_response")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProviderError("invalid_response")
    content = message.get("content")
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
        content = "\n".join(texts)
    if not isinstance(content, str) or len(content) > 200_000:
        raise ProviderError("invalid_response")
    request_id = payload.get("id")
    if not isinstance(request_id, str) or len(request_id) > 80:
        request_id = None
    score = _external_score(choices[0].get("logprobs"))
    return content, request_id, score


def _external_score(value: object) -> dict[str, object] | None:
    """Keep a bounded logprob summary. This is not a Veya probability."""
    if not isinstance(value, dict):
        return None
    try:
        raw = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None
    if len(raw) > 4000:
        return {"present": True, "retained": False}
    return {"present": True, "retained": True, "logprobs": value}
