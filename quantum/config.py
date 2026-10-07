"""Explicit external-model configuration.

The default policy is disabled. A hosted endpoint is used only when the
operator sets the policy, the base URL, and a key. The key is never copied
into a log line by this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

POLICY_DISABLED = "disabled"
POLICY_SELF_HOSTED = "self_hosted"
POLICY_HOSTED = "hosted"
POLICIES = (POLICY_DISABLED, POLICY_SELF_HOSTED, POLICY_HOSTED)

ISING_MODEL = "nvidia/ising-calibration-1.5-31b"
SAFETY_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
PROMPT_VERSION = "quantum-ising-v1"
SCHEMA_VERSION = "1"

_CRITIC_IDS = (
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-20b",
)

# image is true only for the designated calibration specialist.
# Other models stay text-only until a separate recorded probe says otherwise,
# and the runtime path does not consult that probe.
REGISTRY: dict[str, dict[str, object]] = {
    ISING_MODEL: {
        "role": "specialist",
        "text": True,
        "image": True,
        "image_verified": True,
        "structured_output": None,
        "context_tokens": 128_000,
        "research_only": False,
        "enabled_by_default": False,
    },
    SAFETY_MODEL: {
        "role": "safety",
        "text": True,
        "image": False,
        "image_verified": False,
        "structured_output": None,
        "context_tokens": None,
        "research_only": False,
        "enabled_by_default": False,
    },
}
for _model_id in _CRITIC_IDS:
    REGISTRY[_model_id] = {
        "role": "critic",
        "text": True,
        "image": False,
        "image_verified": False,
        "structured_output": None,
        "context_tokens": None,
        "research_only": True,
        "enabled_by_default": False,
    }

_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal"}
_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class QuantumConfig:
    """Resolved operator configuration. `api_key` is a secret."""

    enabled: bool
    policy: str
    base_url: str
    api_key: str
    ising_model: str
    critic_models: tuple[str, ...]
    icl_k: int
    cache_enabled: bool
    cache_entries: int
    safety_enabled: bool
    connect_timeout_s: float
    read_timeout_s: float
    max_retries: int
    max_response_bytes: int
    max_tokens: int
    errors: tuple[str, ...]

    def allowed_models(self) -> frozenset[str]:
        models = {self.ising_model}
        models.update(self.critic_models)
        if self.safety_enabled:
            models.add(SAFETY_MODEL)
        return frozenset(models)

    def public(self) -> dict[str, object]:
        """Fields that are safe to log or return to a caller."""
        parts = urlsplit(self.base_url) if self.base_url else None
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "base_url_scheme": parts.scheme if parts else "",
            "base_url_host": parts.hostname if parts else "",
            "ising_model": self.ising_model,
            "critic_models": list(self.critic_models),
            "icl_k": self.icl_k,
            "cache_enabled": self.cache_enabled,
            "safety_enabled": self.safety_enabled,
            "errors": list(self.errors),
        }


def redact(text: str, secret: str) -> str:
    if not secret or secret not in text:
        return text
    return text.replace(secret, "[redacted]")


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float(raw: str | None, default: float, low: float, high: float) -> float | None:
    if raw is None or raw.strip() == "":
        return default
    try:
        number = float(raw)
    except ValueError:
        return None
    if number < low or number > high:
        return None
    return number


def _bounded_int(raw: str | None, default: int, low: int, high: int) -> int | None:
    if raw is None or raw.strip() == "":
        return default
    try:
        number = int(raw)
    except ValueError:
        return None
    if number < low or number > high:
        return None
    return number


def validate_base_url(url: str, policy: str) -> str | None:
    """Return an error code, or None when the operator URL is acceptable."""
    if not url or any(ch in url for ch in ("\n", "\r", "\x00")):
        return "base_url_invalid"
    parts = urlsplit(url)
    if parts.username or parts.password:
        return "base_url_has_credentials"
    if parts.query or parts.fragment:
        return "base_url_invalid"
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or host in _METADATA_HOSTS:
        return "base_url_rejected"
    if policy == POLICY_HOSTED:
        if parts.scheme != "https":
            return "hosted_requires_https"
    elif policy == POLICY_SELF_HOSTED:
        if parts.scheme not in ("https", "http"):
            return "base_url_invalid"
    else:
        return "base_url_invalid"
    return None


def load_config(env: Mapping[str, str] | None = None) -> QuantumConfig:
    """Resolve configuration. Invalid combinations stay disabled."""
    source = os.environ if env is None else env
    errors: list[str] = []
    policy = (source.get("VEYA_EXTERNAL_MODEL_POLICY") or POLICY_DISABLED).strip().lower()
    if policy not in POLICIES:
        errors.append("policy_invalid")
        policy = POLICY_DISABLED
    requested = _flag(source.get("VEYA_QUANTUM_ENABLED"))
    base_url = (source.get("VEYA_NIM_BASE_URL") or "").strip().rstrip("/")
    api_key = source.get("VEYA_NIM_API_KEY") or ""
    if any(ch in api_key for ch in ("\n", "\r", "\x00")):
        errors.append("api_key_invalid")
        api_key = ""
    model = (source.get("VEYA_ISING_MODEL") or ISING_MODEL).strip()
    if model not in REGISTRY or REGISTRY[model].get("role") != "specialist":
        errors.append("ising_model_not_allowed")
        model = ISING_MODEL
    critics: list[str] = []
    raw_critics = source.get("VEYA_QUANTUM_CRITIC_MODELS") or ""
    for item in raw_critics.split(","):
        name = item.strip()
        if not name:
            continue
        spec = REGISTRY.get(name)
        if spec is None or spec.get("role") != "critic":
            errors.append("critic_not_allowed")
            continue
        if name not in critics:
            critics.append(name)
    icl_k = _bounded_int(source.get("VEYA_QUANTUM_ICL_K"), 0, 0, 8)
    if icl_k is None:
        errors.append("icl_k_invalid")
        icl_k = 0
    cache_entries = _bounded_int(source.get("VEYA_QUANTUM_CACHE_ENTRIES"), 32, 1, 256)
    if cache_entries is None:
        errors.append("cache_entries_invalid")
        cache_entries = 32
    connect = _bounded_float(source.get("VEYA_NIM_CONNECT_TIMEOUT"), 5.0, 0.2, 30.0)
    read = _bounded_float(source.get("VEYA_NIM_READ_TIMEOUT"), 60.0, 0.2, 120.0)
    if connect is None or read is None:
        errors.append("timeout_invalid")
        connect = connect or 5.0
        read = read or 60.0
    retries = _bounded_int(source.get("VEYA_NIM_MAX_RETRIES"), 1, 0, 2)
    if retries is None:
        errors.append("retries_invalid")
        retries = 1
    max_tokens = _bounded_int(source.get("VEYA_ISING_MAX_TOKENS"), 2048, 64, 8192)
    if max_tokens is None:
        errors.append("max_tokens_invalid")
        max_tokens = 2048
    if requested and policy == POLICY_DISABLED:
        errors.append("policy_disabled")
    if requested and policy != POLICY_DISABLED:
        if not base_url:
            errors.append("base_url_missing")
        else:
            url_error = validate_base_url(base_url, policy)
            if url_error:
                errors.append(url_error)
        if policy == POLICY_HOSTED and not api_key:
            errors.append("api_key_missing")
    safety = _flag(source.get("VEYA_QUANTUM_SAFETY_GUARD"))
    enabled = requested and policy in (POLICY_HOSTED, POLICY_SELF_HOSTED) and not errors
    if not enabled:
        # A disabled runtime must not keep a usable outbound target.
        if policy == POLICY_DISABLED or not requested or errors:
            pass
    return QuantumConfig(
        enabled=enabled,
        policy=policy if enabled else POLICY_DISABLED,
        base_url=base_url if enabled else "",
        api_key=api_key if enabled else "",
        ising_model=model,
        critic_models=tuple(critics) if enabled else (),
        icl_k=icl_k,
        cache_enabled=_flag(source.get("VEYA_QUANTUM_CACHE")) and enabled,
        cache_entries=cache_entries,
        safety_enabled=safety and enabled,
        connect_timeout_s=connect,
        read_timeout_s=read,
        max_retries=retries,
        max_response_bytes=1_048_576,
        max_tokens=max_tokens,
        errors=tuple(errors),
    )


def capability(model_id: str) -> dict[str, object] | None:
    spec = REGISTRY.get(model_id)
    if spec is None:
        return None
    return {"model": model_id, **spec}


def record_probe(model_id: str, probe: Mapping[str, object]) -> dict[str, object] | None:
    """Research-only capability note. Does not change the runtime registry."""
    spec = capability(model_id)
    if spec is None or not isinstance(probe, Mapping):
        return None
    noted = dict(spec)
    if probe.get("image_input") == "verified" and spec.get("role") == "critic":
        noted["image"] = True
        noted["image_verified"] = True
        noted["probe"] = "recorded"
    else:
        noted["image"] = False if spec.get("role") != "specialist" else spec.get("image")
        noted["probe"] = "recorded"
    if isinstance(probe.get("context_tokens"), int) and not isinstance(probe.get("context_tokens"), bool):
        noted["context_tokens"] = probe["context_tokens"]
    if isinstance(probe.get("structured_output"), bool):
        noted["structured_output"] = probe["structured_output"]
    return noted


def is_slug(value: object) -> bool:
    return isinstance(value, str) and _SLUG.match(value) is not None
