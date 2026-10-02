"""Same-origin website and bounded Veya model gateway. Run: python3 -m web.server."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from html import escape
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from .contracts import (
    Primitive,
    RequestError,
    UpstreamContractError,
    parse_json,
    validate,
    validate_response,
)

ROOT = Path(__file__).resolve().parent
UPSTREAM = os.environ.get("VEYA_MODEL_URL", "http://127.0.0.1:8080").rstrip("/")
PORT = int(os.environ.get("VEYA_WEB_PORT", "3000"))
WEB_HOST = os.environ.get("VEYA_WEB_HOST", "127.0.0.1")
PUBLIC_ORIGIN = os.environ.get("VEYA_PUBLIC_ORIGIN", "").rstrip("/")
if PUBLIC_ORIGIN:
    parsed_origin = urlsplit(PUBLIC_ORIGIN)
    if (
        parsed_origin.scheme != "https"
        or not parsed_origin.hostname
        or parsed_origin.path
        or parsed_origin.query
        or parsed_origin.fragment
        or parsed_origin.username
        or parsed_origin.password
    ):
        raise ValueError("VEYA_PUBLIC_ORIGIN must be an HTTPS origin without a path.")
if WEB_HOST not in ("127.0.0.1", "::1", "localhost") and not PUBLIC_ORIGIN:
    raise ValueError("Public binding requires VEYA_PUBLIC_ORIGIN.")
VEYA_SHADOW_MODE = os.environ.get("VEYA_SHADOW_MODE", "")
SITE_CONFIG = json.loads((ROOT / "static/site-config.json").read_text())
PAGES = {
    page["href"]: (page["title"], page["description"])
    for page in SITE_CONFIG["pages"]
}


def _page_html(path: str) -> bytes:
    title, description = PAGES[path]
    page = (ROOT / "static/index.html").read_text()
    page = page.replace("__VEYA_TITLE__", escape(title, quote=True))
    page = page.replace("__VEYA_DESCRIPTION__", escape(description, quote=True))
    seo = ""
    if PUBLIC_ORIGIN:
        canonical = escape(PUBLIC_ORIGIN + path, quote=True)
        image = escape(PUBLIC_ORIGIN + "/static/veya-lockup.png", quote=True)
        seo = (
            f'<link rel="canonical" href="{canonical}">'
            f'<meta property="og:url" content="{canonical}">'
        )
        page = page.replace('content="/static/veya-lockup.png"', f'content="{image}"')
    page = page.replace("<!--VEYA_SEO-->", seo)
    return page.encode("utf-8")
MAX_REQUEST_BYTES = 32_768
MAX_RESPONSE_BYTES = 524_288
# Per-request inference ceiling (G12). Configurable so operators can trade
# latency headroom against client-visible timeout behavior without a redeploy.
INFER_TIMEOUT_S = int(os.environ.get("VEYA_REQUEST_TIMEOUT", "30"))
WINDOW_S = 60
CAPACITY = 12
MAX_INFLIGHT = 2
# Graceful shutdown (G12): SIGTERM stops accepting new work, in-flight
# requests get SHUTDOWN_DRAIN_S to finish, then the process exits hard.
SHUTDOWN_DRAIN_S = 30
_SHUTTING_DOWN = False
rate_hits: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
rate_lock = threading.Lock()
inflight = threading.BoundedSemaphore(MAX_INFLIGHT)
version_lock = threading.Lock()
version_cache: tuple[float, dict[str, Any]] = (0, {})
logger = logging.getLogger("veya.web")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Machine-readable error codes (G12). Every >=400 JSON response carries the
# human-readable legacy fields ("code"/"detail") plus a stable uppercase
# "error_code" and an "error" message, so clients can branch on codes that
# never change even if the human-readable copy is reworded.
_ERROR_CODES = {
    "invalid_input": "VALIDATION_ERROR",
    "request_too_large": "VALIDATION_ERROR",
    "model_rejected": "VALIDATION_ERROR",
    "unsupported_language": "UNSUPPORTED_LANGUAGE",
    "model_busy": "OVERLOADED",
    "rate_limited": "RATE_LIMITED",
    "timeout": "TIMEOUT",
    "model_unavailable": "INTERNAL_ERROR",
    "model_error": "INTERNAL_ERROR",
    "invalid_upstream_response": "INTERNAL_ERROR",
    "benchmark_unavailable": "INTERNAL_ERROR",
    "shutting_down": "SHUTTING_DOWN",
    "not_found": "NOT_FOUND",
}


def _graceful_shutdown(signum: int, frame: Any) -> None:
    """SIGTERM/SIGINT handler: drain in-flight work, then exit unconditionally."""
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True
    logging.getLogger("veya.server").info(
        "signal %s received, draining in-flight requests for %ss...",
        signum,
        SHUTDOWN_DRAIN_S,
    )
    threading.Timer(SHUTDOWN_DRAIN_S, lambda: os._exit(0)).start()

# Security audit trail for rejected requests. WARNING-level only, structured
# fields rendered by the dedicated formatter below. Records the client IP as a
# truncated SHA-256 hash plus path and error code/reason — never raw request
# bodies or candidate text. Propagation stays enabled so operators who attach
# root handlers also receive audit events.
_audit_log = logging.getLogger("veya.security_audit")
if not _audit_log.handlers:
    _audit_handler = logging.StreamHandler()
    _audit_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "status=%(status)s client=%(client_hash)s path=%(path)s "
            "code=%(error_code)s reason=%(reason)s"
        )
    )
    _audit_log.addHandler(_audit_handler)
_audit_log.setLevel(logging.WARNING)


def _hash_ip(ip: str) -> str:
    """Truncated SHA-256 of a client IP: correlatable, not identifying."""
    return hashlib.sha256(ip.encode("utf-8", "replace")).hexdigest()[:16]


def _audit_text(value: Any, limit: int = 120) -> str:
    """One capped line of printable text; keeps control chars out of the log."""
    text = value if isinstance(value, str) else str(value)
    return "".join(ch if ch.isprintable() else "_" for ch in text)[:limit]


def _upstream_request(
    path: str, body: bytes | None = None, timeout: float = 3
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    token = os.environ.get("VEYA_MODEL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        UPSTREAM + path,
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise UpstreamContractError("Model response exceeds limit.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise UpstreamContractError("Model response must be an object.")
    return payload


def _version() -> dict[str, Any]:
    global version_cache
    now = time.monotonic()
    with version_lock:
        if now - version_cache[0] < 60 and version_cache[1]:
            return version_cache[1]
    source = _upstream_request("/version", timeout=2)
    if (
        not isinstance(source.get("model_version"), str)
        or len(source["model_version"]) > 100
    ):
        raise UpstreamContractError("Missing model version.")
    payload: dict[str, Any] = {}
    for key in (
        "model_version",
        "release",
        "architecture_version",
        "checkpoint",
        "calibration_version",
        "reranker_id",
        "hypothesis_scorer_id",
        "release_manifest_sha256",
    ):
        value = source.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 150:
                raise UpstreamContractError("Invalid version metadata.")
            payload[key] = value
    for key in ("supported_languages", "supported_primitives"):
        value = source.get(key)
        if value is not None:
            if (
                not isinstance(value, list)
                or len(value) > 50
                or any(not isinstance(item, str) or len(item) > 80 for item in value)
            ):
                raise UpstreamContractError("Invalid version capability metadata.")
            payload[key] = value
    status = source.get("release_status")
    if status is not None:
        if (
            not isinstance(status, dict)
            or len(status) > 30
            or any(
                not isinstance(k, str)
                or not isinstance(v, str)
                or len(k) > 80
                or len(v) > 80
                for k, v in status.items()
            )
        ):
            raise UpstreamContractError("Invalid release status metadata.")
        payload["release_status"] = status
    with version_lock:
        version_cache = (now, payload)
    return payload


def _weight(primitive: str, body: dict[str, Any]) -> int:
    if primitive == "choice":
        count = len(body["criteria"])
    elif primitive == "systemone":
        count = sum(
            len(q["criteria"])
            for q in body["questions"].values()
            if q["type"] == "choice"
        )
    else:
        count = 0
    return min(CAPACITY, 1 + math.ceil(max(0, count - 19) / 20))


def _charge(key: str, cost: int) -> int:
    """Return seconds to retry, or zero if accepted. IP comes from the socket only."""
    now = time.monotonic()
    with rate_lock:
        if len(rate_hits) > 4096:
            for client, events in list(rate_hits.items()):
                if not events or now - events[-1][0] >= WINDOW_S:
                    del rate_hits[client]
        events = rate_hits[key]
        while events and now - events[0][0] >= WINDOW_S:
            events.popleft()
        if sum(weight for _, weight in events) + cost > CAPACITY:
            return (
                max(1, math.ceil(WINDOW_S - (now - events[0][0])))
                if events
                else WINDOW_S
            )
        events.append((now, cost))
        return 0


def _shadow_enabled() -> bool:
    """Shadow mode is opt-in via VEYA_SHADOW_MODE=1|true."""
    return str(VEYA_SHADOW_MODE).strip().lower() in ("1", "true")


def _unit_probability(value: Any) -> bool:
    """Finite probability in [0, 1]; booleans never count as numbers."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _telemetry_max_probability(
    primitive: str | None, upstream: dict[str, Any]
) -> float | None:
    """Peak model probability for the telemetry log only — never for clients.

    validate_response (web/contracts.py) whitelists the client-facing payload
    and deliberately drops max_probability, while shadow evaluation and drift
    monitoring (research/veyabench/drift_monitor.py confidence distributions)
    need the score. The gateway therefore reads the raw model-service payload
    before the whitelist is applied: the audited bundle serves the field
    directly (veya/release_bundle.py public_choice); plain agent responses
    omit it, so it is derived from the served distribution the same way the
    bundle computes it. Multi-question systemone requests have no single
    top-level distribution, so the field stays null there, as it does on
    defer and error paths.
    """
    value = upstream.get("max_probability")
    if _unit_probability(value):
        return float(value)
    probabilities = upstream.get("probabilities")
    if isinstance(probabilities, dict) and probabilities and all(
        _unit_probability(p) for p in probabilities.values()
    ):
        return max(float(p) for p in probabilities.values())
    if primitive == "noul":
        answer = upstream.get("answer")
        if _unit_probability(answer):
            # Noul answers carry P(yes); the implied binary distribution's
            # peak is max(p, 1 - p) — the agent's own confidence semantics.
            return max(float(answer), 1.0 - float(answer))
    return None


def _safe_telemetry(
    request_id: str,
    primitive: str | None,
    candidate_count: int,
    route: str | None,
    deferred: bool | None,
    status: int,
    started: float,
    model_version: str | None = None,
    error_code: str | None = None,
    timestamp: str | None = None,
    confidence: float | None = None,
    max_probability: float | None = None,
    automation: str | None = None,
    config_hash: str | None = None,
) -> None:
    entry: dict[str, Any] = {
        "request_id": request_id,
        "primitive": primitive,
        "candidate_count": candidate_count,
        "route": route,
        "deferred": deferred,
        "status": status,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "model_version": model_version,
        "error_code": error_code,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
        "max_probability": max_probability,
        "automation": automation,
        "config_hash": config_hash,
    }
    if _shadow_enabled():
        entry["shadow"] = True
    logger.info(json.dumps(entry, separators=(",", ":")))


# Operational probes (G18). The gateway never loads weights itself; each
# component is derived from the audited model service, which only starts
# serving after load_release_manifest verified every release-manifest hash.
# Component state latches once observed. A healthy service that cannot
# present the audited release bindings is an integrity failure: startup_failed
# is sticky and the gateway never reports ready again for the life of the
# process (fail-closed).
_READINESS = {
    "model_loaded": False,
    "policy_loaded": False,
    "calibration_loaded": False,
    "shortlist_loaded": False,
    "hashes_valid": False,
    "startup_failed": False,
}
_READINESS_LOCK = threading.Lock()
_READINESS_PROBE = {"at": 0.0}
READINESS_TTL_S = 2.0
_RELEASE_BINDINGS = {"model_hash": "", "policy_hash": ""}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _digest(value: Any) -> str | None:
    """Return the value when it is a SHA-256 hex digest, else None."""
    if not isinstance(value, str) or len(value) != 64:
        return None
    return value if all(char in _HEX_DIGITS for char in value) else None


def _readiness_snapshot() -> dict[str, bool]:
    with _READINESS_LOCK:
        return dict(_READINESS)


def _is_ready(snapshot: dict[str, bool]) -> bool:
    return (
        snapshot["model_loaded"]
        and snapshot["policy_loaded"]
        and snapshot["calibration_loaded"]
        and snapshot["shortlist_loaded"]
        and snapshot["hashes_valid"]
        and not snapshot["startup_failed"]
    )


def _get_model_hash() -> str:
    with _READINESS_LOCK:
        return _RELEASE_BINDINGS["model_hash"] or "unknown"


def _get_policy_hash() -> str:
    with _READINESS_LOCK:
        return _RELEASE_BINDINGS["policy_hash"] or "unknown"


def _refresh_readiness() -> None:
    """Latch component readiness from the audited model service.

    /health proves the serving process is up; /version exposes the release
    bindings that exist only after every manifest hash was verified. Probing
    stops once every component is latched or startup has failed, so the
    settled state is deterministic and repeated probes cannot flap it.
    """
    now = time.monotonic()
    with _READINESS_LOCK:
        if _READINESS["startup_failed"]:
            return
        if all(_READINESS.values()):
            return
        if now - _READINESS_PROBE["at"] < READINESS_TTL_S:
            return
        _READINESS_PROBE["at"] = now
    try:
        health = _upstream_request("/health", timeout=2)
        if health.get("status") != "ok":
            return
    except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
        return
    with _READINESS_LOCK:
        _READINESS["model_loaded"] = True
    try:
        version = _upstream_request("/version", timeout=2)
    except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
        return
    manifest_hash = _digest(version.get("release_manifest_sha256"))
    checkpoint_hash = _digest(version.get("checkpoint_sha256"))
    if manifest_hash is None or checkpoint_hash is None:
        # The service answers but cannot present the audited release bindings.
        with _READINESS_LOCK:
            _READINESS["startup_failed"] = True
        return
    shortlist_hash = _digest(version.get("shortlist_sha256"))
    policy_hash = _digest(version.get("selective_policy_sha256"))
    calibration = version.get("route_calibration_temperature")
    calibration_ok = isinstance(calibration, (int, float)) and not isinstance(
        calibration, bool
    )
    with _READINESS_LOCK:
        _READINESS["shortlist_loaded"] = shortlist_hash is not None
        _READINESS["policy_loaded"] = policy_hash is not None
        _READINESS["calibration_loaded"] = calibration_ok
        _READINESS["hashes_valid"] = True
        _RELEASE_BINDINGS["model_hash"] = checkpoint_hash
        _RELEASE_BINDINGS["policy_hash"] = policy_hash or ""


def _probe_send(
    handler: Handler,
    code: int,
    payload: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
) -> None:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    for name, value in (extra_headers or {}).items():
        handler.send_header(name, value)
    handler.headers_common()
    handler.end_headers()
    handler.wfile.write(raw)


def handle_healthz(handler: Handler) -> None:
    """Liveness: the process is answering; the model service is not consulted."""
    _probe_send(handler, 200, {"status": "alive"})


def handle_readyz(handler: Handler) -> None:
    """Readiness: 200 only once the audited release is confirmed loaded."""
    _refresh_readiness()
    snapshot = _readiness_snapshot()
    ready = _is_ready(snapshot)
    headers: dict[str, str] | None = None
    if ready:
        headers = {
            "X-Veya-Model-Hash": _get_model_hash(),
            "X-Veya-Policy-Hash": _get_policy_hash(),
        }
    _probe_send(
        handler,
        200 if ready else 503,
        {"status": "ready" if ready else "not_ready", "components": snapshot},
        headers,
    )


def handle_startupz(handler: Handler) -> None:
    """Startup: same gate as readiness; Kubernetes sets longer thresholds."""
    handle_readyz(handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:
        pass

    def _audit_reject(self, status: int, error_code: str, reason: Any = None) -> None:
        """Emit one WARNING-level security audit record for a rejected request.

        Logs the hashed client IP, request path, and error code/reason only;
        raw request bodies and candidate text never reach the audit trail.
        """
        try:
            client_ip = self.client_address[0]
        except (AttributeError, IndexError, TypeError):
            client_ip = "unknown"
        _audit_log.warning(
            "rejected request",
            extra={
                "client_hash": _hash_ip(str(client_ip)),
                "path": _audit_text(self.path.split("?")[0], 300),
                "status": int(status),
                "error_code": _audit_text(error_code, 80),
                "reason": _audit_text(reason if reason is not None else error_code),
            },
        )

    def headers_common(self, cache_control: str = "no-store") -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        # Public deployments are HTTPS-only (VEYA_PUBLIC_ORIGIN is validated as
        # an HTTPS origin); RFC 6797 user agents ignore the header over plain
        # HTTP, so sending it unconditionally is safe for the local preview.
        self.send_header(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        self.send_header(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", cache_control)

    def send(
        self,
        code: int,
        payload: dict[str, Any],
        retry_after: int | None = None,
        shadow: bool = False,
    ) -> None:
        legacy_code = "error"
        if code >= 400:
            legacy_code = str(payload.get("code", "error"))
            self._audit_reject(code, legacy_code, payload.get("detail"))
            # Structured error envelope (G12): the legacy "code"/"detail" fields
            # stay for existing clients; "error_code" is the stable machine
            # contract and "error" the human-readable message.
            payload = {
                **payload,
                "error": str(payload.get("detail", legacy_code)),
                "error_code": _ERROR_CODES.get(legacy_code, "INTERNAL_ERROR"),
            }
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        if code >= 400 and isinstance(payload.get("request_id"), str):
            self.send_header("X-Request-Id", payload["request_id"])
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        if shadow:
            self.send_header("X-Veya-Shadow", "true")
        self.headers_common()
        self.end_headers()
        self.wfile.write(raw)

    def _take_body(self) -> tuple[bytes | None, int]:
        """Read the request body once, up front, keeping keep-alive in sync.

        Error paths that answer before parsing the body must either consume it
        or refuse connection reuse: unread bytes otherwise desynchronize the
        next request on the same socket, and proxies that pool keep-alive
        origin connections (cloudflared included) then parse the next request
        line as e.g. '{}POST'. Oversized or unparsable bodies mark the
        connection for closure instead of draining forever; the body is
        returned as None in that case.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self.close_connection = True
            return None, -1
        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            return None, length
        if length <= 0:
            return b"", length
        return self.rfile.read(length), length

    def _shutdown_reject(self) -> bool:
        """503 + Retry-After while draining; new work must go elsewhere."""
        if not _SHUTTING_DOWN:
            return False
        self.send(
            503,
            {
                "code": "shutting_down",
                "detail": "Server is shutting down. Retry later.",
                "request_id": secrets.token_hex(8),
            },
            SHUTDOWN_DRAIN_S,
        )
        return True

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/healthz":  # liveness answers even while draining
            handle_healthz(self)
            return
        if path == "/readyz":
            handle_readyz(self)
            return
        if path == "/startupz":
            handle_startupz(self)
            return
        if self._shutdown_reject():
            return
        if path in ("/robots.txt", "/sitemap.xml"):
            if path == "/robots.txt":
                body = (
                    "User-agent: *\nAllow: /\nSitemap: "
                    + PUBLIC_ORIGIN
                    + "/sitemap.xml\n"
                    if PUBLIC_ORIGIN
                    else "User-agent: *\nDisallow: /\n"
                )
                content_type = "text/plain; charset=utf-8"
            elif PUBLIC_ORIGIN:
                body = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    + "".join(
                        f"<url><loc>{escape(PUBLIC_ORIGIN + route)}</loc></url>"
                        for route in PAGES
                    )
                    + "</urlset>"
                )
                content_type = "application/xml; charset=utf-8"
            else:
                self.send(404, {"code": "not_found"})
                return
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.headers_common()
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/benchmark":
            try:
                audit = json.loads(
                    (
                        ROOT.parent / "artifacts/release/blind_clinc150_val_result.json"
                    ).read_text()
                )
                choice = audit["choice"]
                self.send(
                    200,
                    {
                        "dataset": audit["dataset"],
                        "n": audit["n"],
                        "candidate_count": audit["candidate_count"],
                        "release": audit["release"],
                        "split_status": audit["status"],
                        "architecture": audit["architecture"],
                        "recall_at_8": choice["retrieval_recall_at_8"]["rate"],
                        "conditional_accuracy": choice["conditional_accuracy_given_hit"]["rate"],
                        "accuracy": choice["end_to_end_accuracy"]["rate"],
                        "ece": choice["ece_end_to_end"],
                        "retrieval_misses": choice["retrieval_misses"],
                        "shortlist_hits": choice["retrieval_recall_at_8"]["count"],
                        "correct_on_hit": choice["conditional_accuracy_given_hit"]["count"],
                        "scoring_misses": choice["reranker_misses_with_target_present"],
                    },
                )
            except (OSError, KeyError, ValueError):
                self.send(503, {"code": "benchmark_unavailable"})
            return
        if path in ("/api/health", "/api/version"):
            try:
                payload = (
                    _upstream_request("/health", timeout=2)
                    if path.endswith("health")
                    else _version()
                )
                if path.endswith("health") and payload.get("status") != "ok":
                    raise UpstreamContractError("Model is unhealthy.")
                self.send(200, payload)
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                self.send(503, {"status": "unavailable", "code": "model_unavailable"})
            return
        if path in PAGES:
            name = "index.html"
        elif path.startswith("/static/"):
            name = path.removeprefix("/static/")
            static_root = (ROOT / "static").resolve()
            candidate = (static_root / name).resolve()
            if (
                not candidate.is_relative_to(static_root)
                or candidate.suffix not in {".css", ".js", ".mjs", ".json", ".png", ".svg"}
                or not candidate.is_file()
            ):
                name = None
        else:
            name = None
        if not name:
            if path.startswith(("/api/", "/static/")):
                self.send(404, {"code": "not_found"})
                return
            self._audit_reject(404, "not_found")
            raw = (
                '<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<meta name="robots" content="noindex">'
                '<title>Page not found — Veya</title>'
                '<link rel="icon" href="/static/veya-symbol.png">'
                '<link rel="stylesheet" href="/static/site.css"></head>'
                '<body><main id="main"><section class="page-hero">'
                '<span class="eyebrow">404 / Veya</span><h1>Page not found.</h1>'
                '<p>This address is not part of the Veya site.</p>'
                '<a class="button" href="/">Return home ↗</a>'
                '</section></main></body></html>'
            ).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.headers_common()
            self.end_headers()
            self.wfile.write(raw)
            return
        file = ROOT / "static" / name
        raw = _page_html(path) if path in PAGES else file.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            {
                "html": "text/html; charset=utf-8",
                "css": "text/css; charset=utf-8",
                "js": "text/javascript; charset=utf-8",
                "mjs": "text/javascript; charset=utf-8",
                "json": "application/json; charset=utf-8",
                "svg": "image/svg+xml",
                "png": "image/png",
            }[name.rsplit(".", 1)[-1]],
        )
        self.send_header("Content-Length", str(len(raw)))
        # Modules and route data are deployed together. Avoid a mixed-version
        # module graph after an incremental release; cache only image assets.
        static_image = path.startswith("/static/") and name.rsplit(".", 1)[-1] in {"png", "svg"}
        self.headers_common("public, max-age=300" if static_image else "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        raw_body, length = self._take_body()
        if self.path != "/api/infer":
            self.send(404, {"code": "not_found"})
            return
        if self._shutdown_reject():
            return
        started = time.monotonic()
        request_id = secrets.token_hex(8)
        primitive: str | None = None
        candidate_count = 0
        route: str | None = None
        deferred: bool | None = None
        model_version: str | None = None
        confidence: float | None = None
        max_probability: float | None = None
        automation: str | None = None
        config_hash: str | None = None
        status = 500
        error_code: str | None = None
        try:
            if length < 0:
                raise RequestError("Invalid Content-Length.")
            if length <= 0:
                raise RequestError("Request body is required.")
            if raw_body is None:
                status, error_code = 413, "request_too_large"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Request exceeds 32 KB.",
                        "request_id": request_id,
                    },
                )
                return
            if (
                self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                != "application/json"
            ):
                raise RequestError("Content-Type must be application/json.")
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            allowed_origins = {f"http://{host}", f"https://{host}"} | {
                item.strip()
                for item in os.environ.get("VEYA_ALLOWED_ORIGINS", "").split(",")
                if item.strip()
            }
            if origin and origin not in allowed_origins:
                raise RequestError("Cross-origin model requests are not allowed.")
            retry_after = _charge(self.client_address[0], 1)
            if retry_after:
                status, error_code = 429, "rate_limited"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Too many model requests. Try again shortly.",
                        "retry_after": retry_after,
                        "request_id": request_id,
                    },
                    retry_after,
                )
                return
            data = parse_json(raw_body)
            body = validate(data)
            primitive = data["primitive"]
            candidate_count = (
                len(body.get("criteria", {}))
                if primitive == "choice"
                else sum(
                    len(q.get("criteria", {}))
                    for q in body["questions"].values()
                    if q["type"] == "choice"
                )
                if primitive == "systemone"
                else 0
            )
            cost = _weight(primitive, body)
            retry_after = _charge(self.client_address[0], cost - 1) if cost > 1 else 0
            if retry_after:
                status, error_code = 429, "rate_limited"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Too many model requests. Try again shortly.",
                        "retry_after": retry_after,
                        "request_id": request_id,
                    },
                    retry_after,
                )
                return
            if not inflight.acquire(blocking=False):
                # Overload (G12): refuse fast instead of queueing; the
                # Retry-After header and body field keep clients well-behaved.
                status, error_code = 503, "model_busy"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Model is busy. Try again shortly.",
                        "retry_after": 5,
                        "request_id": request_id,
                    },
                    5,
                )
                return
            try:
                raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
                upstream = _upstream_request(
                    "/v1/" + primitive, raw, timeout=INFER_TIMEOUT_S
                )
                result = validate_response(
                    cast(Primitive, primitive), upstream, body
                )
                if (
                    result["deferred"]
                    and result.get("defer_reason") == "unsupported_language"
                ):
                    # The model language gate rejected the content even though
                    # the HTTP exchange itself succeeded; audit it as a rejection.
                    self._audit_reject(
                        200, "unsupported_language", "model deferred: unsupported language"
                    )
            finally:
                inflight.release()
            try:
                version = _version()
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                version = {}
            for key in (
                "model_version",
                "release",
                "architecture_version",
                "supported_languages",
                "release_status",
            ):
                if key in version:
                    result[key] = version[key]
            # Public product name; the upstream's internal checkpoint identifier
            # remains an implementation detail and is not a user-facing model name.
            result["model"] = "Veya Core"
            result["request_id"] = request_id
            result["gateway_latency_ms"] = round((time.monotonic() - started) * 1000)
            route = result.get("route")
            deferred = result["deferred"]
            model_version = result.get("model_version")
            confidence = result.get("confidence")
            # Telemetry-only capture (G16): the response whitelist strips the
            # model's peak probability from the client payload, so it is read
            # from the raw upstream response instead. Clients never see the
            # score; drift_monitor.py gets the confidence distribution it needs.
            max_probability = _telemetry_max_probability(primitive, upstream)
            automation = result.get("automation")
            config_hash = version.get("release_manifest_sha256")
            status = 200
            self.send(status, result, shadow=_shadow_enabled())
        except RequestError as exc:
            status, error_code = 400, "invalid_input"
            self.send(
                status,
                {"code": error_code, "detail": str(exc), "request_id": request_id},
            )
        except HTTPError as exc:
            if exc.code in (400, 413, 422):
                try:
                    upstream_error = json.loads(exc.read(16_385))
                except (ValueError, UnicodeError):
                    upstream_error = {}
                reason = (
                    upstream_error.get("defer_reason")
                    if isinstance(upstream_error, dict)
                    else None
                )
                status, error_code = 400, "model_rejected"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "defer_reason": reason
                        if isinstance(reason, str) and len(reason) <= 100
                        else "invalid_schema",
                        "detail": "The model rejected this input.",
                        "request_id": request_id,
                    },
                )
            elif exc.code == 429:
                status, error_code = 429, "rate_limited"
                try:
                    retry_after = max(
                        1, min(3600, int(exc.headers.get("Retry-After", "30")))
                    )
                except ValueError:
                    retry_after = 30
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "Too many model requests. Try again shortly.",
                        "retry_after": retry_after,
                        "request_id": request_id,
                    },
                    retry_after,
                )
            else:
                status, error_code = 502, "model_error"
                self.send(
                    status,
                    {
                        "code": error_code,
                        "detail": "The model service failed.",
                        "request_id": request_id,
                    },
                )
        except (socket.timeout, TimeoutError):
            status, error_code = 504, "timeout"
            self.send(
                status,
                {
                    "code": error_code,
                    "detail": "The model took too long. Please retry.",
                    "request_id": request_id,
                },
            )
        except URLError as exc:
            is_timeout = isinstance(exc.reason, (socket.timeout, TimeoutError))
            status, error_code = (
                (504, "timeout") if is_timeout else (503, "model_unavailable")
            )
            self.send(
                status,
                {
                    "code": error_code,
                    "detail": "The model took too long. Please retry."
                    if is_timeout
                    else "Model service unavailable.",
                    "request_id": request_id,
                },
            )
        except (UpstreamContractError, json.JSONDecodeError, UnicodeError):
            status, error_code = 502, "invalid_upstream_response"
            self.send(
                status,
                {
                    "code": error_code,
                    "detail": "The model returned an invalid response.",
                    "request_id": request_id,
                },
            )
        finally:
            _safe_telemetry(
                request_id,
                primitive,
                candidate_count,
                route,
                deferred,
                status,
                started,
                model_version,
                error_code,
                confidence=confidence,
                max_probability=max_probability,
                automation=automation,
                config_hash=config_hash,
            )


class WebsiteHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    print(f"Veya website: http://{WEB_HOST}:{PORT} | model: {UPSTREAM}", flush=True)
    WebsiteHTTPServer((WEB_HOST, PORT), Handler).serve_forever()
