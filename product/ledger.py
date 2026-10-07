"""Usage records. Raw customer content and credentials are rejected."""

from __future__ import annotations

from datetime import datetime
from typing import Any

FORBIDDEN = frozenset(
    {
        "state",
        "instructions",
        "criteria",
        "prompt",
        "messages",
        "content",
        "authorization",
        "api_key",
        "token",
        "secret",
    }
)
ALLOWED = frozenset(
    {
        "organization_id",
        "project_id",
        "api_key_id",
        "endpoint",
        "primitive",
        "task",
        "selected_model",
        "route",
        "http_status",
        "latency_ms",
        "release",
        "runtime_version",
        "created_at",
    }
)


class LedgerError(ValueError):
    pass


def usage_event(fields: dict[str, Any], *, created_at: datetime) -> dict[str, Any]:
    unknown = set(fields) - ALLOWED
    leaked = unknown & FORBIDDEN
    if leaked or unknown:
        raise LedgerError("usage record contains a field that is not allowed")
    required = ("organization_id", "project_id", "api_key_id", "endpoint", "http_status")
    if any(fields.get(name) in (None, "") for name in required):
        raise LedgerError("usage record is missing an identity field")
    return {**fields, "created_at": created_at.isoformat()}
