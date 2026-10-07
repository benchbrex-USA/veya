"""Customer API keys. The raw key is returned once and is never stored."""

from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime

KEY_RE = re.compile(r"^veya_live_[A-Za-z0-9_-]{43}$")
SCOPES = frozenset({"decisions:invoke", "responses:invoke", "models:read"})


@dataclass(frozen=True)
class KeyRecord:
    id: str
    organization_id: str
    project_id: str
    prefix: str
    key_hash: str
    scopes: frozenset[str]
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None


@dataclass(frozen=True)
class Identity:
    organization_id: str
    project_id: str
    key_id: str
    scopes: frozenset[str]


class AuthError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def generate_api_key(
    organization_id: str,
    project_id: str,
    scopes: set[str] | frozenset[str],
    *,
    created_at: datetime,
    expires_at: datetime | None = None,
) -> tuple[str, KeyRecord]:
    unknown = set(scopes) - SCOPES
    if unknown or not scopes:
        raise AuthError("invalid_scope", "API key scope is not allowed.")
    if expires_at is not None and expires_at <= created_at:
        raise AuthError("invalid_expiry", "API key expiry must be in the future.")
    raw = "veya_live_" + secrets.token_urlsafe(32)
    if not KEY_RE.fullmatch(raw):
        raise AuthError("invalid_key", "Generated API key did not match the key format.")
    record = KeyRecord(
        id=str(uuid.uuid4()),
        organization_id=organization_id,
        project_id=project_id,
        prefix=raw[:16],
        key_hash=hash_key(raw),
        scopes=frozenset(scopes),
        created_at=created_at,
        expires_at=expires_at,
    )
    return raw, record


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_api_key(presented: str, record: KeyRecord, now: datetime) -> Identity:
    """Reject unknown, expired, revoked, and cross-checked keys in constant time."""
    candidate = hash_key(presented) if isinstance(presented, str) else hash_key("")
    matches = secrets.compare_digest(candidate, record.key_hash)
    if record.revoked_at is not None or (record.expires_at is not None and record.expires_at <= now):
        matches = False
    if not matches:
        raise AuthError("unauthorized", "Authentication is required.")
    return Identity(
        organization_id=record.organization_id,
        project_id=record.project_id,
        key_id=record.id,
        scopes=record.scopes,
    )


def require_scope(identity: Identity, scope: str) -> None:
    if scope not in identity.scopes:
        raise AuthError("forbidden", "API key does not allow this operation.")


def require_project(identity: Identity, organization_id: str, project_id: str) -> None:
    if identity.organization_id != organization_id or identity.project_id != project_id:
        raise AuthError("forbidden", "API key does not belong to this project.")
