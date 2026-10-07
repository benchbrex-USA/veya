"""Production auth, contracts, and the disabled-specialist policy."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from product.contracts import ProductError, parse_decision, parse_response
from product.keys import AuthError, generate_api_key, require_project, require_scope, verify_api_key
from product.ledger import LedgerError, usage_event
from product.policy import PRODUCTION_ENABLED, model_catalog, production_route

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def test_api_key_is_hashed_and_isolated():
    raw, record = generate_api_key(
        "org-a",
        "project-a",
        {"decisions:invoke"},
        created_at=NOW,
        expires_at=NOW + timedelta(days=30),
    )
    assert raw not in record.key_hash
    assert record.prefix == raw[:16]
    identity = verify_api_key(raw, record, NOW)
    assert identity.organization_id == "org-a"
    require_scope(identity, "decisions:invoke")
    with pytest.raises(AuthError) as denied:
        require_scope(identity, "responses:invoke")
    assert denied.value.code == "forbidden"
    with pytest.raises(AuthError):
        require_project(identity, "org-b", "project-a")
    with pytest.raises(AuthError):
        verify_api_key(raw[:-1] + ("A" if raw[-1] != "A" else "B"), record, NOW)


def test_expired_and_revoked_keys_fail():
    raw, record = generate_api_key(
        "org-a",
        "project-a",
        {"models:read"},
        created_at=NOW - timedelta(days=2),
        expires_at=NOW - timedelta(seconds=1),
    )
    with pytest.raises(AuthError) as expired:
        verify_api_key(raw, record, NOW)
    assert expired.value.code == "unauthorized"
    fresh, live = generate_api_key(
        "org-a", "project-a", {"models:read"}, created_at=NOW, expires_at=NOW + timedelta(days=1)
    )
    revoked = type(live)(**{**live.__dict__, "revoked_at": NOW})
    with pytest.raises(AuthError):
        verify_api_key(fresh, revoked, NOW)


def test_decisions_stay_separate_from_specialists():
    body = parse_decision(
        {
            "primitive": "choice",
            "state": "Please refund the duplicate March invoice.",
            "instructions": "Which label best fits the text?",
            "criteria": {
                "billing": "Invoice and refund questions",
                "access": "Login and permission questions",
            },
        }
    )
    assert body["endpoint"] == "/v1/decisions"
    assert body["primitive"] == "choice"
    assert body["body"]["criteria"]["billing"].startswith("Invoice")
    with pytest.raises(ProductError) as blocked:
        parse_decision({"primitive": "systemone", "state": "text", "questions": {}})
    assert blocked.value.code == "unsupported_primitive"
    with pytest.raises(ProductError) as wrong:
        parse_response({"task": "bounded_decision"})
    assert wrong.value.code == "wrong_endpoint"
    with pytest.raises(ProductError):
        parse_response({"task": "general_text", "model": "openai/gpt-oss-20b"})


def test_listed_specialists_are_not_production_enabled():
    assert PRODUCTION_ENABLED == frozenset()
    decision = production_route({"task": "general_text"})
    assert decision["selected"] == "defer"
    assert decision["invokes_external"] is False
    catalog = model_catalog()
    enabled = [item["model_id"] for item in catalog if item["production_enabled"]]
    assert enabled == ["veya-core"]
    assert all(item["benchbrex_tuned"] is False for item in catalog)
    unavailable = parse_response({"task": "general_text"})
    assert unavailable["code"] == "specialist_not_enabled"


def test_usage_ledger_rejects_customer_content():
    event = usage_event(
        {
            "organization_id": "org-a",
            "project_id": "project-a",
            "api_key_id": "key-1",
            "endpoint": "/v1/decisions",
            "primitive": "choice",
            "http_status": 200,
            "release": "0.3.5-audit-baseline.1",
        },
        created_at=NOW,
    )
    assert "state" not in event
    with pytest.raises(LedgerError):
        usage_event(
            {
                "organization_id": "org-a",
                "project_id": "project-a",
                "api_key_id": "key-1",
                "endpoint": "/v1/decisions",
                "http_status": 200,
                "state": "customer text",
            },
            created_at=NOW,
        )


def test_schema_and_runtime_manifest_do_not_upgrade_the_model():
    sql_text = (ROOT / "product" / "migrations" / "0001_product.sql").read_text(encoding="utf-8")
    sql = "\n".join(line for line in sql_text.lower().splitlines() if not line.strip().startswith("--"))
    assert "password" not in sql
    assert "raw_key" not in sql
    assert "key_hash" in sql
    manifest = (ROOT / "product" / "runtime_manifest.json").read_text(encoding="utf-8")
    assert "RESEARCH RELEASE" in manifest
    assert "RESEARCH ONLY" in manifest
    assert '"enabled_specialists": []' in manifest
    assert '"replicas": 2' in manifest
    config = (ROOT / "wrangler.production.jsonc").read_text(encoding="utf-8")
    assert '"name": "veya-production"' in config
    assert '"max_instances": 2' in config
    assert "Not deployed" in config
