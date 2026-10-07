"""Migration 0002 (projects.rate_tier) is additive and matches the Worker's auth query."""

import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "product" / "migrations"


def _statements(path):
    text = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    return [s.strip() for s in text.split(";") if s.strip()]


def test_rate_tier_migration_is_additive_only():
    statements = _statements(MIGRATIONS / "0002_project_rate_tier.sql")
    assert statements, "empty migration"
    for statement in statements:
        upper = " ".join(statement.upper().split())
        assert upper.startswith("ALTER TABLE PROJECTS ADD COLUMN") or upper.startswith("CREATE "), statement
        for word in ("DROP", "DELETE", "UPDATE", "RENAME", "NOT NULL"):
            assert word not in upper, (word, statement)


def test_existing_projects_keep_the_default_tier_and_the_auth_query_runs():
    db = sqlite3.connect(":memory:")
    db.executescript((MIGRATIONS / "0001_product.sql").read_text(encoding="utf-8"))
    db.execute("INSERT INTO projects (id, organization_id, name, created_at) VALUES ('p1', 'o', 'p', 't')")
    db.execute(
        "INSERT INTO api_keys (id, organization_id, project_id, prefix, key_hash, scopes, created_at) "
        "VALUES ('k1', 'o', 'p1', 'veya_live_x', 'h1', '[]', 't')"
    )
    db.executescript((MIGRATIONS / "0002_project_rate_tier.sql").read_text(encoding="utf-8"))
    assert db.execute("SELECT rate_tier FROM projects WHERE id = 'p1'").fetchone() == (None,)
    source = (ROOT / "worker" / "api_keys.mjs").read_text(encoding="utf-8")
    query = "".join(re.findall(r'"([^"]*)"', source.split("TIERED_KEY_QUERY =", 1)[1].split(";", 1)[0]))
    assert "p.rate_tier" in query
    db.execute("UPDATE projects SET rate_tier = 'site' WHERE id = 'p1'")
    row = db.execute(query, ("h1",)).fetchone()
    assert row[0] == "k1" and row[-1] == "site"
