-- Production product metadata. No raw API key, password, prompt, or state column.
CREATE TABLE IF NOT EXISTS organizations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
  organization_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  prefix TEXT NOT NULL,
  key_hash TEXT NOT NULL,
  scopes TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT,
  last_used_at TEXT,
  revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS usage_ledger (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  api_key_id TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  primitive TEXT,
  task TEXT,
  selected_model TEXT,
  route TEXT,
  http_status INTEGER NOT NULL,
  latency_ms INTEGER,
  release TEXT,
  runtime_version TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL,
  actor_key_id TEXT,
  action TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_configurations (
  id TEXT PRIMARY KEY,
  routing_policy TEXT NOT NULL,
  enabled_models TEXT NOT NULL,
  image_digest TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS api_keys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS usage_ledger_project ON usage_ledger (organization_id, project_id, created_at);
