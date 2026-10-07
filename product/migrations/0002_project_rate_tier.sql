-- Additive only: per-project rate-limit tier for veya-production.
-- NULL (every existing project) keeps the default tier: per API key,
-- 20 inference / 6 heavy requests per minute. 'site' selects the per-project
-- VEYA_RATE_SITE / VEYA_RATE_SITE_HEAVY bindings (worker/rate_tiers.mjs).
-- Not idempotent (SQLite has no ADD COLUMN IF NOT EXISTS): apply once.
ALTER TABLE projects ADD COLUMN rate_tier TEXT;
