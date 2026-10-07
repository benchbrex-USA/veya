# Veya production runbook

This runbook covers the Cloudflare production stack:

- **Worker:** `veya-production`, entrypoint `worker/veya-production.mjs`, logic in `worker/production_edge.mjs`, config `wrangler.production.jsonc`.
- **Containers:** application `veya-production-runtime`, 2 × `standard-2` instances, Durable Object class `VeyaContainer`.
- **D1:** `veya-production` (`3fadb994-4107-4821-8cf3-c494073a41e0`), holding the API keys and `usage_ledger`.
- **R2:** `veya-model-artifacts`. The checkpoint lives at `releases/<VEYA_RELEASE_ID>/model.safetensors`.
- **Analytics Engine:** binding `VEYA_USAGE`, dataset `veya_production_usage`.
- **Rate limits:** `VEYA_RATE_INFERENCE` (20/min per key) and `VEYA_RATE_HEAVY` (6/min per key for non-`choice` primitives).
- **Admin secret:** `VEYA_EDGE_TOKEN`. It is a Worker secret, so it is never in git and never in logs.

The account is `8d0a6a64de03f4c9b14184398e75aa6e` and its workers.dev subdomain is `veya-8d0`. Every command below assumes `npx wrangler` is run from the repo root by someone authenticated to that account. **Always pass `--config wrangler.production.jsonc`.**

---

## 1. Config separation and deploy triggers

| Config | Worker | Deployed by |
|---|---|---|
| `wrangler.jsonc` (root) | `veya`: plain edge proxy (`worker/veya-edge.mjs`, `VEYA_ORIGIN`) | Workers Builds on `main` (`npx wrangler deploy`). Non-`main` branches get `npx wrangler preview` (previews enabled) |
| `wrangler.staging.jsonc` | `veya-staging`: container stack (`worker/veya-container.mjs`, container `veya-runtime`, rate limit ns 1001/1002) | Workers Builds on `deploy/cloudflare-staging` |
| `wrangler.production.jsonc` | `veya-production`: container stack (`worker/veya-production.mjs`, container `veya-production-runtime`, rate limit ns 2001/2002) | Workers Builds on `production/veya-dynamics` (`npx wrangler deploy --config wrangler.production.jsonc`), or `.github/workflows/deploy-production.yml` |

`tests/test_wrangler_configs.py` locks this mapping. It checks names, entrypoints, distinct container apps and rate-limit namespaces, that there are no placeholders, and that release pins agree everywhere.

> **Required before merging the config split:** change the `veya-staging` Workers Builds deploy command from `npx wrangler deploy` to `npx wrangler deploy --config wrangler.staging.jsonc`. Without that change, the next staging build deploys the plain proxy config under the wrong name. The build command (`python3 scripts/write_deploy_meta.py && npm ci && …`) stays the same; `write_deploy_meta.py` now checks the pins in both container configs.

Workers Builds pushes to `main`, `production/veya-dynamics` and `deploy/cloudflare-staging` **deploy automatically**. Only push those branches when you mean to deploy.

## 2. Deploying

### 2a. Code-only change (Worker JS, no image or config change): gradual

Run GitHub Actions → **Deploy veya-production** with `mode=gradual` and `confirm=veya-production`, or push a `prod-v*` tag. The workflow:

1. Runs node tests and config/pin tests. It refuses `gradual` if `deploy/cloudflare/`, `veya/`, `web/`, `requirements.lock`, `artifacts/release/manifest.json`, `models/production/` or `.dockerignore` changed since the previous `prod-v*` tag.
2. Runs `wrangler versions upload --config wrangler.production.jsonc`, which uploads the new version without serving it.
3. Runs `wrangler versions deploy NEW@10% OLD@90%`, soaks, then checks `/healthz` 20 times with at most 1 failure allowed. It repeats at 50%, then goes to 100%.
4. If any health check fails, it automatically reverts to `OLD@100%`.

Gradual deployments move **Worker code only**. The container image and container config are not versioned this way.

### 2b. Image, config, binding or manifest change: full

Use `mode=full` (`wrangler deploy --config wrangler.production.jsonc`). This also happens automatically when `production/veya-dynamics` is pushed. A full deploy builds the image and starts a Containers rollout (`rollout_step_percentage` defaults to 10% → 100% across the 2 instances). During the rollout, the Worker routes around a replica that is not ready (see §4).

### 2c. Manual verification after any deploy

```bash
BASE=https://veya-production.veya-8d0.workers.dev
curl -fsS $BASE/healthz            # edge only; never touches a container
curl -fsS $BASE/v1/models
curl -fsS -H "authorization: Bearer $VEYA_EDGE_TOKEN" $BASE/readyz
curl -fsS -H "authorization: Bearer $VEYA_EDGE_TOKEN" $BASE/__admin/replicas | jq
curl -sS -X POST $BASE/v1/decisions -H "authorization: Bearer $VEYA_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"primitive":"choice","state":"I was charged twice","instructions":"Route this ticket","criteria":{"billing":"Payments and refunds","technical":"Bugs and errors"}}'
```

## 3. Rollback

Worker versions currently on record (check them before acting with `npx wrangler deployments list --config wrangler.production.jsonc`):

| Version | Notes |
|---|---|
| `3a1122f3…` | current production at the time of writing |
| `f0fcbe63…` | previous |
| `53cd4471…` | before that |

```bash
npx wrangler deployments list --config wrangler.production.jsonc     # full version IDs
npx wrangler rollback <full-version-id> --config wrangler.production.jsonc \
  --message "rollback: <reason>"
# Or keep both versions live and shift traffic:
npx wrangler versions deploy <old-id>@100% --config wrangler.production.jsonc -y
```

Caveats:

- `wrangler rollback` restores **Worker code and bindings only**. It does not roll back the container image, D1 data, or secrets. If a bad image ships, re-deploy the previous git revision in full mode.
- A version older than the current release pins can fail the container boot check. `supervise.py` compares the Worker's `VEYA_*` vars with `/app/boot/release-pins.json` in the image. If the pins differ, the container refuses to start and `/readyz` returns 503. When the manifest changed between releases, roll back code **and** image together with a full deploy of the matching commit.
- If you need to override a gradual split, use `npx wrangler versions deploy <old>@100%`.

## 4. Failover drill (run monthly and before every launch)

The Worker picks one of the two replicas at random. If that replica's `/readyz` probe fails, or the replica returns 5xx, the Worker tries the other one. Readiness is cached per isolate for `VEYA_READY_CACHE_MS` (default 15 s). A failure clears the cache.

```bash
BASE=https://veya-production.veya-8d0.workers.dev
AUTH="authorization: Bearer $VEYA_EDGE_TOKEN"

# 0. Baseline: both replicas report ready_status 200.
curl -fsS -H "$AUTH" $BASE/__admin/replicas | jq

# 1. Start load (separate terminal). Must stay green throughout.
VEYA_BASE_URL=$BASE VEYA_API_KEY=$VEYA_API_KEY VEYA_DURATION=10m k6 run loadtest/k6-decisions.js

# 2. Stop replica 0.
curl -fsS -X POST -H "$AUTH" $BASE/__admin/replicas/0/stop      # 202
curl -fsS -H "$AUTH" $BASE/__admin/replicas | jq               # replica 0 ready_status != 200

# 3. Watch k6. Expect no sustained 5xx; requests fail over to replica 1.
#    Check that the AE replica column moves to 1 (query in §5).

# 4. Bring replica 0 back (destroy + start, in the background).
curl -fsS -X POST -H "$AUTH" $BASE/__admin/replicas/0/restart   # 202
#    Poll until ready; a cold start fetches about 842 MB from R2.
watch -n10 "curl -s -H '$AUTH' $BASE/__admin/replicas | jq -c '.replicas[]|{index,ready_status,state}'"

# 5. Repeat with replica 1.
```

Pass criteria:

- The k6 thresholds pass (p95 under `VEYA_P95_MS`, error rate under 1%).
- No 503 `container_unavailable` responses while one replica is up.
- Both replicas report `ready_status: 200` within 10 minutes of a restart.

Record the results (date, p95, errors, recovery time) in the incident log.

Admin route responses: `202` accepted, `404 unknown_replica` replica index out of range, `501 not_supported` the container class lacks the method, `502 replica_action_failed` the stop or restart threw. Every `/__admin/*`, `/readyz`, `/startupz` and `/version` route returns `401` without the admin secret, and `401` for everyone if the secret isn't set (fail closed).

## 5. Analytics Engine readback

Each inference writes one data point:

| Field | Meaning |
|---|---|
| `index1` | organization id |
| `blob1` | project id |
| `blob2` | API key id |
| `blob3` | endpoint (`/v1/decisions`, `/v1/responses`) |
| `blob4` | primitive or task |
| `blob5` | decision route from the model response (for example `packed` or `shortlist_k8_packed_rerank`) |
| `blob6` | HTTP status, as a string |
| `blob7` | replica index that served the request (new) |
| `double1` | edge latency in ms |

Query with the SQL API. The token needs *Account Analytics: Read*.

```bash
curl -sS "https://api.cloudflare.com/client/v4/accounts/8d0a6a64de03f4c9b14184398e75aa6e/analytics_engine/sql" \
  -H "Authorization: Bearer $CF_ANALYTICS_TOKEN" --data-binary @- <<'SQL'
SELECT blob3 AS endpoint, blob6 AS status, blob7 AS replica,
       SUM(_sample_interval) AS requests,
       quantileWeighted(0.95)(double1, _sample_interval) AS p95_ms
FROM veya_production_usage
WHERE timestamp > NOW() - INTERVAL '1' HOUR
GROUP BY endpoint, status, replica
ORDER BY requests DESC
SQL
```

Other useful queries (same endpoint):

```sql
-- Per-project usage, last 24 h
SELECT blob1 AS project, SUM(_sample_interval) AS requests
FROM veya_production_usage WHERE timestamp > NOW() - INTERVAL '1' DAY
GROUP BY project ORDER BY requests DESC;

-- Error rate per 5 minutes
SELECT toStartOfInterval(timestamp, INTERVAL '5' MINUTE) AS t,
       SUM(IF(blob6 >= '500', _sample_interval, 0)) / SUM(_sample_interval) AS error_rate
FROM veya_production_usage WHERE timestamp > NOW() - INTERVAL '6' HOUR
GROUP BY t ORDER BY t;

-- Route mix
SELECT blob5 AS route, SUM(_sample_interval) AS n
FROM veya_production_usage WHERE timestamp > NOW() - INTERVAL '1' DAY
GROUP BY route ORDER BY n DESC;
```

D1 has the same data at full fidelity:

```bash
npx wrangler d1 execute veya-production --remote --config wrangler.production.jsonc \
  --command "SELECT route, http_status, COUNT(*) FROM usage_ledger WHERE created_at > datetime('now','-1 hour') GROUP BY 1,2"
```

Follow-up migration (not applied): `ALTER TABLE usage_ledger ADD COLUMN replica INTEGER;`, then add `replica` to the INSERT in `production_edge.mjs`. Until then the replica is recorded only in AE `blob7`. Before this change, D1 `route` held the replica index; it now holds the decision route.

## 6. Tailscale cutover sequence (Funnel origin → Cloudflare Containers)

The root `veya` Worker still proxies to `VEYA_ORIGIN=https://benchbrex.taila8b570.ts.net`, a Tailscale Funnel to the Mac. Cut over in this order:

1. **Prove production.** Run `veya-production` through §2c and pass the failover drill in §4 under k6.
2. **Freeze.** Note the current `veya` Worker version (`npx wrangler deployments list`, root config). This is the Tailscale rollback point.
3. **Attach domains to `veya-production`** (§7) and check them over HTTPS.
4. **Shift clients.** Point the site/API base URL at `https://api.veyadynamics.com` (or `veya-production.veya-8d0.workers.dev`) and redeploy the site.
5. **Soak 24 h** with Funnel still up. Watch the AE error rate (§5) and `veya` Worker traffic, which should drop to zero.
6. **Retire the origin.** Stop the Funnel (`tailscale funnel reset`) and the local LaunchAgent. Then repeat §2c with the Mac offline.
7. **Retire the proxy.** Either change `veya` (root config on `main`) to redirect to the new domain, or keep it as the documented emergency path.

**Rollback during steps 4–6:** restart the Funnel (`tailscale funnel --bg 3000`, matching the previous setup), then roll `veya` back to the step-2 version. DNS and custom domains on `veya-production` can stay in place.

## 7. Custom domains, TLS and WAF

Zone: `veyadynamics.com` (it must be an active zone in this account).

| Hostname | Target |
|---|---|
| `api.veyadynamics.com` | `veya-production` (API) |
| `app.veyadynamics.com` | the site Worker, `veya-production` once it serves `site/dist`. **Decision needed:** the production config does not bind assets yet |
| `docs.veyadynamics.com` | docs build. **Decision needed:** which Worker or Pages project |

Attach a domain as a Worker Custom Domain. Cloudflare creates the DNS record and the edge certificate. Either use the dashboard (Workers & Pages → `veya-production` → Settings → Domains & Routes → Add → Custom domain), or add it to `wrangler.production.jsonc` and full-deploy:

```jsonc
"routes": [
  { "pattern": "api.veyadynamics.com", "custom_domain": true }
]
```

Then:

- **TLS:** zone → SSL/TLS → Overview → **Full (strict)**. Set *Always Use HTTPS* on, minimum TLS **1.2**, TLS 1.3 on. The Worker already sends `Strict-Transport-Security: max-age=63072000; includeSubDomains` on every response. Only add `preload` after every subdomain is confirmed HTTPS-only.
- **WAF:** turn on the Cloudflare Managed Ruleset and the OWASP Core Ruleset (start in log mode, then block). Add custom rules:
  - Block methods other than `GET`, `HEAD`, `POST` and `OPTIONS` on `api.`.
  - Challenge or block `/__admin/*` from outside your admin IPs. The secret is still required, so this is defence in depth.
  - Rate-limit `POST /v1/*` per IP, generously (for example 300/min), as a backstop to the per-key Worker limiter.
  - Limit request body size. The Worker rejects anything over 32 KB itself.
- Keep `workers_dev` enabled until the domains are proven, then consider turning it off.

## 8. Operational knobs (Worker vars and container env)

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `VEYA_INFER_TIMEOUT_MS` | Worker var | 15000 (1000–120000) | Container inference deadline. Returns `504 upstream_timeout` |
| `VEYA_PROBE_TIMEOUT_MS` | Worker var | 5000 | Readiness probe deadline |
| `VEYA_READY_CACHE_MS` | Worker var | 15000 (0–60000) | Per-isolate readiness cache. `0` probes on every request |
| `VEYA_EDGE_TENANT_CAPACITY` | container env | 30 (12–600) | Gateway token bucket per API key, when the edge sets `X-Veya-Tenant` |
| `VEYA_TORCH_THREADS` | container env | CPU affinity count (≤64) | `torch.set_num_threads` in the model server |
| `VEYA_EMBED_CACHE_MAX` | container env | 50000 | LRU bound for the embedding cache |

Edge errors are JSON with `code`, `detail` and `request_id`:

- `413 request_too_large`: more than 32 KB streamed, whatever the `content-length` says.
- `503 auth_unavailable`: D1 key lookup failed. Comes with `retry-after: 5`.
- `503 rate_limit_unavailable`
- `503 container_unavailable`: no ready replica. Comes with `retry-after: 5`.
- `504 upstream_timeout`

## 9. Release identity and re-minting

This release has the following identity. The `release_id` is unchanged, so the R2 object key is unchanged:

- **Release ID:** `0.3.5-audit-baseline.1`
- **Checkpoint:** `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
- **Manifest:** `80720058c6b07dfef0640171d4193ff7bcaa3f40629ac60df7f9b28e074d65a4`. It was re-minted after switching `torch.load` to `weights_only=True` and bounding the embedding cache. The previous manifest was `cd1e2029…`.

`artifacts/release/runtime_audit.json` → `manifest_repin` records that the audit measurements were taken under `cd1e2029…` and have not been re-run. Re-run `research/veyabench/run_runtime_audit.py` with checkpoint access before activation.

The manifest digest must be identical in all of these places (`tests/test_wrangler_configs.py` enforces it):

- `artifacts/release/manifest.json`
- `wrangler.staging.jsonc` and `wrangler.production.jsonc` vars
- `deploy/cloudflare/release-pins.json`
- the `deploy/cloudflare/Dockerfile` ENV
- `scripts/write_deploy_meta.py`
- `worker/build-meta.mjs`

If any of them disagree, the container refuses to boot.

The R2 `releases/<id>/manifest.json` object (if one was uploaded) still has the old digest. Re-upload it before relying on it.
