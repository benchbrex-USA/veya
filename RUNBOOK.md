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

### 3a. `wrangler rollback` is only for code-only releases

`wrangler rollback` and `wrangler versions deploy <old>@100%` swap **Worker code, vars and secrets only**. The container image, the container application config, and D1 data are **not** rolled back. This was verified on 7 Oct 2026 on both staging and production: after rolling the Worker back, both containers kept running image `24a5bdca…` (manifest `80720058…`).

That leaves an old Worker in front of a new image. The old Worker injects its own `VEYA_MANIFEST_SHA256`, and the image's `release-pins.json` pins the new one. Instances that are already running keep serving. **Any instance that starts afterwards fails `supervise.py` `pins_match` and never becomes ready.** Instances start on deploys, sleep/wake, crashes and failover drills.

**Rule:** if the release being rolled back changed the image, the manifest, the container config or bindings, do **not** use `wrangler rollback`. Run a **full deploy of the previous commit** instead. That rebuilds and rolls out the matching image.

```bash
# Example for release 42e0c9e (PR #14): the previous production commit is a76bc65.
git fetch origin && git checkout --detach a76bc65
python3 scripts/write_deploy_meta.py && npm ci && npm --prefix site ci && npm --prefix site run build
npx wrangler deploy --config wrangler.production.jsonc --message "rollback: full deploy of a76bc65"
# or: GitHub Actions -> Deploy veya-production, mode=full, on a prod-v* tag pointing at a76bc65
```

Expect a container rollout: about 2–3 minutes per instance, during which that instance is not ready. Verify with §2c, then check `/__admin/version` for `source_commit`.

Measured on 7 Oct 2026, for reference:

| Step | Command time | Serving the target |
|---|---|---|
| Staging rollback / roll-forward | 3.0 s / 2.4 s | 116/116 probes 200 |
| Production rollback / roll-forward | 3.0 s / 2.3 s | old code within 20 s, new code within 4 s |

These were code-only swaps over an unchanged image.

Other caveats:

- A rollback restores the secret values of the target version. A secret rotated after that version (for example `VEYA_EDGE_TOKEN`, rotated 7 Oct 09:42 IST in `2bcbff1f`) reverts to its old value. Re-run `wrangler secret put` after rolling back.
- To override a gradual split, use `npx wrangler versions deploy <old>@100%`. This is code-only, with the same rule as above.

## 4. Failover drill (run monthly and before every launch)

The Worker picks one of the two replicas at random and works within one overall deadline (`VEYA_INFER_TIMEOUT_MS`, 15 s).

- **Attempt budget.** Each replica attempt (readiness probe plus inference) gets its own budget, `VEYA_ATTEMPT_TIMEOUT_MS`, default 6 s.
- **One retry.** If the attempt fails, the Worker retries once on the other replica. A failure is a 5xx, a timeout or a throw. The retry uses whatever remains of the deadline.
- **Suspect replicas.** A replica that timed out or errored is "suspect" for `VEYA_SUSPECT_MS` (10 s) and is tried second.
- **Readiness cache.** Readiness is cached per isolate for `VEYA_READY_CACHE_MS` (15 s). A failure clears the cache.

**Stopping a replica doesn't keep it down.** The next request routed to it, including the admin report's own `/readyz` probe, starts it again (cold start ≈ 60–70 s). Until it's ready, requests hitting it block inside the attempt budget.

Results of the 7 Oct 2026 drill, before the per-attempt timeout: 36 req/min, stop of replica 1 at 09:46:02 IST, restart at 09:50:30.

- **Errors:** 7/361 failed (1.9%): 4 × 504 at 15.3 s and 3 × 503. All of them were in the first ~20 s after the stop and after the restart: requests that landed on the cold instance used the whole deadline.
- **Latency:** p50 1.51 s, 2xx p95 2.24 s.
- **Recovery:** replica 1 served again about 60 s after the stop and about 70 s after the restart.

The per-attempt timeout and retry exist to remove those failures. Re-run the drill after deploying them.

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
| `VEYA_ATTEMPT_TIMEOUT_MS` | Worker var | 6000 (500 to deadline−1000) | Per-replica attempt budget before retrying on the other replica (`/v1/decisions` only) |
| `VEYA_SUSPECT_MS` | Worker var | 10000 (0–120000) | How long a replica that timed out or errored is tried second |
| `VEYA_SLEEP_AFTER` | Worker var → container class | staging `15m`, production `1h` (fallback `8h`; 1m–24h) | Container idle timeout. Each wake costs a cold start of about 60–70 s with the checkpoint fetched from R2 |
| `VEYA_MAX_INFLIGHT` | Worker var → container env | 2 (1–8) | Gateway admission limit per replica; above it the gateway sheds load with `503 model_busy` |
| `VEYA_INFLIGHT_STALE_MS` | Worker var → container class | 120000 (30000–3600000) | If the container class reports requests in flight but none has started or finished for this long, the count is treated as leaked and reset so `sleepAfter` can expire |

Edge errors are JSON with `code`, `detail` and `request_id`:

- `413 request_too_large`: more than 32 KB streamed, whatever the `content-length` says.
- `503 auth_unavailable`: D1 key lookup failed. Comes with `retry-after: 5`.
- `503 rate_limit_unavailable`
- `503 container_unavailable`: no ready replica. Comes with `retry-after: 5`.
- `503 model_busy`: both replicas shed load (gateway admission limit reached). Comes with `retry-after: 2` and `retry_after` in the body. Clients should back off and retry; this is capacity, not an outage.

### Per-project rate limits (tiers)

`veya-production` reads each project's tier from D1 `projects.rate_tier` (migration `product/migrations/0002_project_rate_tier.sql`, additive). NULL means `default`. The mapping lives in `worker/rate_tiers.mjs`:

| Tier | Choice | Noul / Score | Counted per | Bindings (namespace) |
|---|---|---|---|---|
| `default` (NULL) | 20/min | 6/min | API key | `VEYA_RATE_INFERENCE` (2001), `VEYA_RATE_HEAVY` (2002) |
| `site` | 150/min | 45/min | project | `VEYA_RATE_SITE` (2003), `VEYA_RATE_SITE_HEAVY` (2004) |

- **Sizing.** `site` is sized to the pool: 2 replicas serve about 2.6 rps (~156/min). Beyond that the gateway sheds load with `503 model_busy`, which is still the hard cap. Workers rate limits are counted per Cloudflare location and are approximate.
- **Per-visitor limits stay at the front end.** The `veya` edge keeps its 12 units/min per IP.
- **Fallbacks.** Unknown tiers, or missing tier bindings, fall back to `default`, counted per key. A missing `rate_tier` column (migration not applied) falls back to the key-only auth query.
- A rate-limited response carries `x-veya-rate-tier`.

Set or clear a tier (production D1):

```bash
npx wrangler d1 execute veya-production --remote -c wrangler.production.jsonc \
  --command "UPDATE projects SET rate_tier='site' WHERE id='project-veya-public-site'"   # or rate_tier=NULL
```

Changes apply on the next request; there is no cache.

### Idle sleep (`sleepAfter`)

`@cloudflare/containers` 0.3.7 counts a request as in flight until its response body has been **piped to the end**. `containerFetch` decrements only in `res.body.pipeTo(...).finally(...)`, and the pipe has no buffer. While anything is in flight, `isActivityExpired()` renews the timer, so the container never sleeps. On 7 Oct 2026 the staging container (`VEYA_SLEEP_AFTER=15m`) was still running after 24 idle minutes.

Rules (enforced by `worker/body_hygiene.mjs` and `tests/test_container_body_hygiene.mjs`):

- Every container response the edge does not forward to the client is read or cancelled. That covers readiness probes, retried attempts, load-shed 503s, responses rejected by the pool, admin report probes, and answers that arrive after a deadline (cancelled when they land, via `waitUntil`).
- Nothing pings containers on a timer. Readiness probes happen only on behalf of a real request, at most once per `VEYA_READY_CACHE_MS` per isolate and replica.
- Safety net in `VeyaContainer`: if requests have been in flight with no request starting or finishing for `VEYA_INFLIGHT_STALE_MS` (2 min), the count is reset, a `container_inflight_reset` warning is logged, and the idle time counts from the last real activity.

**Observe it.** Each DO alarm (at most every 3 min) logs:

```json
{"event":"container_activity","inflight":0,"idle_ms":…,"sleep_after":"15m","expires_in_ms":…,"expired":false}
```

When the timer expires, the library logs `Activity expired, signalling container to stop`. In Workers Logs, filter `$metadata.service = veya-staging`. The Containers API (`/containers/applications/{id}/deployments`, `current_placement.status.container_status`) shows `running` or stopped. The next request cold-starts the container (about 60–90 s; staging `/readyz` returns 503 meanwhile).

### Capacity (standard-2: 1 vCPU, 6 GiB, torch on one thread)

- A `choice` call is CPU-bound at about 1.1–1.3 s inside the container (edge p50 1.5 s).
- Two replicas sustained **1.5 rps** with p50 1.53 s / p95 2.13 s / p99 3.07 s (7 Oct 2026 baseline). Short bursts at that rate produced 3/450 `model_busy` responses.
- At 0.6 rps, the healthy windows of the drill had p50 ~1.5 s / p95 ~2.3 s and no errors.
- `VEYA_MAX_INFLIGHT` stays at **2**. With one vCPU, admitting more calls only queues them inside the model server: latency grows linearly while throughput stays the same, and RSS is already about 2 GiB per replica. For more than ~1.2 rps sustained, add replicas or move to `standard-3` (2 vCPU) rather than raising the limit.
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
