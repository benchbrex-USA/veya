# Customer Worker `veya`: Cloudflare-native edge

`veya` (https://veya.veya-8d0.workers.dev) used to reverse-proxy every request to the
`web/server.py` gateway on a Mac via Tailscale Funnel (`VEYA_ORIGIN`). When the Mac was off,
customers got Cloudflare 530 "Origin DNS error" pages. It now runs entirely on Cloudflare.

## Design

| Path | Served by |
|---|---|
| `/`, the pages in `web/static/site-config.json` | Worker renders `web/static/index.html` with the page title and description, as `_page_html()` did |
| `/static/<file>` (`.css .js .mjs .json .png .svg`) | Workers static assets (`ASSETS`, directory `web/static`), only through the Worker (`run_worker_first: true`) |
| `/robots.txt`, `/sitemap.xml` | Same rules as the gateway (`VEYA_PUBLIC_ORIGIN` unset gives `Disallow: /` and no sitemap) |
| `GET /api/benchmark` | `artifacts/release/blind_clinc150_val_result.json`, bundled at build time; same payload as the gateway |
| `POST /api/infer` | `veya-production` `POST /v1/decisions` over the `VEYA_API` **service binding**, with the server-side project key `VEYA_SITE_KEY` (project `veya-public-site`, scope `decisions:invoke`). Browsers never see a key. |
| `GET /api/health` | Probes `veya-production` `/healthz` over the binding |
| `GET /api/version` | `veya-production` `/v1/version`, mapped to the gateway's version shape |
| `/healthz` | Edge liveness `{"status":"alive"}` |
| `/readyz`, `/startupz`, `/__edge_health` | Real probe of `veya-production` (3 s timeout). 200 when it answers, 503 JSON when it doesn't. |

### Gateway behaviour kept for `/api/infer`

- 32 KB cap, counted on the streamed bytes (chunked uploads included).
- `Content-Type: application/json` only.
- Same-origin check (`Origin` must be this host, or listed in `VEYA_ALLOWED_ORIGINS`).
- Weighted per-IP rate limit, 12 units / 60 s (Choice with more than 19 candidates costs more), via the `VEYA_IP_RATE` rate-limit binding keyed on `CF-Connecting-IP`.
- Error envelope `{code, detail, error, error_code, request_id}`, plus `Retry-After` and `X-Request-Id`.

Every upstream failure is JSON:

| Upstream condition | Response |
|---|---|
| `model_busy` | 503 `model_busy` |
| deadline | 504 `timeout` |
| `veya-production` 429 | 429 `rate_limited` |
| 401/403/404/5xx, non-JSON, binding error | 503 `model_unavailable` (no upstream detail leaked) |
| 2xx with an invalid body | 502 `invalid_upstream_response` |

## Rate limits: what the public site can actually do

- **Per visitor:** 12 units/min per IP at the `veya` edge (as before).
- **Whole site:** all public traffic shares **one** production key. `veya-production` limits per key with
  `VEYA_RATE_INFERENCE` (20/min) and `VEYA_RATE_HEAVY` (6/min for non-Choice primitives: Noul, Score).
  These are Workers rate-limit bindings, so the counters are per Cloudflare location, not global.
  Beyond that, visitors get 429 `rate_limited` (Retry-After 60).
- **Changing the site-wide budget:** `veya-production` has no per-project limit setting today. Raising it needs a
  production change, e.g. a separate rate-limit namespace selected for project `veya-public-site`, or a
  per-project limit column read by `limited()`.

## Contract differences vs the Mac gateway

- `primitive: "systemone"` is rejected (400 `unsupported_primitive`); production doesn't serve it in this release.
- `/api/version` is derived from production `/v1/version` plus pinned vars. Fields such as `calibration_version`,
  `reranker_id` and `hypothesis_scorer_id` aren't included; inference responses still carry `model_version`,
  `release`, `architecture_version`, `release_status`, `calibration_version`.
- `/api/health` reflects the production edge (`/healthz`), not a model-server `/health` call.
- The site-wide budget above is new; the Mac gateway had no global cap besides `MAX_INFLIGHT=2`.
- `/readyz` and `/startupz` now describe this edge's upstream; `/__edge_health` returns 503 when production doesn't answer.

## Operations

- **Deploys are gated.** The Workers Builds trigger on `main` runs `npx wrangler deploy --dry-run`. Ship with:

  ```bash
  npx wrangler versions upload --message "<what>"
  npx wrangler versions deploy <new>@10% <current>@90% -y   # verify, then
  npx wrangler versions deploy <new>@100% -y
  ```
- **Secret:** `VEYA_SITE_KEY` is set with `wrangler versions secret put VEYA_SITE_KEY` (doesn't deploy). To rotate:
  1. create a new key in production D1;
  2. `versions secret put`;
  3. deploy;
  4. revoke the old key (`revoked_at`).
- **Rollback:** `npx wrangler rollback <previous Cloudflare-backed version>`. The last Funnel-proxy version
  (`2d0891c9-23f5-492f-b050-1e8b8f5d5bc0`) only works while Tailscale Funnel is running on the Mac.
