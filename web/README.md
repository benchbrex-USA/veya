# Veya model website

The browser calls only the same-origin `/api/*` gateway. The gateway validates requests, forwards them to the existing Veya HTTP service, validates and whitelists responses, then adds request ID, latency, and release metadata. The public response names the served system **Veya Core**; the audited upstream checkpoint identifier is kept internal. There is **no mock inference path** in the website. If the model is unavailable, the UI reports an outage.

## Run

From the repository root, in separate terminals:

```bash
VEYA_DEVICE=cpu python3 -m web.model_server
python3 -m web.server
```

Open <http://127.0.0.1:3000/experience>. The model process loads and verifies `artifacts/release/manifest.json` before binding port 8080. `VEYA_MODEL_URL` defaults to `http://127.0.0.1:8080`; `VEYA_MODEL_PORT` and `VEYA_WEB_PORT` set the two ports. `VEYA_WEB_HOST` defaults to loopback; binding another address requires `VEYA_PUBLIC_ORIGIN` set to the canonical HTTPS origin. An optional `VEYA_MODEL_TOKEN` is sent only by the gateway as a bearer token. It is never included in browser assets or responses.

## Actual model contract

| Endpoint | Website body | Current response |
|---|---|---|
| `POST /v1/choice` | `state`, `instructions`, `criteria` object | `decision`, `answer`, candidate `probabilities`, `confidence`, `route`, `calibration`, `shortlist`, `candidate_count`, `defer_reason` |
| `POST /v1/noul` | `state`, `instructions` | `decision`, numeric true probability in `answer`, `confidence`, `defer_reason`; no probability map |
| `POST /v1/score` | `state`, `instructions`, ordered `criteria` array | `decision`, categorical expected value in `answer`, indexed `probabilities`, `confidence`, `defer_reason`; gateway attaches the submitted label legend |
| `POST /v1/systemone` | `state`, `questions` keyed by question ID | One `decision`; `answers` keyed by submitted IDs; `routes` where available. This release does not expose partial success. |
| `GET /health` | none | `{ "status": "ok" }` |
| `GET /version` | none | Version, checkpoint, release, supported languages/primitives, route and scorer metadata. Gateway publishes a bounded public subset. |

The website accepts 2–19 candidates on the packed Choice route and 20–100 on the learned top-eight shortlist route. The experimental hypothesis scorer is **not loaded** by `artifacts/release/manifest.json`. Unsupported language returns `decision: "defer"` and `defer_reason: "unsupported_language"`. The audited release verdict classifies English Choice as a research release; Noul and categorical Score remain research only. This is not a production reliability guarantee.

## Gateway boundaries

- At most 32 KB request bytes, 12 KB state, 500-character question, 2–100 Choice candidates, 2–10 Score labels, and 1–8 SystemOne questions. Candidate IDs are unique after Unicode NFKC/case/whitespace normalization. Candidate content is forwarded as structured data without semantic rewriting.
- Same-origin JSON POSTs only. Optional `VEYA_ALLOWED_ORIGINS` lists additional browser origins. Do not expose the model service directly to the public internet.
- Per-socket-IP sliding window: 12 cost units per 60 seconds, with higher cost for large Choice sets. At most two concurrent upstream inferences per website process. Returns `429` with `Retry-After` or `503 model_busy` under pressure. This limiter is process-local; public multi-instance deployment needs a shared limiter at the ingress or in the gateway.
- 20-second inference timeout, 512 KB upstream response cap, server-side response schema checks, and bounded `/version` metadata. Invalid model output returns `502`, never a fabricated result.
- Safe structured telemetry logs request ID, primitive, candidate count, route, deferred state, status, latency, model version, and error code. Raw state, questions, candidate text, IP, and credentials are not logged by this gateway. A future reverse proxy or platform may have its own logging and retention policy.

Errors are JSON with `code`, `detail`, and `request_id`; rate limits also include `retry_after` and the `Retry-After` header. Upstream schema rejection keeps its structured `defer_reason`. The browser offers cancellation and retry, and prevents duplicate submissions while a request is pending.

## Checks

```bash
python3 -m pytest -q tests/test_web_gateway.py tests/test_server_security.py
node --test tests/test_model_client.mjs
node --test tests/test_site_content.mjs
python3 -m ruff check web tests/test_web_gateway.py
node --check web/static/site.js
node --check web/static/model-client.mjs
python3 -m py_compile web/server.py web/contracts.py web/model_server.py
```

The website binds loopback by default. With `VEYA_PUBLIC_ORIGIN`, HTML gets route-specific canonical/OG URLs, `/robots.txt` permits indexing, and `/sitemap.xml` lists the route registry. The local preview disallows indexing. Image assets use a five-minute cache; HTML, CSS, JavaScript, JSON route data, and API responses use `no-store` to prevent mixed-version modules. `python3 -m pip wheel . --no-deps --no-build-isolation` packages the website modules and assets. The model service still requires the audited release bundle from this repository; the wheel alone is not a model deployment. Public deployment needs TLS termination, trusted proxy configuration, a shared rate limiter, and infrastructure privacy review. The checkpoint's research status remains a model-release gate.

For site ownership, content changes, and the protected incremental workflow, see [`docs/WEBSITE_ARCHITECTURE.md`](../docs/WEBSITE_ARCHITECTURE.md) and [`docs/CHANGE_PROTOCOL.md`](../docs/CHANGE_PROTOCOL.md).
