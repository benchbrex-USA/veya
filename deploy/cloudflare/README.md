# Cloudflare Container

Staging Worker name: `veya-staging`.
Account: `veya` (`8d0a6a64de03f4c9b14184398e75aa6e`).
`wrangler deploy` from this repository root cannot replace the production Worker `veya`.

Do not deploy this branch until a fresh staging rollout builds the image, `/readyz` returns 200, and the same public Worker returns real `choice`, `noul`, and categorical `score` results. The live staging Worker is still an older upload that does not export `ContainerProxy`. Production `veya` and the Tailscale Funnel stay as they are.

## What ships

Browser → one Worker → one `veya-core` container → the existing gateway → the existing model server → one verified checkpoint.

The container reads that checkpoint through the private R2 binding `VEYA_ARTIFACTS` (`veya-model-artifacts`), host `veya-r2.internal`. No R2 credential is placed in the container. The object is `releases/0.3.5-audit-baseline.1/model.safetensors`.

Scheduling is `default`. `instance_type` is `standard-2`. `max_instances` is 1. `enableInternet` is false. `sleepAfter` is `8h` so an idle demo does not refetch the checkpoint; that also keeps the instance billable while it is awake. Warm `GET /readyz` before a meeting.

`pingEndpoint` is `127.0.0.1/readyz`. `@cloudflare/containers` 0.3.7 treats any HTTP response from that URL as "the port is open". The Worker still refuses `POST /api/infer` until `/readyz` is 200.

Experts, quantum, SystemOne, the second site Worker, and D1 are not on this path. `site/future/` is unused.

## Boot

`deploy/cloudflare/supervise.py` fetches and verifies the checkpoint, then starts the model server and the gateway. If the fetch fails, the process exits before either service starts. If either child exits, the supervisor stops the other and exits. The image pins in `release-pins.json` must match the container environment.

## Identity

`scripts/write_deploy_meta.py` hashes `artifacts/release/manifest.json`, checks the release id and checkpoint hash, and writes:

- `worker/build-meta.mjs` (source commit and dirty flag)
- `deploy/cloudflare/release-pins.json`

`source_dirty` is `false` only when the paths that ship are clean. Research files outside that list do not count. A committed commit id is stale by the time it is committed; regenerate during the build.

Workers Builds command:

```text
python3 scripts/write_deploy_meta.py && npm ci && npm --prefix site ci && npm --prefix site run build
```

Deploy command: `npx wrangler deploy`.

`versions upload` does not roll out the container image. Set `SKIP_DEPENDENCY_INSTALL=true` so the builder does not `pip install .` (that pulls CUDA torch).

## Credentials

Two Worker secrets, set at deploy time with `wrangler secret put`. Do not write the values into git, wrangler vars, or the site bundle.

- `VEYA_EDGE_TOKEN` authorizes `GET /__admin/version` only.
- `VEYA_DEMO_TOKEN` authorizes `POST /api/infer` only. It is the temporary demo token. Keep it out of `sessionStorage`. Rotate it after the meeting.

The Playground holds the demo token in memory. Rate limits are 20 inference requests per minute and 6 heavy requests per minute, keyed `veya-demo`. The isolate-local limiter is the fallback when the Cloudflare binding is absent.

## Staging indexing

`site/public/_headers` and Worker responses send `X-Robots-Tag: noindex`. Remove that header before any production hostname cutover. This staging URL is not the production URL.
