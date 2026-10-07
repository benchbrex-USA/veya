# What was built

This is the investor deployment of one Veya model. It is not a second model, and the multipart upload did not split the checkpoint. Cloudflare assembled one object.

The live site is up. The model container is not ready yet.

## One system

| Piece | What it is |
|---|---|
| Model | Veya Core, release `0.3.5-audit-baseline.1` |
| Checkpoint | `releases/0.3.5-audit-baseline.1/model.safetensors` |
| Storage | Private R2 bucket `veya-model-artifacts` |
| Container | One instance, `veya-core`, class `VeyaContainer`, app `veya-runtime` |
| Worker | `veya-staging` |
| Public URL | https://veya-staging.veya-8d0.workers.dev |
| Account | veya, `8d0a6a64de03f4c9b14184398e75aa6e` |

The same Worker serves the React site, `/console`, `/docs`, health routes, and `/api/*`. `/api/*` goes to the one container. There is no second website Worker, no second API Worker, and no Tailscale origin in this path.

## Checkpoint

| Check | Value |
|---|---|
| Size | 842,609,210 bytes |
| SHA-256 | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` |
| Manifest | same hash |
| Remote read-back | same hash, twice |

The bucket is R2 Standard. `r2.dev` is off. There is no public custom domain. The temporary upload Worker `veya-r2-uploader` was deleted after the hash check. The checkpoint object was kept.

The container is supposed to fetch that one object over the private host `veya-r2.internal`, hash it while streaming, and only then start the model. `/readyz` stays unavailable until that finishes.

## What you can open now

https://veya-staging.veya-8d0.workers.dev

Checked on the deployed Worker version `f18ff4a5-c3cc-4015-9911-affeb176e067`:

| Request | Result |
|---|---|
| `GET /` | 200, Veya Dynamics page |
| `GET /console` | 200, same application, console route |
| `GET /docs` | 200, same application, docs route |
| `GET /__edge_health` | 200, `{"edge":"ok","backend":"cloudflare-container"}` |
| `GET /healthz`, `/readyz`, `/api/version` | 500 |

The 500 is:

```text
Failed to start container: ctx.exports.ContainerProxy is undefined,
export ContainerProxy from the containers package in your worker entrypoint
```

That version of the Worker does not re-export `ContainerProxy`. The model therefore has not loaded, and Choice, Noul, and Score have not been run on this URL.

## Source

Branch: `deploy/cloudflare-staging`

| Commit | What it contains |
|---|---|
| `ff10486` | Staging Worker, container config, private R2 fetch, site, uploader tooling |
| `07d3367` | CPU Torch install, `web/static` in the image, site served by the same Worker, Playground calls `POST /api/infer` |
| `0e2feb1` | Re-export of `ContainerProxy` |

`0e2feb1` is on GitHub. It is not the Worker version that is live. The two builds of that commit failed while pushing the image to Cloudflare's registry:

```text
net/http: timeout awaiting response headers
```

The image that did push is the one from `07d3367`, without the `ContainerProxy` export. Instance type on that deploy was `standard-2`.

## What the image does

`deploy/cloudflare/Dockerfile`:

- Installs `torch==2.14.0` from the PyTorch CPU index before `requirements.lock`, so Linux does not take the CUDA wheel.
- Copies `web/static/`, including `site-config.json`, which `web/server.py` reads at import.
- Does not copy `model.safetensors`. Boot fetches the one R2 object.
- Refuses to start if an R2 or AWS secret is present in the container.

The site build is `site/dist`. Wrangler serves it as static assets. These paths run the Worker first: `/api/*`, `/__*`, `/healthz`, `/readyz`, `/startupz`, `/version`. Everything else, including `/console` and `/docs`, is the React application.

The Playground asks for a staging token in a password field. The token is not compiled into the site. On this Mac it is in `/tmp/veya-edge-token` and it is also set as the Worker secret `VEYA_EDGE_TOKEN` on `veya-staging`. It is not in git.

## What was deliberately left out

No `veyadynamics.com` DNS, no WorkOS, no D1 product database, no billing, no second model, and no promotion of the max-fusion retriever. English Choice remains a research release. Noul and categorical Score remain research only. Automation stays disabled.

## Tests that passed locally

- `node --test tests/test_veya_container_edge.mjs`
- `python3 -m pytest tests/test_cloudflare_config.py`
- `cd site && npm test && npm run build`

The built site contains `/api/infer` and does not contain `VEYA_EDGE_TOKEN`.

## Still open

The next build has to finish pushing `veya-runtime` and roll out commit `0e2feb1`. Until that Worker is the live version, the investor URL shows the site and the container does not start.

**BLOCKED — the fixed Worker is pushed, but the container image push to Cloudflare's registry timed out, so the live Worker is still the previous version and the model is not ready.**
