# What was already there, and what was added

Checked against `origin/main` at commit `0944d31` and the staging branch `deploy/cloudflare-staging` at `0e2feb1`. Line numbers below are the files as they are on that commit. Generated lockfiles are counted, not described line by line.

Nothing in `veya/`, `models/production/`, `artifacts/release/`, `web/contracts.py`, or `web/model_server.py` was edited for this deployment.

## Already on origin/main

This is the product that existed before the staging branch.

The decision model is the `veya/` package. Release `0.3.5-audit-baseline.1` is described by `artifacts/release/manifest.json`. The checkpoint file `models/production/model.safetensors` is 842,609,210 bytes and its SHA-256 is `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`. That file is gitignored. It was not rewritten.

`web/server.py` on `origin/main` was 1,066 lines. It already:

- reads `web/static/site-config.json` at import (`SITE_CONFIG`, then line 55)
- serves `/healthz`, `/readyz`, `/startupz`, `/version`, and `/api/version`
- accepts `POST /api/infer` for choice, noul, score, and systemone through `web/contracts.py`
- applies its own request-size and process-local rate limits
- starts beside `web/model_server.py`, which loads the release manifest and the checkpoint

`web/static/` was already the older public site. `deploy/hf-space/Dockerfile` already installed `torch==2.14.0` from the CPU index before `requirements.lock`. `requirements.lock` already pinned `torch==2.14.0`. `pyproject.toml` still says `torch>=2.0.0,<3`.

The public Worker on `origin/main` was not a container. `wrangler.jsonc` named the Worker `veya`, pointed `main` at `worker/veya-edge.mjs`, and set `VEYA_ORIGIN` to `https://benchbrex.taila8b570.ts.net`. That 95-line file fetched the Mac through Tailscale Funnel. It is deleted on this branch.

## Commits that only added notes

These three commits were already on the local branch before the implementation commits. They do not change runtime code.

| Commit | What it added |
|---|---|
| `239dc31` | `docs/superpowers/specs/2026-10-04-cloudflare-container-deployment-design.md`, 120 lines |
| `41bc644` | `docs/superpowers/plans/2026-10-04-cloudflare-container-deployment.md`, 566 lines |
| `c96d5f7` | `.gitignore` entries for local worktrees |

## Added on this branch

Three commits: `ff10486`, `07d3367`, `0e2feb1`. Compared with `origin/main`: 66 files, 10,930 insertions, 117 deletions. The deletions are the old edge proxy and a few lines of `web/server.py`.

### Worker that replaced the Tailscale proxy

`worker/veya-edge.mjs` is gone. Three new modules replace it.

`worker/veya-container.mjs`, 63 lines.

| Lines | What they do |
|---|---|
| 9–11 | Import `Container`, `getContainer`, and `ContainerProxy`, and re-export `ContainerProxy`. Line 11 was added in `0e2feb1`. The live Worker does not have it yet. |
| 27–37 | `VeyaContainer`: port 8000, no public internet, checkpoint URL and expected SHA for release `0.3.5-audit-baseline.1`. |
| 40–42 | Requests to `veya-r2.internal` go to the R2 handler. That handler is not a public route. |
| 44–62 | Public `fetch`: reject bad requests, apply the edge rate limit, answer `/__admin/version`, otherwise call container instance `veya-core`. |

`worker/edge_policy.mjs`, 201 lines. It never calls the container.

| Lines | What they do |
|---|---|
| 5–17 | Instance name `veya-core`, 4 MB body cap, public GET paths. |
| 37–66 | `guardRequest`: method, size, public health, bearer `VEYA_EDGE_TOKEN`, in-process limit of 60 per minute. |
| 68–75 | `forwardHeaders` strips `authorization` before the container sees the request. |
| 101–110 | Constant-time bearer compare. |
| 112–114 | Rate-limit key is `cf-connecting-ip`, not `X-Forwarded-For`. |
| 116–129 | Local limiter sets `Retry-After: 60`. |
| 132–168 | Cloudflare rate-limit bindings. Choice with 20 or more candidates, plus noul, score, quantum, and expert routes, use the heavy limiter. |
| 170–184 | `/__admin/version` returns release, hashes, and Worker version id. It does not return the token. |

`worker/artifact_route.mjs`, 69 lines.

| Lines | What they do |
|---|---|
| 7 | Host `veya-r2.internal`. |
| 9–13 | Only `model.safetensors`, `manifest.json`, and `model.safetensors.sha256`. |
| 17–31 | Reject another release id, `%`, `\`, `.`, and `..`. |
| 33–62 | GET and HEAD only. Stream `env.VEYA_ARTIFACTS`. Do not buffer the checkpoint. |

### Wrangler config

`wrangler.jsonc` was  the Tailscale proxy. It is now 85 lines.

| Lines | What they do |
|---|---|
| 5–6 | Name `veya-staging`, account `8d0a6a64de03f4c9b14184398e75aa6e`. The production Worker name `veya` is not this file. |
| 9–21 | Static assets from `./site/dist`. `/api/*`, `/__*`, `/healthz`, `/readyz`, `/startupz`, and `/version` hit the Worker. Other paths are the React app. Added in `07d3367`. |
| 22–35 | One container, `veya-runtime`, scheduling `default`, `instance_type` `standard-2`, `max_instances` 1. |
| 51–55 | R2 binding `VEYA_ARTIFACTS` to `veya-model-artifacts`. |
| 57–67 | Two rate-limit bindings, 60/min and 10/min. |
| 69–71 | Secret name `VEYA_EDGE_TOKEN` only. No secret value. |
| 75–81 | Release id and the checkpoint SHA. `VEYA_SOURCE_COMMIT` is still `c96d5f7`, which is older than this branch tip. |

`package.json` is 11 lines: `@cloudflare/containers` `0.3.7` and `wrangler` `4.147.0`. `package-lock.json` is the generated lock, 1,599 lines.

### Container image

`deploy/cloudflare/Dockerfile`, 72 lines. New file. It does not copy the checkpoint.

| Lines | What they do |
|---|---|
| 10–26 | Install `torch==2.14.0` from `https://download.pytorch.org/whl/cpu`, then `requirements.lock`, then assert CUDA is off. This order was copied from the existing Hugging Face Dockerfile in `07d3367`. |
| 34–39 | Copy `veya/`, `experts/`, `quantum/`, four gateway modules, and `web/static/`. Line 39 fails the image if `site-config.json` is missing. The static copy was required because the old gateway already reads that file. |
| 42–50 | Copy the rest of `models/production` except `*.safetensors`. |
| 56–68 | Boot environment, including the private checkpoint URL and the expected size and SHA. |
| 72 | Fetch and verify the checkpoint, then start the existing model server and the existing gateway. |

`deploy/cloudflare/fetch_release.py`, 97 lines.

| Lines | What they do |
|---|---|
| 16–22 | Names of storage credentials that must not be set. |
| 34–39 | Fail if any of those names are present. |
| 41–84 | Stream to `model.safetensors.part`, hash, check size, fsync, rename. A mismatch deletes the partial file. Redirects are refused. |
| 87–96 | `main` reads the URL, SHA, and size from the environment. |

`deploy/cloudflare/README.md` is 21 lines of deploy notes.

`.dockerignore` gained `node_modules`, `.wrangler`, `site`, `.env`, and the production `*.safetensors` paths so the image context does not carry the React app or the checkpoint.

### Gateway change

`web/server.py` grew from 1,066 lines to 1,232. Five lines were removed and 171 were added. The infer path was not replaced.

The new lines import `experts.router` and `quantum.service`, define `/api/quantum/analyze`, `/api/quantum/decide`, and `/api/experts/route`, and dispatch those POSTs before `/api/infer`. Choice, noul, and score still go through the original infer handler. `experts/` and `quantum/` did not exist on `origin/main`. They were added as new packages so those imports resolve. They are not loaded by the production release manifest.

`experts/` is 988 lines across 11 files: a registry, a router, evaluation helpers, and a promotion record that does not promote anything.

`quantum/` is 2,636 lines across 12 files: an optional Ising client, a decision schema, and a service used by the two new routes. It is off unless its own environment flags are set.

### Investor site

`site/` did not exist on `origin/main`. It is a Vite React application. The Worker serves `site/dist`. The container image does not contain it.

| File | Lines | Role |
|---|---|---|
| `src/main.tsx` | 14 | Mounts the app and the open-source fonts Inter and Source Serif 4. |
| `src/app.tsx` | 38 | `/` marketing, `/console`, `/docs`. |
| `src/mark.tsx` | 38 | Original line-and-node mark. Not a copied logo. |
| `src/marketing.tsx` | 166 | Header, hero, the three manifest primitives, the sealed CLINC150 numbers, and `POST /api/infer` examples. |
| `src/docs.tsx` | 40 | Short API notes taken from the current release. |
| `src/styles.css` | 262 | Paper, ink, hairline, and the existing Veya cobalt `#020076`. |
| `src/release.ts` | 55 | Release facts copied from the manifest. Choice is `RESEARCH RELEASE`. Noul and categorical score are `RESEARCH ONLY`. Max fusion stays `RESEARCH` and `promoted: false`. |
| `src/contract.ts` | 76 | Builds the same choice, noul, and score JSON the gateway already accepts. |
| `src/console.tsx` | 254 | Console screens. `Playground` starts at line 153. |
| `src/console.tsx` 160–167 | | Token is read from `sessionStorage` key `veya-demo-auth` or left empty. |
| `src/console.tsx` 182–197 | | If the token is shorter than 16 characters, the form stops. Otherwise it `POST`s `/api/infer` with `Authorization: Bearer` and the constructed body. Changed from `/api/playground` in `07d3367`. |
| `src/security.ts` | 89 | Header, key-hash, and host helpers. Not used by the live inference click. |
| `src/worker.ts` | 212 | An earlier separate console Worker. It is not deployed. The live Worker is `worker/veya-container.mjs`. |
| `wrangler.jsonc` | 41 | Unused second-worker config. Not the file Wrangler deploys. |
| `migrations/0001_console.sql` | 33 | Unused D1 schema. D1 was not created. |
| `test/product.test.ts` | 131 | Contract, key masking, and release-label tests. |
| `package-lock.json` | 1,901 | Generated. |

### Uploader that has been deleted from Cloudflare

`tools/cloudflare-r2-uploader/` uploaded the one checkpoint and was then removed from the account. The source remains.

| File | Lines | Role |
|---|---|---|
| `worker.mjs` | 135 | Bearer check, one object key, multipart create, part, complete, abort, and `GET /verify-object`. |
| `worker.mjs` line 6 | | `releases/0.3.5-audit-baseline.1/model.safetensors` only. |
| `upload.py` | 173 | 16 MiB parts, four at a time, then stream the object back and hash it. |
| `worker.test.mjs` | 38 | Auth and path tests. |
| `wrangler.jsonc` | 16 | Binding to `veya-model-artifacts`. |

### Tests

| File | Lines | What it checks |
|---|---|---|
| `tests/test_cloudflare_config.py` | 106 | One container, `standard-2`, R2 binding, assets directory, CPU Torch before the lock, `web/static` copied, Playground calls `/api/infer`, no `VEYA_ORIGIN`. |
| `tests/test_veya_container_edge.mjs` | 196 | Auth, size, method, rate-limit key, heavy choice, artifact path. |
| `tests/test_fetch_release.py` | 124 | Hash, rename, redirect refusal, no storage credential. |

## What was not built

The model, tokenizer, router, shortlist, calibration, and manifest were already there and were not edited.

No second checkpoint was written. The multipart upload assembled the same single `model.safetensors` object.

`veyadynamics.com`, WorkOS, billing, and a separate docs host were not built.

The container on the live URL is not serving the model. `/`, `/console`, and `/docs` return the site from Worker version `f18ff4a5-c3cc-4015-9911-affeb176e067`. `/readyz` returns 500 because that version lacks the `ContainerProxy` export. Commit `0e2feb1` has the export. Its image push timed out, so that commit is not the live Worker.
