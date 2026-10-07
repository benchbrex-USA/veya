# Cloudflare Container Deployment Design

> Historical design note from 2026-10-04. It is not the operator runbook.
> The active Worker configuration is `wrangler.jsonc`.

# Cloudflare Container Deployment Design

## Goal

Run the exact audited Veya release entirely on Cloudflare so the public website and inference API no longer depend on the user's Mac, Tailscale, or an external tunnel.

## Current constraints

- The model is a Python 3.11 CPU PyTorch service using pinned `safetensors` weights and release-manifest hash checks.
- Audited cold start is approximately 91 seconds and measured RSS is 1,638.9 MB.
- A normal Worker has a 128 MB memory limit and cannot host this runtime.
- Workers AI exposes Cloudflare-supported models; it is not a general upload target for this custom audited model.
- Cloudflare Containers are generally available on Workers Paid. Containers have no free allocation.
- The selected operating mode is always-on. Estimated cost is approximately $25–30/month for a 1-vCPU, 3-GiB instance plus the $5 Workers Paid plan, depending on active CPU, disk, and egress.

## Architecture

```text
Browser
  -> Veya Worker (public edge, existing workers.dev URL)
    -> singleton Veya Container (private, 1 vCPU / 3 GiB RAM)
      -> web.server gateway on the exposed container port
        -> web.model_server on 127.0.0.1:8080

Veya Container boot
  -> secret-authenticated Worker artifact endpoint
    -> private R2 bucket
      -> model.safetensors
```

The existing Worker remains the sole public entry point. It routes normal requests to one deterministic Container instance. The model service remains loopback-only inside the container; only the validated website gateway is exposed to the Worker.

## Components

### Worker and Container controller

Replace the external `VEYA_ORIGIN` fetch with a Cloudflare Container binding and a singleton instance ID. The Worker preserves the current public URL and edge health endpoint. It also owns:

- container lifecycle and request routing;
- a scheduled warm-up request so platform restarts recover before user traffic where possible;
- a private checkpoint-download route backed by R2;
- Worker and Container logs.

The controller uses the current Container API rather than the legacy external-origin proxy. Host restarts are expected platform behavior; the next scheduled warm-up or user request starts a replacement.

### Container runtime

Create Cloudflare-specific deployment scaffolding under `deploy/cloudflare-container/`, reusing the existing audited sources and boot supervisor. The image uses Python 3.11 and CPU-only PyTorch 2.14.0. It exposes only the gateway port. The gateway starts first for liveness, then the checkpoint is fetched and verified, then the model starts. `/readyz` remains 503 until all release hashes and model bindings pass.

The instance type is:

- 1 vCPU;
- 3,072 MiB memory;
- 8,000 MB ephemeral disk;
- one singleton instance.

### Private checkpoint storage

Create a private R2 bucket for the checkpoint. Upload `model.safetensors` under a release-specific object key. Do not put the checkpoint in Git, a public image, or a public bucket.

The Worker exposes an internal artifact route that:

1. requires a long random bearer secret stored as a Worker secret;
2. accepts only the fixed checkpoint path;
3. streams the object from its R2 binding;
4. is not linked or advertised publicly.

The same secret is passed to the Container as a runtime environment variable. The boot fetcher downloads the object, computes SHA-256, and refuses to start the model unless it matches `artifacts/release/manifest.json`.

## Availability

The design targets an always-warm singleton. A Cron Trigger periodically calls `/readyz` through the Container binding. Cloudflare may still restart a host or replace an instance; therefore availability is best-effort, not a guarantee that the process never restarts. After any restart, the gateway becomes live first and readiness stays false until the checkpoint and model are verified.

Once production verification passes, Tailscale Funnel remains available only as a manual rollback origin. The laptop LaunchAgent is not part of the production request path.

## Security

- Only the Worker is public.
- The model service binds to `127.0.0.1` inside the container.
- R2 is private; checkpoint access requires a Worker secret.
- No token, credential, checkpoint, `.env`, or Wrangler state is committed.
- Existing request size, schema, origin, response-shape, concurrency, and global rate-limit checks remain active in `web.server`.
- The Worker preserves security headers and strips hop-by-hop headers.
- Release-manifest digest checks remain fail-closed: unavailable is preferred to serving unverified weights.

## Deployment flow

1. Upgrade the target Cloudflare account to Workers Paid.
2. Create the private R2 bucket and upload the pinned checkpoint.
3. Add the artifact bearer token as a Worker secret.
4. Add Container configuration, Durable Object migration, and Cron Trigger to `wrangler.jsonc`.
5. Add the Container controller and Cloudflare-specific boot/fetch files.
6. Use the GitHub-connected Workers Build to build and deploy, avoiding the local Wrangler account mismatch. The build environment must provide a Docker-compatible engine; otherwise publish a pre-built private image from CI and reference it in the Container config.
7. Verify the Container directly through the Worker, then remove `VEYA_ORIGIN` and stop relying on Tailscale.

## Failure behavior

- Checkpoint missing or digest mismatch: gateway liveness can answer, readiness stays 503, inference stays unavailable.
- Model load failure: container exits so Cloudflare can restart it; logs retain the failure reason.
- Container cold start: Worker returns a clear 503 until readiness succeeds rather than forwarding partial responses.
- R2 unavailable: boot fails closed and retries only through a new container start.
- Cloudflare rollout failure: retain the last working Worker deployment and Tailscale origin as rollback until cloud-hosted inference is proven.

## Verification

Before cutover:

- assemble the deployment tree and verify every manifest-pinned file;
- build the Linux image and confirm CPU-only PyTorch with no CUDA packages;
- run unit and packaging tests;
- verify `/healthz`, `/readyz`, `/api/health`, `/api/version`, and `/api/benchmark`;
- run real two-candidate `packed` inference;
- run >19-candidate `shortlist_k8_packed_rerank` inference;
- confirm response probabilities sum to 1 and release identifiers match the manifest;
- stop the local model, gateway, and Tailscale Funnel, then repeat all public checks through `https://veya.veya-8d0.workers.dev`;
- verify R2 objects and secrets are not publicly readable.

## Rollback

Keep the current Tailscale configuration intact until the Container passes full public verification. Rollback is a single Worker configuration change restoring `VEYA_ORIGIN` to the stable Tailscale Funnel, followed by the GitHub-connected deployment. The checkpoint digest gate prevents either path from silently serving different weights.
