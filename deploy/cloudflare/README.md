# Cloudflare Container

The default Wrangler name is `veya-staging` on Cloudflare account `veya`
(`8d0a6a64de03f4c9b14184398e75aa6e`). That keeps `wrangler deploy` from
replacing the live production Worker `veya`.

Scheduling is Cloudflare's default policy. Every instance uses the same image
and `instance_type` `standard-2` (1 vCPU, 6 GiB memory, 12 GB disk), with
`max_instances` 1. The container has no public internet. It fetches
`http://veya-r2.internal/releases/0.3.5-audit-baseline.1/model.safetensors`,
checks the release SHA-256, and only then starts the model service. `/readyz`
stays 503 until that service reports the audited release.

The checkpoint bucket binding is `VEYA_ARTIFACTS` /
`veya-model-artifacts`. The container does not receive an R2 access key.
Inference requires Worker secret `VEYA_EDGE_TOKEN` (`wrangler secret put`).
Do not put the token in `wrangler.jsonc`.

This configuration is not deployed. The account audit found no R2 bucket and
no container application, Wrangler on this machine is authenticated to a
different account, and Docker is not installed.
