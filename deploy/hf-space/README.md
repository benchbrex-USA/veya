---
title: Veya Serve
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Audited serving stack for the Veya System-1 decision model, behind the origin the veya.dev website proxies to.
pinned: false
---

# Veya Serve

This Space runs the Veya serving stack exactly as it is audited in the source
repository: a bounded HTTP gateway in front of a model service that refuses to
start until every SHA-256 pin in the release manifest verifies.

It is an **origin**, not a demo. The public website at
`https://veya.veya-8d0.workers.dev` proxies `/api/*` here; the same container
also serves the static site, so the Space can be browsed directly.

## What is served

| Field | Value |
| --- | --- |
| Release | `0.3.5-audit-baseline.1` |
| Architecture | `veya-shortlist-k8-packed-rerank-v1` |
| Served scorer | `base_decision_head` |
| Supported languages | `en` |
| Supported primitives | `choice`, `noul`, `score-categorical` |
| Encoder | ModernBERT-large, CPU only |

Status per the release manifest: English `choice` is a **RESEARCH RELEASE**;
English `noul`, English `score-categorical`, and every multilingual route are
**RESEARCH ONLY** and are not offered for production decisions. Read
`artifacts/release/manifest.json` → `known_limitations` before drawing
conclusions from any output.

## Health and readiness

| Endpoint | Meaning |
| --- | --- |
| `GET /healthz` | Liveness. The gateway process is up. Available within seconds of boot. |
| `GET /readyz` | The audited release is latched: model, shortlist, selective policy, and route calibration all verified. **503 until then.** |
| `GET /startupz` | Which component, if any, failed the startup gate. |
| `GET /api/version` | Release bindings the gateway observed from the model service. |
| `GET /api/health` | Proxied model-service health. |
| `GET /api/benchmark` | Blind CLINC150 validation result, read from `artifacts/release/blind_clinc150_val_result.json`. 503 `benchmark_unavailable` if that file is absent. |
| `POST /api/infer` | The only inference entry point. |

A cold start takes roughly three to four minutes: the 804 MB checkpoint is
downloaded and hash-verified (~11 s), then ModernBERT-large is loaded on CPU
(~79 s). `/healthz` answers long before `/readyz` does, which is intentional —
treat 503 from `/readyz` as "still booting", not as a failure, unless
`/startupz` names a component.

## Inference contract

`POST /api/infer` takes `Content-Type: application/json` and a body of at most
32 KB. `web/contracts.py` is the authority; the bounds are:

| Field | Bound |
| --- | --- |
| `primitive` | `choice`, `noul`, `score`, or `systemone` |
| `state` | Nonempty string, object, or list; at most 12 KB serialized |
| `instructions` | Nonempty text, at most 500 characters |
| `criteria` (choice) | Object of 2–100 `id → description` pairs; id ≤ 80 chars, description ≤ 240 chars, ids unique after NFKC casefolding |
| `criteria` (score) | List of 2–10 unique ordered labels, each ≤ 100 characters |
| `criteria` (noul) | Must be omitted |
| `questions` (systemone) | 1–8 questions, each `{type, instructions, criteria?}` |

Any other field is rejected with 400 `Unexpected request field.` Duplicate JSON
keys and non-finite numbers are rejected at parse time. Responses are
whitelisted on the way out as well: a payload whose probabilities do not sum to
one within 0.015, or whose choice answer is absent from its own distribution,
becomes 502 rather than reaching the caller.

```json
{
  "primitive": "choice",
  "state": "I was charged twice for the same subscription this month.",
  "instructions": "Which label best fits the text?",
  "criteria": {
    "duplicate_charge": "Dispute a charge that was taken more than once",
    "cancel_subscription": "End the recurring subscription",
    "update_payment": "Change the card on file",
    "request_refund": "Ask for money back",
    "human_agent": "Escalate to a person"
  }
}
```

Above 19 candidates the request takes the shortlist route: the encoder projects
all candidates, keeps the best 8, and the decision head reranks those. That
route reports `automation: "defer"` because the selective policy ships with
`operating_epsilon: "disabled"`, so no abstention threshold is active — the
shortlist answer is returned, but it is not cleared for unattended action.

Every `/api/infer` response carries a `request_id`, and the status codes are
fixed:

| Status | `code` | Cause |
| --- | --- | --- |
| 400 | `invalid_input` | Contract violation, or an `Origin` header outside the allowlist |
| 413 | `request_too_large` | Body over 32 KB |
| 429 | `rate_limited` | Rate budget exhausted; `retry_after` is included |
| 502 | `invalid_upstream_response` | The model answered, but not in a shape the contract accepts |
| 503 | `model_unavailable` | The model service is down or still booting |
| 503 | `model_busy` | More than two inferences already in flight; `retry_after: 5` |
| 504 | `timeout` | Inference exceeded `VEYA_REQUEST_TIMEOUT` |

### Rate limiting behind a proxy — read this

The limiter is a sliding window of **12 cost units per 60 seconds**, keyed on
`self.client_address[0]`, the TCP peer address. It deliberately ignores
`X-Forwarded-For` and every other forwarding header, because those are
client-controlled and trusting them would let any caller pick their own bucket.
That is the right choice for a directly-exposed server and the wrong one behind
an ingress: inside this container every external request arrives from the same
platform address, so the window is effectively **one shared budget for the whole
Space** rather than one per visitor.

Two consequences follow. A burst of traffic from anywhere — including a crawler
that finds the public Space URL — spends the budget the website needs, and
legitimate requests start returning 429. And the budget is small enough that
this happens without anyone being malicious.

`web/README.md` records the same limitation upstream: the limiter is
process-local, and public multi-instance deployment needs a shared limiter at
the ingress. The Cloudflare Worker in front of this Space is that ingress, and
it is where real client IPs are available. Until per-client limiting is enforced
there, treat 429 from this origin as expected under load rather than as a fault.

## Why the checkpoint is not in this repository

`models/production/model.safetensors` (804 MB) lives outside the image.
`space/fetch_checkpoint.py` downloads that single file at boot and asserts its
digest against `manifest.base_checkpoint_hash` before the model service is
allowed to start. On Hugging Face it uses a pinned private-repo revision. On
Cloudflare it uses `VEYA_CHECKPOINT_URL` with `VEYA_CHECKPOINT_TOKEN`. Either
way the weights are re-verified on every boot rather than trusted from an
image layer.

If the digest does not match, the fetch fails, the model service is never
started, and the Space exits so the runtime restarts it. `/readyz` stays 503 in
that state.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `VEYA_PUBLIC_ORIGIN` | Canonical HTTPS origin. Required whenever the gateway binds a non-loopback address; defaults to `https://$SPACE_HOST`. |
| `VEYA_ALLOWED_ORIGINS` | Comma-separated CORS allowlist. Browsers only — it is not an access control. |
| `VEYA_DEVICE` | `cpu`. No accelerator is attached on the free tier. |
| `VEYA_MODEL_PORT` | Loopback port for the model service (default `8080`). Never published. |
| `VEYA_HF_TOKEN` | Read access to the private model repository. Falls back to the runtime-injected `HF_TOKEN`. Used only when `VEYA_CHECKPOINT_URL` is unset. |
| `VEYA_CHECKPOINT_URL` | Authenticated checkpoint URL. When set, this replaces the Hugging Face download. |
| `VEYA_CHECKPOINT_TOKEN` | Bearer credential for `VEYA_CHECKPOINT_URL`. Required with that URL. Never log or commit it. |
| `VEYA_REQUEST_TIMEOUT` | Upstream inference timeout in seconds (default `30`). |

The model service binds `127.0.0.1` inside the container and is not reachable
from outside it. Only the gateway port is exposed.

## Exposure notes

This Space is public, so the gateway is directly reachable, and CORS is a
browser-enforced convention rather than a boundary — it stops a page on another
origin from reading a response, not a client from sending a request. There is no
inbound bearer authentication on the gateway. What does bound the blast radius:
every request is capped by the contract above, at most two inferences run at
once, the rate limiter refuses the rest, and the model service itself listens on
loopback inside the container and is never published. The shared rate budget
described above is the weakest of these, not the strongest.

If this origin needs to be closed to direct traffic, the follow-up is a private
Space plus an `Authorization` header injected by the proxy — see
`deploy/hf-space/DEPLOY.md` in the source repository. That path is documented but
not implemented, and Hugging Face's support for authenticated Space-to-Space
calls from a third-party proxy has not been verified here.

## Cloudflare Worker

The active edge is `worker/veya-container.mjs`. It forwards to one Cloudflare
Container named `veya-core`. It does not use an external origin. The image is
`deploy/cloudflare/Dockerfile`. Historical notes about a laptop tunnel are not
the deployment path.

`/healthz` answers as soon as the gateway binds. `/readyz` stays 503 while
ModernBERT-large loads on CPU. That 503 means the audited release is not
latched yet. `/startupz` names a component only after a startup gate fails.

## Reproducing this Space

The container is built from `deploy/hf-space/` in the source repository. That
directory contains the Dockerfile, the two Space-only entry points under
`space/`, and `assemble.py`, which stages the upload tree and re-verifies every
manifest-pinned digest against the staged bytes. No pinned file is edited
during deployment; if a digest would change, the release has to be re-minted
under the repository's change protocol first.

License: Apache-2.0, matching the source repository.
