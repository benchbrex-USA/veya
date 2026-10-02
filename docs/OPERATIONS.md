# Veya Operations Guide

Deployment, rollback, configuration, artifact updates, and health monitoring
for the audited serving stack. Incident response is in
[`docs/RUNBOOK.md`](RUNBOOK.md); scope and limits are in
[`docs/PRODUCTION_SCOPE.md`](PRODUCTION_SCOPE.md).

## Architecture

One container (or pod) runs both serving processes, exactly as the
`Dockerfile` CMD defines:

- **Model service** (`web/model_server.py` → `veya/server.py`): loads and
  verifies `artifacts/release/manifest.json` via
  `veya.release_bundle.load_release_manifest`, then binds `127.0.0.1:8080`
  inside the container only. Never publish or route this port; do not expose
  the model service directly.
- **Gateway** (`web/server.py`): same-origin website + `POST /api/infer`,
  listens on port 8000, and is the only exposed surface. It never loads
  weights; every inference is a real HTTP hop to the model service.

Cold start is ~34 s (manifest verification + 2.2 GB checkpoint load) and can
exceed a minute under CPU load — every deployment path below budgets 120 s
before declaring the container unhealthy.

## Deployment — Docker Compose

`docker-compose.yml` builds the image from the repository root. Before any
public deployment, set `VEYA_PUBLIC_ORIGIN` to the real canonical HTTPS
origin: the gateway refuses a published (non-loopback) bind without it
(`web/server.py`, "Public binding requires VEYA_PUBLIC_ORIGIN").

```bash
docker compose up -d
curl http://localhost:8000/readyz  # Wait for 200
```

`/readyz` returns 200 only after the model service verified and latched the
audited release (all manifest hashes, checkpoint, shortlist, calibration,
policy). Expect roughly 34 s of cold start before it flips; the compose
healthcheck allows a 120 s start period.

Notes:

- Both `models/production` and `artifacts/release` are bind-mounted read-only
  into the container, so a re-minted manifest and evidence can be swapped in
  without rebuilding the image (see "Model Update Procedure").
- The service is named `veya-server`; logs: `docker compose logs -f veya-server`.
- Resource envelope: 4 GB reserved / 6 GB limit per container (serving RSS is
  ~2030 MB plus checkpoint-load headroom).

## Deployment — Kubernetes

The manifests in `k8s/` (applied as a kustomization) deploy the same
two-process image: 2 replicas by default, ClusterIP Service (port 80 →
container 8000), HPA 2–8 replicas on 70% CPU, and a ConfigMap
(`veya-config`) supplying `public-origin` and `shadow-mode`. TLS terminates
upstream (Ingress / load balancer) at the origin named by
`VEYA_PUBLIC_ORIGIN`; the Service never routes the in-container model service
port.

```bash
kubectl apply -k k8s/
kubectl rollout status deployment/veya
```

Rolling updates are safe by construction: `maxUnavailable: 0` and
`maxSurge: 1` mean a new pod must pass `/readyz` before any old pod retires.
Probe wiring (`k8s/deployment.yaml`):

- `startupProbe`: `/startupz` every 5 s, 24 failures allowed (120 s budget) —
  covers the ~34 s cold start.
- `readinessProbe`: `/readyz` every 5 s.
- `livenessProbe`: `/healthz` every 10 s — process-level only.

Verify after rollout:

```bash
kubectl get pods -l app.kubernetes.io/name=veya
curl -s http://<ingress>/readyz | python3 -m json.tool
```

## Rollback

The release is hash-pinned: a rollback is simply redeploying a known-good
tree. Nothing is mutated in place at runtime.

**Docker Compose** — the compose file builds from the local working tree, so
check out the known-good release tag (or point `docker-compose.yml` at the
previously published image tag) first:

```bash
git checkout <known-good-release-tag>
docker compose down && docker compose up -d
curl http://localhost:8000/readyz  # Wait for 200
```

**Kubernetes** — undo the rollout to the previous ReplicaSet:

```bash
kubectl rollout undo deployment/veya
kubectl rollout status deployment/veya
```

## Configuration Reference

Environment variables consumed by the serving stack. Defaults are the code
defaults in `web/server.py` / `web/model_server.py`; the Docker image and
`docker-compose.yml` override several of them.

| Variable | Default | Description |
|---|---|---|
| `VEYA_WEB_HOST` | `127.0.0.1` (loopback) | Gateway bind address. Image and compose set `0.0.0.0`; a non-loopback bind requires `VEYA_PUBLIC_ORIGIN` or the gateway refuses to start. |
| `VEYA_WEB_PORT` | `3000` (code) / `8000` (image, compose, K8s) | Gateway listen port. |
| `VEYA_PUBLIC_ORIGIN` | unset | Canonical HTTPS origin (no path/query/fragment/userinfo), validated at gateway import time. Required for any public bind; drives canonical/OG URLs, `/robots.txt`, `/sitemap.xml`, and HSTS. K8s reads it from ConfigMap `veya-config` key `public-origin`. |
| `VEYA_SHADOW_MODE` | unset (`""`) | `"1"` or `"true"` flags telemetry entries as shadow (`"shadow": true`, `X-Veya-Shadow: true` response header); `"0"` serves normal traffic. |
| `VEYA_REQUEST_TIMEOUT` | `30` (seconds) | Per-request inference ceiling for the upstream hop. K8s sets `30` explicitly. |
| `VEYA_MODEL_URL` | `http://127.0.0.1:8080` | Model service URL the gateway forwards to. |
| `VEYA_MODEL_PORT` | `8080` | Model service bind port (always loopback). |
| `VEYA_MODEL_DIR` | `/app/models/production` (image) | Base checkpoint directory named by the manifest. |
| `VEYA_DEVICE` | `cpu` | Torch device for the model service. |
| `VEYA_LOG_LEVEL` | `INFO` | Gateway log level. |
| `VEYA_MODEL_TOKEN` | unset | Optional bearer token the gateway sends to the model service; never included in browser assets or responses. |
| `VEYA_ALLOWED_ORIGINS` | unset | Comma-separated extra browser origins accepted on `POST /api/infer` (same-origin policy otherwise). |

Fixed gateway limits (not environment-configurable): `MAX_INFLIGHT = 2`
concurrent upstream inferences, 12 cost units per 60 s per client IP, 32 KB
request cap, 512 KB upstream response cap, 30 s shutdown drain
(`SHUTDOWN_DRAIN_S`). See [`docs/CONCURRENCY.md`](CONCURRENCY.md).

## Model Update Procedure

Any change to a manifest-pinned file (checkpoint, tokenizer, router,
shortlist, calibration, selective policy, `veya/*.py` runtime files) makes
`load_release_manifest` raise at startup: the server never becomes ready.
The update procedure is therefore:

1. **Update the model files** under `models/production/` (checkpoint
   `model.safetensors`, tokenizer, `choice_router.json`, `shortlist_k8.pt`,
   calibration artifacts).
2. **Re-mint the manifest** per [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md).
   Recompute the checkpoint record with `veya/manifest.py`'s
   `build_manifest(model_dir, git_commit)`, then update every affected SHA-256
   pin in `artifacts/release/manifest.json` (`base_checkpoint_hash`,
   `model_config_hash`, `tokenizer_files`, `routing.choice_router_hash`,
   `choice.shortlist_hash`, calibration hashes, `runtime_files` if any
   `veya/*.py` changed). Regenerate the release metadata record:
   ```bash
   python -m research.veyabench.release_metadata
   ```
3. **Verify before shipping** (expect ~34 s for the full load):
   ```bash
   python -m pytest tests/test_artifact_integrity.py
   python -c "from veya.release_bundle import load_release_manifest; load_release_manifest('artifacts/release/manifest.json', root='.')"
   ```
4. **Rebuild the image** (`docker compose build`) — or, with compose, mount
   the updated `models/production` and `artifacts/release` (both are read-only
   bind mounts by default) and restart.
5. **Deploy** and confirm `/readyz` returns 200 and the `X-Veya-Model-Hash`
   header matches the new `base_checkpoint_hash`.

## Selective Policy Regeneration

1. **Run selective policy fitting** on a hashed validation split (thresholds
   selected via `veya/selective.py`; the policy must keep
   `chosen_on: "validation"` and its `validation_hash` — `load_policy`
   rejects anything else).
2. **Update `artifacts/release/selective_policy.json`**. The policy is bound
   to the exact `route`, `checkpoint_hash`, `shortlist_sha256`, and
   `scorer_id`; `load_policy` rejects artifact mixing. Note that
   `operating_epsilon` currently stays `"disabled"` — see
   [`docs/PRODUCTION_SCOPE.md`](PRODUCTION_SCOPE.md) for the activation path
   (blind evaluation on prospective target-domain data is required before any
   automation is enabled).
3. **Re-mint the manifest** (`selective_policy_hash` changes) per the
   procedure above.
4. **Deploy** and confirm the `X-Veya-Policy-Hash` header on `/readyz`.

## Calibration Update

1. **Re-fit the temperature**: per-cardinality packed buckets in
   `models/production/calibration.json`; the hierarchical route temperature in
   `models/production/calibration_shortlist_k8.json` (manifest pins
   `hierarchical_temperature: 1.971127355603397` and both artifact hashes).
2. **Update the calibration artifacts** on disk.
3. **Re-mint the manifest** (`packed_choice_hash`,
   `hierarchical_choice_hash`, `hierarchical_temperature`) per the model
   update procedure.
4. **Rebuild and deploy**, then verify `/readyz`.

## Health Monitoring

| Probe | Purpose | Behavior |
|---|---|---|
| `/healthz` | Liveness | `200 {"status": "alive"}` whenever the gateway process answers. Never consults the model service; still answers 200 while draining on shutdown. |
| `/readyz` | Readiness | `200` only after the audited release is confirmed loaded. Body: `{"status": "ready", "components": {...}}` with per-component booleans (`model_loaded`, `policy_loaded`, `calibration_loaded`, `shortlist_loaded`, `hashes_valid`, `startup_failed`). Ready responses carry `X-Veya-Model-Hash` and `X-Veya-Policy-Hash`. Stays `503` for the life of the process after a manifest mismatch (fail-closed, sticky). |
| `/startupz` | Startup | Same gate as readiness; Kubernetes gets a longer budget (120 s) for the ~34 s cold start. |

The gateway also proxies the model service at `/api/health` and
`/api/version` (bounded release/checkpoint metadata), and publishes benchmark
evidence at `/api/benchmark`.

Telemetry: one JSON line per request (logger `veya.web`) with `request_id`,
`primitive`, `candidate_count`, `route`, `deferred`, `status`, `latency_ms`,
`model_version`, `error_code`, `timestamp`, `confidence`, `max_probability`,
`automation`, and `config_hash`. Raw request bodies and client IPs are never
logged (rejected requests go to the `veya.security_audit` logger with a
truncated SHA-256 of the client IP). Rejected requests (`>=400`) are audit
logged at WARNING with status, hashed client, path, and error code.

## Local Verification (no Docker)

From the repository root, in separate terminals
(`web/README.md`):

```bash
VEYA_DEVICE=cpu python3 -m web.model_server   # verifies manifest, binds 127.0.0.1:8080
python3 -m web.server                          # gateway on 127.0.0.1:3000 by default
```

Gateway and packaging checks:

```bash
python3 -m pytest -q tests/test_web_gateway.py tests/test_server_security.py
python3 -m pytest -q tests/test_artifact_integrity.py
python3 -m ruff check web tests/test_web_gateway.py
```

## Selective Automation Activation

Selective automation is **disabled**: `operating_epsilon: "disabled"` in
`artifacts/release/selective_policy.json` (SHA-256
`80e7be582c9c890cf73da701cbd86a71c6d908331186ee68f1021763011cd620`), manifest
SHA-256
`cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de`, and
`automation.status: "DISABLED"` in the manifest. The machine-readable gate
procedure is `artifacts/release/activation_procedure.json`
(`veya-selective-activation-procedure-v1`, status `BLOCKED_EXTERNAL`);
`tests/test_activation_procedure.py` pins its recorded hashes and disabled
state against the actual artifacts, so any drift fails the suite. This section
documents the conditional procedure only — it does not enable anything.

> **DO NOT enable selective automation before the G6/G8 prospective blind
> gates PASS on target-domain labeled data.** The validation-selected 5%
> threshold observed 6.99% accepted risk on development-exposed reporting data,
> so an enabled policy is justified only by a prospective blind PASS
> ([`docs/PROSPECTIVE_BLIND_WORKFLOW.md`](PROSPECTIVE_BLIND_WORKFLOW.md)). The
> blind runner refuses to start unless the manifest automation status is
> `DISABLED` (`research/veyabench/run_prospective_blind.py`,
> `_verify_manifest_links`), so the run always precedes activation.

**Prerequisites** — every gate below must be `PASS` in
`artifacts/release/prospective_blind_result.json` (exit 0 from
`.venv/bin/python -m research.veyabench.run_prospective_blind`):

| Gate | Requirement |
|---|---|
| G6 | Freeze record (`veya-freeze-record-v1`) validated against the pinned re-minted manifest; automation stayed `DISABLED` during the run. |
| G8 | `verdict.all_primary_gates_pass` is true over the supplied target-domain labeled dataset. |
| P1 data validity | Contamination screens flag ≤ tolerance (default 0) blind texts vs the controlled training side. |
| P2 selective safety | Clean-slice accepted-risk UCB95 ≤ frozen epsilon (G1). |
| P3 useful coverage | Clean-slice coverage ≥ frozen c_min (0.80). |
| P4 shift robustness | G2 per-shift risk ≤ epsilon, G3 shift coverage ≥ 0.95 × clean, G4 no slice below the collapse floors (coverage 0.05, accuracy 0.30). |
| P5 calibration transfer | Frozen temperature (1.971127355603397) applied, never refit. |
| P6 operational viability | Caller-supplied operational measurements with no failures. |

**1. Verify the prospective verdict** (repository root):

```bash
.venv/bin/python -c "
import json
result = json.load(open('artifacts/release/prospective_blind_result.json'))
assert result['format'] == 'veya-prospective-blind-result-v1'
assert result['status'] == 'PASS', result.get('summary')
verdict = result['verdict']
assert verdict['all_primary_gates_pass'] is True
assert len(verdict['primary_gates']) == 6 and all(verdict['primary_gates'].values())
print('verified:', result['summary'])
print('epsilon:', verdict['sample_size_check']['epsilon'])
"
```

The printed epsilon is the preregistered epsilon the approved policy must
select in step 2 (the frozen run config template pins `"0.10"`).

**2. Mint the approved policy.** Edit
`artifacts/release/selective_policy.json` changing **only** `operating_epsilon`
— from `"disabled"` to the exact preregistered epsilon that passed (`"0.10"`,
threshold `0.896261827143334`). Do not change any threshold value. The policy
stays bound to `route` `shortlist_k8_packed_rerank`, `scorer_id`
`base_decision_head`, `score` `max_probability`, `checkpoint_hash`
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`,
`shortlist_sha256`
`b974fd75de2f3f8b93a69501d17035d59c05d165ea693dc1340f6df78f8562e1`, and
`chosen_on` `"validation"` with its `validation_hash`:
`veya.selective.load_policy` fails closed on artifact mixing, a missing
`validation_hash`, or an `operating_epsilon` not present in `thresholds`.

**3. Re-mint the manifest.** Update `selective_policy_hash` to the new policy
SHA-256 and set the automation record to the enabled state citing
`prospective_blind_result.json` as evidence; preserve every other pin
byte-identically (`base_checkpoint_hash`, `model_config_hash`,
`tokenizer_files`, `runtime_files`, `choice.shortlist_hash`,
`routing.choice_router_hash`, calibration hashes, data lineage and blind
evaluation pins, `git_commit`). There is no dedicated v2 manifest builder —
surgically edit only the affected fields. Then regenerate the second-order
pins **in this order**, and refresh `current_state` in
`artifacts/release/activation_procedure.json` in the same change:

```bash
.venv/bin/python -m research.veyabench.run_runtime_audit   # ~34-90 s CPU, ~2 GB RSS
.venv/bin/python -m research.veyabench.release_metadata    # AFTER the runtime audit
```

`runtime_audit.json` must record the new `manifest_sha256`, and
`release_metadata.json` must mirror the re-minted manifest and policy
(`tests/test_artifact_integrity.py::test_runtime_audit_matches_manifest` and
`tests/test_release_metadata.py` assert both; `tests/test_activation_procedure.py`
fails until the recorded hashes are refreshed).

**4. Verify fail-closed loading locally** (expect ~34 s for the full load):

```bash
.venv/bin/python -m pytest tests/test_selective_extra.py tests/test_artifact_integrity.py tests/test_activation_procedure.py -v
.venv/bin/python -c "from veya.release_bundle import load_release_manifest; load_release_manifest('artifacts/release/manifest.json', root='.')"
```

**5. Run the release-candidate gate** — the `release-candidate-gate` job in
`.github/workflows/release-gate.yml`, triggered by pushing a `v*` tag:

```bash
git tag <version> && git push origin <version>
```

The job runs the full test suite (`pytest --tb=short -q`), artifact integrity
validation, the license inventory check, freeze record validation, and a
Docker build.

**6. Run the production-evidence gate** — the `production-evidence-gate` job
in the same workflow, manual dispatch only (it never runs on a tag push
because it needs prospective evaluation data):

```bash
gh workflow run release-gate.yml --ref <version-tag> -f run_production_gate=true
```

**Rollback.** Trigger: the observed accepted-risk UCB exceeds the approved
epsilon, or a drift alert fires (`drift_triggers` `risk_breach`,
`coverage_drop`, `distribution_shift`, or `config_mismatch` — see
[`docs/RUNBOOK.md`](RUNBOOK.md)).

1. Set `operating_epsilon` back to `"disabled"` in
   `artifacts/release/selective_policy.json`; thresholds and all bindings stay
   untouched. Never re-select a threshold on the data that showed the breach.
2. Re-mint the manifest and regenerate the second-order pins exactly as in
   step 3 above.
3. Deploy the previous known-good image:
   `git checkout <known-good-release-tag> && docker compose down && docker compose up -d`
   (or `kubectl rollout undo deployment/veya`).
4. Confirm `/readyz` returns 200 and the `X-Veya-Policy-Hash` header matches
   the disabled policy hash.
