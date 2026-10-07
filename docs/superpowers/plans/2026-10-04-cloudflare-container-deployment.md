# Cloudflare Container Deployment Implementation Plan

> Historical implementation plan from 2026-10-04. It is not the operator runbook.
> The active Worker configuration is `wrangler.jsonc`.

# Cloudflare Container Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the exact audited Veya release in a private Cloudflare Container behind the existing `https://veya.veya-8d0.workers.dev` Worker, with no laptop or Tailscale dependency.

**Architecture:** The public Worker routes all application traffic to one Durable Object-managed Container. The Container starts the existing Python gateway first, retrieves the private checkpoint from an authenticated Worker route backed by R2, verifies the manifest-pinned SHA-256, then starts the loopback-only model service. A Cron Trigger periodically wakes the singleton so it remains warm and recovers after platform restarts.

**Tech Stack:** Cloudflare Workers, Cloudflare Containers, Durable Objects, R2, Wrangler 4.147+, JavaScript modules, Python 3.11, CPU-only PyTorch 2.14.0, pytest, Node `node:test`.

**Spec:** `docs/superpowers/specs/2026-10-04-cloudflare-container-deployment-design.md`

## Global Constraints

- Target account is Cloudflare account `8d0a6a64de03f4c9b14184398e75aa6e`; local Wrangler currently authenticates to another account, so production resources and deploy credentials must be created for the target account.
- Cloudflare Containers require Workers Paid; user approved the paid plan and an always-on workload.
- Use one singleton Container with 1 vCPU, 3,072 MiB memory, and 8,000 MB disk.
- Keep `web.model_server` bound to `127.0.0.1:8080`; only `web.server` is exposed on container port `7860`.
- Keep `model.safetensors` private: never commit it, place it in a public bucket, or log any artifact token.
- Continue enforcing `artifacts/release/manifest.json`; a missing or mismatched checkpoint must fail closed.
- Preserve the public URL `https://veya.veya-8d0.workers.dev` and existing HTTP contracts.
- Keep Tailscale Funnel as rollback until public Container inference passes with the laptop serving processes stopped.
- Do not change global git configuration; use the repository's noreply identity per command if a commit requires it.

---

### Task 1: Add authenticated checkpoint download support

**Files:**
- Modify: `deploy/hf-space/space/fetch_checkpoint.py:23-129`
- Create: `tests/test_checkpoint_fetch.py`

**Interfaces:**
- Consumes: `VEYA_CHECKPOINT_URL`, `VEYA_CHECKPOINT_TOKEN`, existing `VEYA_HF_TOKEN` fallback, and `manifest["base_checkpoint_hash"]`.
- Produces: `download_from_url(url: str, credential: str, target: Path) -> Path`; `main() -> int` chooses URL download when `VEYA_CHECKPOINT_URL` is set, otherwise preserves the Hugging Face path.

- [ ] **Step 1: Write failing URL-download tests**

Create `tests/test_checkpoint_fetch.py` with an in-process `ThreadingHTTPServer`. Load `deploy/hf-space/space/fetch_checkpoint.py` through `importlib.util.spec_from_file_location`. Cover:

```python
def test_download_from_url_sends_bearer_and_atomically_writes(tmp_path):
    target = tmp_path / "model.safetensors"
    downloaded = module.download_from_url(server_url, "secret-token", target)
    assert downloaded == target
    assert target.read_bytes() == CHECKPOINT_BYTES
    assert handler.authorization == "Bearer secret-token"
    assert not (tmp_path / "model.safetensors.part").exists()


def test_main_rejects_url_download_with_wrong_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MANIFEST", manifest_with_other_hash)
    monkeypatch.setenv("VEYA_CHECKPOINT_URL", server_url)
    monkeypatch.setenv("VEYA_CHECKPOINT_TOKEN", "secret-token")
    assert module.main() == 1
    assert not checkpoint_path.exists()
```

Also assert that the secret value never appears in captured stdout.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_checkpoint_fetch.py -v
```

Expected: FAIL because `download_from_url` and URL-source selection do not exist.

- [ ] **Step 3: Implement atomic authenticated download**

Add a downloader that writes `model.safetensors.part`, streams in `CHUNK` blocks, flushes and closes it, then uses `os.replace()` to publish the complete file. Build the request as:

```python
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {credential}"},
)
```

In `main()`, prefer `VEYA_CHECKPOINT_URL`. Require `VEYA_CHECKPOINT_TOKEN` for that source. Preserve the current Hugging Face branch unchanged when the URL is absent. Keep the existing post-download SHA-256 check and unlink-on-mismatch behavior.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_checkpoint_fetch.py tests/test_manifest.py tests/test_health_probes.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/hf-space/space/fetch_checkpoint.py tests/test_checkpoint_fetch.py
git commit -m "feat(deploy): support authenticated checkpoint downloads"
```

---

### Task 2: Add testable Worker routing helpers

**Files:**
- Create: `worker/veya-routing.mjs`
- Create: `tests/test_veya_routing.mjs`

**Interfaces:**
- Produces: `CHECKPOINT_PATH`, `CONTAINER_NAME`, `isArtifactRequest(url)`, `isAuthorizedArtifactRequest(request, secret)`, `rewriteForContainer(request)`, and `createEdgeHandler(dependencies)`.
- `createEdgeHandler` accepts injected `getContainerStub`, `getCheckpoint`, and `artifactSecret` functions/values so edge behavior is testable in Node without importing `cloudflare:workers`.
- Consumed by: `worker/veya-edge.mjs` in Task 3.

- [ ] **Step 1: Write failing routing tests**

Use Node's built-in test runner. Cover exact artifact-path matching, rejected missing/wrong bearer tokens, accepted matching tokens, preservation of method/body/query/origin, rewriting protocol/host to `http://container`, and removal of the `Host` header.

```javascript
test("only the fixed checkpoint path is internal", () => {
  assert.equal(isArtifactRequest(new URL("https://x/__artifacts/releases/0.3.5-audit-baseline.1/model.safetensors")), true);
  assert.equal(isArtifactRequest(new URL("https://x/__artifacts/other")), false);
});
```

- [ ] **Step 2: Run the test and verify failure**

```bash
node --test tests/test_veya_routing.mjs
```

Expected: FAIL because `worker/veya-routing.mjs` does not exist.

- [ ] **Step 3: Implement the pure helpers**

Use the fixed object path:

```javascript
export const CHECKPOINT_PATH = "/__artifacts/releases/0.3.5-audit-baseline.1/model.safetensors";
export const CONTAINER_NAME = "veya-production";
```

`rewriteForContainer` must create a new Request, set `url.protocol = "http:"`, set `url.host = "container"`, and delete `host`. It must not rewrite `Origin`. `createEdgeHandler` must implement edge-health, artifact, normal-routing, and scheduled-warm-up behavior exclusively through its injected dependencies; `worker/veya-edge.mjs` supplies the real Cloudflare bindings.

- [ ] **Step 4: Run the tests**

```bash
node --test tests/test_veya_routing.mjs
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add worker/veya-routing.mjs tests/test_veya_routing.mjs
git commit -m "test(edge): define Container routing boundaries"
```

---

### Task 3: Replace the external-origin proxy with a Container controller

**Files:**
- Modify: `worker/veya-edge.mjs:1-96`
- Modify: `tests/test_veya_routing.mjs`

**Interfaces:**
- Consumes: `env.VEYA_CONTAINER`, `env.VEYA_MODEL_ARTIFACTS`, `env.VEYA_ARTIFACT_TOKEN`, and helpers from `worker/veya-routing.mjs`.
- Produces: exported `VeyaContainer` Durable Object; public Worker `fetch`; scheduled warm-up handler.

- [ ] **Step 1: Extend failing tests for edge behavior**

Add assertions for:

- `/__edge_health` returns JSON with `backend: "container"`;
- the checkpoint route rejects missing/wrong authorization with 404, not 401, to avoid advertising the object;
- a missing R2 checkpoint returns 503;
- normal requests resolve the singleton Durable Object name `veya-production`;
- scheduled warm-up targets `/readyz` on the same singleton.

Use small dependency-injected fakes for R2 and the Durable Object namespace; do not start a real Container in unit tests.

- [ ] **Step 2: Run the tests and verify failure**

```bash
node --test tests/test_veya_routing.mjs
```

Expected: FAIL against the current `VEYA_ORIGIN` proxy.

- [ ] **Step 3: Implement `VeyaContainer` with the Durable Object Container API**

Use the current Cloudflare API pattern:

```javascript
import { DurableObject, scheduler } from "cloudflare:workers";

const PORT = 7860;
const INACTIVITY_TIMEOUT_MS = 10 * 60 * 1000;

export class VeyaContainer extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    if (ctx.container?.running) {
      void ctx.blockConcurrencyWhile(() =>
        ctx.container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS),
      );
    }
  }

  async startAndWaitForGateway() {
    const container = this.ctx.container;
    if (!container.running) {
      container.start({
        image: container.images.veya,
        instance: { vcpu: 1, memoryMib: 3072, diskMb: 8000 },
        enableInternet: true,
        env: {
          VEYA_CHECKPOINT_URL: "https://veya.veya-8d0.workers.dev" + CHECKPOINT_PATH,
          VEYA_CHECKPOINT_TOKEN: this.env.VEYA_ARTIFACT_TOKEN,
          VEYA_ALLOWED_ORIGINS: "https://veya.veya-8d0.workers.dev",
          VEYA_PUBLIC_ORIGIN: "https://veya.veya-8d0.workers.dev",
          VEYA_DEVICE: "cpu",
          VEYA_WEB_HOST: "0.0.0.0",
          VEYA_WEB_PORT: "7860",
        },
      });
    }
    await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);
    // Poll /healthz only; /readyz legitimately remains 503 during model loading.
  }
}
```

Poll `http://container/healthz` with bounded retries and `scheduler.wait()`, then proxy through `container.getTcpPort(PORT).fetch(rewriteForContainer(request))`. Concurrent requests must share one startup promise.

- [ ] **Step 4: Implement Worker fetch and scheduled handlers**

The top-level Worker must:

1. short-circuit `/__edge_health`;
2. serve only `CHECKPOINT_PATH` from `env.VEYA_MODEL_ARTIFACTS` after bearer validation;
3. route every other request to `env.VEYA_CONTAINER.getByName(CONTAINER_NAME)`;
4. convert Container startup failures to JSON 503 without exposing stack traces;
5. issue a `/readyz` request from the scheduled handler and log readiness failures.

- [ ] **Step 5: Run tests and static deployment validation**

```bash
node --test tests/test_veya_routing.mjs
npx wrangler deploy --dry-run --outdir /tmp/veya-worker-dry-run
```

Expected: tests pass; Wrangler bundles the Worker and recognizes the Durable Object export.

- [ ] **Step 6: Commit**

```bash
git add worker/veya-edge.mjs worker/veya-routing.mjs tests/test_veya_routing.mjs
git commit -m "feat(edge): route Veya through a Cloudflare Container"
```

---

### Task 4: Configure the Container, R2, build, and warm-up schedule

**Files:**
- Modify: `wrangler.jsonc:1-21`
- Modify: `.gitignore:53-70`

**Interfaces:**
- Produces bindings `VEYA_CONTAINER`, `VEYA_MODEL_ARTIFACTS`, and required secret `VEYA_ARTIFACT_TOKEN`.
- Consumes `build/hf-space/Dockerfile`, generated by `python3 deploy/hf-space/assemble.py`.

- [ ] **Step 1: Add a config validation test**

Create `tests/test_cloudflare_config.py` that strips JSONC comments, parses `wrangler.jsonc`, and asserts:

```python
assert config["containers"][0]["scheduling_policy"] == "durable_object"
assert "max_instances" not in config["containers"][0]  # unsupported by this policy
assert config["r2_buckets"][0]["bucket_name"] == "veya-model-artifacts"
assert config["triggers"]["crons"] == ["* * * * *"]
assert "VEYA_ORIGIN" not in config.get("vars", {})
```

Also assert the named image uses `build/hf-space/Dockerfile` with `build/hf-space` as context and the `VeyaContainer` export attaches to the configured container application.

- [ ] **Step 2: Run the config test and verify failure**

```bash
.venv/bin/python -m pytest tests/test_cloudflare_config.py -v
```

Expected: FAIL because the config still contains `VEYA_ORIGIN` and has no Container bindings.

- [ ] **Step 3: Replace `wrangler.jsonc` with Container configuration**

Configure:

```jsonc
{
  "name": "veya",
  "main": "worker/veya-edge.mjs",
  "compatibility_date": "2026-10-04",
  "build": { "command": "python3 deploy/hf-space/assemble.py" },
  "containers": [{
    "name": "veya-runtime",
    "class_name": "VeyaContainer",
    "scheduling_policy": "durable_object",
    "images": {
      "veya": {
        "dockerfile": "./build/hf-space/Dockerfile",
        "build_context": "./build/hf-space"
      }
    },
    "observability": { "enabled": true }
  }],
  "durable_objects": {
    "bindings": [{ "name": "VEYA_CONTAINER", "class_name": "VeyaContainer" }]
  },
  "exports": {
    "VeyaContainer": {
      "type": "durable-object",
      "storage": "sqlite",
      "container": "veya-runtime"
    }
  },
  "r2_buckets": [{
    "binding": "VEYA_MODEL_ARTIFACTS",
    "bucket_name": "veya-model-artifacts"
  }],
  "secrets": { "required": ["VEYA_ARTIFACT_TOKEN"] },
  "triggers": { "crons": ["* * * * *"] },
  "observability": { "enabled": true }
}
```

Do not retain `VEYA_ORIGIN` in production config.

- [ ] **Step 4: Protect local secret files**

Add `.dev.vars*` to `.gitignore`; `.env*` is already ignored. Do not create or commit a development secret file.

- [ ] **Step 5: Validate staging and config**

```bash
python3 deploy/hf-space/assemble.py
python3 deploy/hf-space/assemble.py --check
.venv/bin/python -m pytest tests/test_cloudflare_config.py -v
npx wrangler deploy --dry-run --outdir /tmp/veya-worker-dry-run
```

Expected: 67 staged files, all pinned digests verified, checkpoint absent, config test passes, dry-run succeeds.

- [ ] **Step 6: Commit**

```bash
git add wrangler.jsonc .gitignore tests/test_cloudflare_config.py
git commit -m "build(deploy): configure Veya Cloudflare Container"
```

---

### Task 5: Add deployment regression coverage

**Files:**
- Modify: `.github/workflows/ci.yml:123-158`
- Modify: `deploy/hf-space/README.md`

**Interfaces:**
- Consumes: Task 1 checkpoint tests, Task 2 Worker tests, Task 4 config test.
- Produces: CI gate preventing broken Container configuration or accidental checkpoint publication.

- [ ] **Step 1: Add deployment checks to `pr-gate`**

Add commands:

```yaml
- name: Cloudflare Container deployment tests
  run: |
    pytest tests/test_checkpoint_fetch.py tests/test_cloudflare_config.py -v
    node --test tests/test_veya_routing.mjs
    python deploy/hf-space/assemble.py
    python deploy/hf-space/assemble.py --check
```

- [ ] **Step 2: Document the Cloudflare runtime contract**

Update the deployment README to identify the same image as usable by Cloudflare Containers, list the R2 checkpoint source, and state that `/readyz` stays 503 during model loading. Do not publish credentials, account tokens, or checkpoint URLs containing secrets.

- [ ] **Step 3: Run the CI-equivalent checks locally**

```bash
.venv/bin/python -m pytest tests/test_checkpoint_fetch.py tests/test_cloudflare_config.py tests/test_health_probes.py tests/test_web_gateway.py -v
node --test tests/test_veya_routing.mjs tests/test_model_client.mjs tests/test_site_content.mjs
python3 deploy/hf-space/assemble.py
python3 deploy/hf-space/assemble.py --check
```

Expected: all tests pass and staged checkpoint remains absent.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml deploy/hf-space/README.md
git commit -m "ci(deploy): validate Cloudflare Container release"
```

---

### Task 6: Provision paid Cloudflare resources securely

**Files:**
- No repository files.

**Interfaces:**
- Produces: Workers Paid entitlement, private R2 bucket `veya-model-artifacts`, object `releases/0.3.5-audit-baseline.1/model.safetensors`, and Worker secret `VEYA_ARTIFACT_TOKEN` in account `8d0a6a64de03f4c9b14184398e75aa6e`.

- [ ] **Step 1: Confirm target account and Workers Paid entitlement**

In the Cloudflare dashboard, select account `veya` / `8d0a6a64de03f4c9b14184398e75aa6e`. Upgrade to Workers Paid only if the account is not already paid. Confirm Containers are available before creating resources.

- [ ] **Step 2: Create the private R2 bucket**

Create `veya-model-artifacts` with no public development URL and no custom domain.

- [ ] **Step 3: Verify the local checkpoint before upload**

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("artifacts/release/manifest.json").read_text())
path = Path(manifest["base_checkpoint"]) / "model.safetensors"
h = hashlib.sha256(path.read_bytes()).hexdigest()
assert h == manifest["base_checkpoint_hash"], (h, manifest["base_checkpoint_hash"])
print(path, h)
PY
```

Expected digest: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.

- [ ] **Step 4: Upload the checkpoint**

Use the dashboard or a Wrangler session authenticated to the target account to upload:

```text
bucket: veya-model-artifacts
key: releases/0.3.5-audit-baseline.1/model.safetensors
```

Do not use the local Wrangler session while it lists only account `60690c4aa1e8b60819a655ffbf26e122`.

- [ ] **Step 5: Create the Worker secret**

Generate and pipe a 32-byte random secret without printing it:

```bash
openssl rand -hex 32 | npx wrangler secret put VEYA_ARTIFACT_TOKEN
```

Run this only under credentials for the target account. Never pass the secret as a command argument or commit it.

- [ ] **Step 6: Verify private storage**

Confirm the R2 bucket has no public URL and unauthenticated requests cannot read the object. Confirm the Worker secret exists by name; do not retrieve or display its value.

---

### Task 7: Deploy the Worker and Container

**Files:**
- No additional repository files unless the build exposes a concrete compatibility issue.

**Interfaces:**
- Consumes: all code and Cloudflare resources from Tasks 1–6.
- Produces: live Worker revision with Container backend.

- [ ] **Step 1: Push the implementation branch and let CI pass**

Review staged files individually, then push the implementation branch. Do not include `.wrangler/`, `.env*`, credentials, or `model.safetensors`.

- [ ] **Step 2: Configure the GitHub-connected Workers Build**

Use build command:

```bash
npx wrangler deploy
```

The Wrangler custom build command generates `build/hf-space/` before the image build. Verify the build environment has a Docker-compatible engine. If it does not, use the documented Cloudflare fallback: build a private image in CI, push it to a supported registry, pin its digest in `wrangler.jsonc`, and rerun the dry-run before deployment.

- [ ] **Step 3: Deploy to a preview Worker first**

Deploy the same bindings under a preview environment or temporary Worker name. Do not overwrite the production `veya` Worker until the Container is running and ready.

- [ ] **Step 4: Observe startup**

Use Worker and Container logs. Expected sequence:

```text
gateway live
checkpoint downloaded
checkpoint sha256=891102d372688fc2...
audited release latched
```

Any digest mismatch, missing object, missing secret, CUDA dependency, or OOM is a failed deployment.

- [ ] **Step 5: Verify preview endpoints**

Check `/healthz`, `/readyz`, `/api/health`, `/api/version`, and `/api/benchmark`. Run both a two-candidate choice and a 25-candidate choice. Expect `packed` and `shortlist_k8_packed_rerank` respectively.

- [ ] **Step 6: Promote production**

After preview verification, deploy the same revision to Worker `veya`. Confirm the Worker no longer reads `VEYA_ORIGIN` and routes to `VEYA_CONTAINER`.

---

### Task 8: Prove laptop independence and finalize rollback

**Files:**
- Modify: `docs/OPERATIONS.md`
- Delete only after successful verification: `~/Library/LaunchAgents/com.veya.serve.plist` (outside repository; requires explicit confirmation at action time)

**Interfaces:**
- Produces: verified cloud-only serving path and documented rollback.

- [ ] **Step 1: Stop local serving without deleting configuration**

Stop `web.model_server`, `web.server`, and Tailscale Funnel. Keep the LaunchAgent and Tailscale policy until the public validation completes.

- [ ] **Step 2: Run public golden-path verification**

```bash
curl -fsS https://veya.veya-8d0.workers.dev/api/health
curl -fsS https://veya.veya-8d0.workers.dev/api/version
curl -fsS https://veya.veya-8d0.workers.dev/api/benchmark
```

POST the two inference fixtures from Task 7 and verify the release is `0.3.5-audit-baseline.1`, routes are correct, answers belong to the submitted criteria, and probabilities sum to 1.0.

- [ ] **Step 3: Verify restart recovery**

Use a Container rollout or controlled stop, then confirm the Cron Trigger restarts the singleton and `/readyz` returns 200 after the audited cold-start sequence. Do not claim always-on behavior solely from one warm request.

- [ ] **Step 4: Verify browser behavior**

Open the homepage, playground, and benchmarks page. Submit a real inference in the browser. Confirm no console errors, failed requests, CORS errors, or regressions in static pages.

- [ ] **Step 5: Update operations documentation**

Document Container logs, R2 object key, readiness behavior, expected cold-start time, billing dimensions, rollback to Tailscale, and the target account mismatch that prevents local deployment with the current Wrangler credentials.

- [ ] **Step 6: Commit and push operations documentation**

```bash
git add docs/OPERATIONS.md
git commit -m "docs(ops): document Cloudflare Container operations"
git push origin main
```

- [ ] **Step 7: Remove laptop auto-start only with explicit approval**

After at least one restart-recovery test and user confirmation, unload and remove `com.veya.serve`. Until then it remains a rollback path but is not part of production traffic.
