# Veya Runbook — Incident Response

Procedures reference the actual code, paths, and thresholds of this
repository. Scope and limits: [`docs/PRODUCTION_SCOPE.md`](PRODUCTION_SCOPE.md).
Deployment and configuration: [`docs/OPERATIONS.md`](OPERATIONS.md).

## Quick Reference

| Thing | Where |
|---|---|
| Liveness / readiness / startup probes | `/healthz`, `/readyz`, `/startupz` on the gateway (port 8000) |
| Release manifest (all SHA-256 pins) | `artifacts/release/manifest.json` |
| Selective policy | `artifacts/release/selective_policy.json` |
| Performance baseline | `artifacts/release/runtime_audit.json` |
| Drift monitor | `research/veyabench/drift_monitor.py` |
| Selective gates (G1–G4) | `research/veyabench/selective_gates.py` |
| Change protocol | [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md) |
| Concurrency model | [`docs/CONCURRENCY.md`](CONCURRENCY.md) |
| Container logs | `docker compose logs -f veya-server` / `kubectl logs -l app.kubernetes.io/name=veya -c veya` |

Telemetry: the gateway logs one JSON line per request (logger `veya.web`):
`request_id`, `primitive`, `candidate_count`, `route`, `deferred`, `status`,
`latency_ms`, `model_version`, `error_code`, `timestamp`, `confidence`,
`max_probability`, `automation`, `config_hash`. Shadow-mode entries carry
`"shadow": true`. Risk statistics additionally require `ground_truth` /
`correct` joined to the records.

---

## 1. Drift Alert Triggered

**Symptom**: an alert from `drift_triggers` (see below), or a manual review
showing period-over-period change in coverage, risk, or confidence.

**Tool**: `research/veyabench/drift_monitor.py` — build a report from
telemetry records, then evaluate triggers:

```python
import json
from research.veyabench.drift_monitor import drift_report, drift_triggers

records = [json.loads(line) for line in open("telemetry.jsonl")]
report = drift_report(records, period="week")   # or "day" / "month"
alerts = drift_triggers(report, epsilon=0.05)   # coverage_floor defaults to 0.05
print(json.dumps(alerts, indent=2))
```

Alert types (exact semantics from `drift_triggers`):

| Trigger | Fires when |
|---|---|
| `risk_breach` | A period's `risk_ucb95` (Wilson 95% upper bound) exceeds epsilon. Requires ground-truth-joined records. |
| `config_mismatch` | More than one distinct `config_hash` (release manifest SHA-256) inside one period — two release versions served concurrently. |
| `coverage_drop` | Coverage falls below 0.8 × the first period's baseline (baseline must be at least the coverage floor, 0.05). |
| `distribution_shift` | Two-sample KS test on the 10-bin confidence histograms vs the first period: p < 0.01. |

**Procedure**:

1. **Check the drift report per period** — compare current vs baseline
   distributions in `report["partitions"]`: `confidence_distribution`,
   `label_distribution`, `coverage`, `risk` / `risk_ucb95`,
   `mean_latency_ms`, `candidate_count_distribution`.
2. **If the KS test is significant** (`distribution_shift`): the model's
   confidence profile moved. Investigate the data pipeline first (did the
   input mix, candidate counts, or schema change upstream?) and then model
   degradation (rising `error_n` on the same inputs, latency growth).
3. **If `config_mismatch` fired**: two manifests were served in one period.
   Confirm which rollout was live (check `config_hash` values against
   `shasum -a 256 artifacts/release/manifest.json` at the relevant release
   tags) and split the periods on the boundary before trusting other alerts.
4. **Escalate if coverage drops below the minimum**: a `coverage_drop` alert
   means deferral is consuming the traffic — the model is no longer confident
   enough to answer at the operating threshold. Treat it as a model-quality
   incident (see "Risk Exceeds Epsilon") and do not re-enable or raise
   thresholds without new validation data.

---

## 2. Risk Exceeds Epsilon

**Symptom**: `risk_breach` from the drift monitor, or a failed G1/G2 gate in
a selective gates report. (Automation is currently disabled —
`operating_epsilon: "disabled"` — so this appears in shadow telemetry and
evaluation, not in live automated actions.)

**Immediate action**:

- **Tighten the selective threshold** — move to a higher-confidence epsilon
  bucket in `artifacts/release/selective_policy.json` (frozen reference
  values: 0.02 → `0.9796500892257723`, 0.05 → `0.9541679372957224`, 0.10 →
  `0.896261827143334`), or
- **Disable automation** — keep/return `operating_epsilon` to `"disabled"`
  (the current release state): all shortlist-route decisions report
  `automation: "defer"` and no risk is asserted.

**Investigate** — run the selective gates on recent labeled data
(`research/veyabench/selective_gates.py`). Each case needs `prediction`,
`gold_label`, `correct`, `probabilities`, and `slice_tag` (e.g. `clean`,
`short_prefix`, `long_suffix`):

```python
import copy, json
from research.veyabench.selective_gates import gate_report

policy = json.load(open("artifacts/release/selective_policy.json"))
# The shipped policy operates at "disabled" (null threshold accepts nothing,
# so every gate fails closed with "risk_ucb95 is undefined"). To evaluate a
# candidate epsilon bucket, switch operating_epsilon on a COPY — the shipped
# artifact is hash-pinned by the manifest.
candidate = copy.deepcopy(policy)
candidate["operating_epsilon"] = "0.05"
report = gate_report(
    cases, candidate,
    epsilon=0.05,
    coverage_retention_min=0.95,
    collapse_floor={"coverage": 0.05, "accuracy": 0.30},
)
print(report["failed_gates"])
print(report["gates"]["G1_clean_safety"]["reason"])
```

Gate semantics (fail-closed; a statistic that cannot support its claim fails
its gate):

- **G1 clean safety**: clean-slice `risk_ucb95` ≤ epsilon.
- **G2 shift safety**: every non-clean slice's `risk_ucb95` ≤ epsilon.
- **G3 coverage retention**: every non-clean slice keeps ≥
  `coverage_retention_min` of the clean slice's coverage.
- **G4 collapse floor**: no slice falls below the absolute minimums
  (`coverage` 0.05, `accuracy` 0.30).

**Resolution**:

- **If systematic** (G1 fails on clean data, or risk stays above epsilon
  across periods): retrain or recalibrate. Follow the calibration update and
  model update procedures in [`docs/OPERATIONS.md`](OPERATIONS.md); the
  policy must then be re-fit on a fresh validation split and the manifest
  re-minted.
- **If confined to shifted slices** (G2/G3 fail, G1 passes): the input
  distribution moved. Fix the upstream change or collect target-domain data
  and re-validate before changing anything.
- Never re-select a threshold on the data that showed the breach — the policy
  is chosen on validation and applied unchanged (`veya/selective.py`).

---

## 3. Health Probe Failing

**First command** — read the readiness component status:

```bash
curl -s http://localhost:8000/readyz | python3 -m json.tool
```

Response shape: `{"status": "ready"|"not_ready", "components": {...}}` where
`components` has `model_loaded`, `policy_loaded`, `calibration_loaded`,
`shortlist_loaded`, `hashes_valid`, `startup_failed`. Component states latch
monotonically — they cannot flap back once observed.

**Triage by component**:

| Component | Meaning | Action |
|---|---|---|
| `model_loaded: false` | The gateway cannot reach the model service's `/health` — it is down, still loading, or died at startup. A manifest hash mismatch kills the model service before it binds (`load_release_manifest` raises, so it never starts serving). | Check the model service process/container logs (`docker compose logs veya-server`, `kubectl logs -l app.kubernetes.io/name=veya -c veya`): a `ValueError` from `load_release_manifest` names the failing artifact — treat as manifest mismatch and follow [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md) ("Checkpoint Update" below: re-mint, rebuild, redeploy). Otherwise confirm the model files exist under `models/production/` and check disk space (the tree is ~2.2 GB; a truncated `model.safetensors` fails the hash check at startup). |
| `hashes_valid: false` (persistent, with `model_loaded: true`) | The model service is up but the gateway has not confirmed the audited release bindings (`release_manifest_sha256` / `checkpoint_sha256` from `/version`). | Treat as a manifest/integrity mismatch: check [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md), compare the manifest the service loaded against the one on disk (`curl -s http://localhost:8000/api/version | python3 -m json.tool` → `release_manifest_sha256` vs `shasum -a 256 artifacts/release/manifest.json`), and inspect the model service log. If `/version` cannot present valid bindings, `startup_failed` latches (see below). |
| `policy_loaded` / `calibration_loaded` / `shortlist_loaded: false` | The model service answered but could not present one of the release bindings. | Inspect the model service startup log — `load_release_manifest` raises with the artifact name (e.g. "selective policy hash does not match the release manifest"). Restore or re-mint per "Corrupted Artifact Recovery". |
| `startup_failed: true` | Sticky fail-closed: the service answered `/health` but could not present the audited release bindings (`release_manifest_sha256` / `checkpoint_sha256` missing or malformed in `/version`). | The gateway never reports ready again for the life of the process. Fix the root cause first (usually a bad manifest or tampered tree), then restart the pod/container. |

**If `/healthz` itself fails** (not `/readyz`): the gateway process is wedged
or the pod is being killed — liveness never consults the model service.
Check pod events and restart:

```bash
kubectl describe pod -l app.kubernetes.io/name=veya
kubectl get events --sort-by=.lastTimestamp | tail -20
```

---

## 4. High Latency / Overload

**Check HPA status first**:

```bash
kubectl get hpa veya
kubectl describe hpa veya
```

The HPA (`k8s/hpa.yaml`) targets 70% CPU, 2–8 replicas, scale-up of 2 pods
per 60 s (no stabilization — a new replica still needs ~34 s+ cold start
before it absorbs traffic), scale-down deliberately slow (300 s
stabilization, 1 pod per 60 s).

**Scale if needed**:

```bash
kubectl scale deployment/veya --replicas=4
```

For sustained load beyond 8 replicas, raise `maxReplicas` in `k8s/hpa.yaml`
and re-apply. Scale **out** only — per-replica concurrency is capped at
`MAX_INFLIGHT = 2` and inference serializes on `_INFER_LOCK` in
`veya/server.py`; raising concurrency only grows p99
([`docs/CONCURRENCY.md`](CONCURRENCY.md)).

**Check `MAX_INFLIGHT` saturation in logs** — the third concurrent inference
is refused with `503 model_busy` (`error_code: "OVERLOADED"`,
`Retry-After: 5`):

```bash
docker compose logs veya-server | grep model_busy
kubectl logs -l app.kubernetes.io/name=veya -c veya | grep model_busy
```

Rejected requests also appear in the `veya.security_audit` WARNING audit
trail (`rejected request ... code=model_busy`). Heavy `model_busy` counts
with low CPU mean the workload is concurrent-but-serialized — add replicas.

**Review the latency baseline** (`artifacts/release/runtime_audit.json`,
release `0.3.5-audit-baseline.1`, CPU):

| Workload | p50 | Throughput |
|---|---|---|
| Small packed choice | ≈ 105 ms | ≈ 9.5 req/s |
| Choice, 77 candidates | ≈ 163 ms | ≈ 6.1 req/s |
| Noul | ≈ 76 ms | ≈ 11.7 req/s |
| Score (categorical) | ≈ 76 ms | ≈ 12.9 req/s |
| SystemOne, 3 questions | ≈ 215 ms | ≈ 4.5 req/s |

Latency far above baseline at the same workload mix indicates contention.

**Check resource contention**:

```bash
kubectl top pods -l app.kubernetes.io/name=veya
kubectl describe pod -l app.kubernetes.io/name=veya   # look for CPU throttling / OOM events
```

Requests/limits are 1 CPU / 4 Gi and 2 CPU / 6 Gi (`k8s/deployment.yaml`);
serving RSS is ~2030 MB plus checkpoint-load headroom. Also confirm no client
is hitting the per-IP rate limit (12 cost units / 60 s; `429 rate_limited`
with `Retry-After`) and that request sizes stay under the 32 KB cap.

---

## 5. Checkpoint Update

For any change to a manifest-pinned file. Full detail in
[`docs/OPERATIONS.md`](OPERATIONS.md) ("Model Update Procedure"); the
sequence is:

1. **Follow [`docs/CHANGE_PROTOCOL.md`](CHANGE_PROTOCOL.md)** — smallest
   coherent change, focused tests, no unrelated rewrites; audited claims stay
   tied to the current release manifest.
2. **Update the model files** under `models/production/`.
3. **Re-mint the manifest** — recompute the checkpoint record with
   `build_manifest` (`veya/manifest.py`) and update every affected SHA-256
   pin in `artifacts/release/manifest.json` (`base_checkpoint_hash`,
   `model_config_hash`, `tokenizer_files`, `routing.choice_router_hash`,
   `choice.shortlist_hash`, calibration hashes, `runtime_files` if any
   `veya/*.py` changed). Regenerate the metadata record:
   ```bash
   python -m research.veyabench.release_metadata
   ```
4. **Verify before deploying** (the load takes ~34 s on CPU):
   ```bash
   python -m pytest tests/test_artifact_integrity.py
   python -c "from veya.release_bundle import load_release_manifest; load_release_manifest('artifacts/release/manifest.json', root='.')"
   ```
5. **Rebuild and deploy**:
   ```bash
   docker compose build && docker compose up -d
   # or
   kubectl apply -k k8s/ && kubectl rollout status deployment/veya
   ```
6. **Confirm**: `/readyz` returns 200 and `X-Veya-Model-Hash` matches the new
   `base_checkpoint_hash` in the manifest.

---

## 6. Corrupted Artifact Recovery

**Symptom**: the model service will not start (or `/readyz` reports
`hashes_valid: false` / `not_ready`) after disk corruption, partial copy, or
an accidental edit to a pinned file.

**Identify which hash fails** — `/readyz` only shows the component booleans;
the artifact is named in the model service startup log, where
`load_release_manifest` raises before binding:

```bash
curl -s http://localhost:8000/readyz | python3 -m json.tool
docker compose logs veya-server | grep -iE "hash|missing"
kubectl logs -l app.kubernetes.io/name=veya -c veya | grep -iE "hash|missing"
```

Error messages name the artifact, e.g. `"shortlist hash does not match the
release manifest"`, `"runtime hash does not match the release manifest"`,
`"model config file is missing: models/production/rl_agent_config.json"`.

**Confirm the corruption** by hashing the file on disk:

```bash
shasum -a 256 models/production/shortlist_k8.pt
# compare against artifacts/release/manifest.json (choice.shortlist_hash)
```

**Restore from a known-good release tag**:

```bash
git checkout <known-good-release-tag> -- models/production/ artifacts/release/
```

(or restore from your release artifact store, then re-verify the hashes).

**Re-deploy** — readiness is sticky, so restart the container/pod after
restoring the files:

```bash
docker compose down && docker compose up -d
curl http://localhost:8000/readyz  # Wait for 200
```

```bash
kubectl rollout restart deployment/veya
kubectl rollout status deployment/veya
```

If the "corruption" is actually an intentional file change, this is not
corruption — run the "Checkpoint Update" procedure instead; the manifest must
be re-minted or the service will refuse to start forever.

---

## 7. Graceful Shutdown Issues

**Expected behavior** (`web/server.py`, `_graceful_shutdown`): on
SIGTERM/SIGINT the gateway stops admitting new work and drains in-flight
requests for `SHUTDOWN_DRAIN_S = 30` seconds, then exits hard. While
draining, `/healthz` still answers 200 (liveness is process-level) and all
non-probe routes — including `POST /api/infer` — return `503 shutting_down`
(`error_code: "SHUTTING_DOWN"`) with `Retry-After: 30`, so new work diverts
to other replicas. The drain announcement is logged: `signal ... received,
draining in-flight requests for 30s...`.

**Kubernetes timing** (`k8s/deployment.yaml`): preStop `sleep 5` (endpoint
removal propagation) + up to 30 s drain = 35 s worst case;
`terminationGracePeriodSeconds: 45` leaves headroom for the exec/signal
path.

**If requests are still being dropped during shutdown**:

1. **Confirm the budget**: `terminationGracePeriodSeconds` (45) must exceed
   preStop (5 s) + drain (30 s). If someone lowered it, restore 45 or higher.
2. **Check for drain exhaustion**: requests admitted before SIGTERM get the
   full 30 s; any request needing longer than the drain window is dropped by
   the hard `os._exit(0)`. The per-request ceiling is
   `VEYA_REQUEST_TIMEOUT` (default 30 s) — a request that can legally run
   30 s can straddle the entire drain.
3. **Increase the drain timeout** when long in-flight work must complete:
   raise `SHUTDOWN_DRAIN_S` in `web/server.py` and keep
   `terminationGracePeriodSeconds` in `k8s/deployment.yaml` at ≥ 5 s + the
   new drain value. `web/server.py` is outside the manifest's
   `runtime_files` pins (only `veya/*.py` are pinned), so this is an image
   rebuild — no manifest re-mint required — then redeploy.
4. **Docker Compose**: the default `docker compose stop` grace can SIGKILL
   before the 30 s drain finishes; stop with an explicit budget:
   ```bash
   docker compose stop --timeout 45
   ```
   (or set `stop_grace_period: 45s` on the `veya-server` service).
5. **If the load balancer keeps sending traffic to a draining pod**: verify
   endpoint removal — the preStop sleep exists so removal propagates before
   SIGTERM. Check `kubectl get endpoints veya` during a rollout and the
   upstream Ingress/LB health-check configuration.
