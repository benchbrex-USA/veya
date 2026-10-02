# Veya Shadow Deployment Procedure

Operational procedure for the live-shadow evaluation of the audited serving
stack (commercialization gate G20). A shadow deployment runs a full copy of the
production stack that processes mirrored production traffic and records
telemetry for offline evaluation. It never serves clients and never affects
production responses.

Companion documents:

- [`docs/OPERATIONS.md`](OPERATIONS.md) — production deployment, configuration, model updates
- [`docs/RUNBOOK.md`](RUNBOOK.md) — drift alerts and incident response (references the same telemetry fields)
- [`artifacts/release/shadow_schema.json`](../artifacts/release/shadow_schema.json) — record schema, sample-size requirements, acceptance criteria, analysis commands, gap status
- [`docs/PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) — G20 summary row and next action

Current status: **READY in-repository; BLOCKED_EXTERNAL on live traffic and
cluster access** (see "Status: READY vs BLOCKED_EXTERNAL" at the end).

## What shadow mode does

- `VEYA_SHADOW_MODE=1|true` (`web/server.py`, `_shadow_enabled`) switches the
  gateway into shadow mode. The same gateway binary serves both modes.
- Every successful `POST /api/infer` response carries the header
  `X-Veya-Shadow: true` (`web/server.py`, `send(..., shadow=...)`). Error
  responses (4xx/5xx) do **not** carry the header, but their telemetry lines
  still carry `"shadow": true`.
- Every telemetry line the gateway writes to stdout/stderr (logger `veya.web`,
  `_safe_telemetry`) carries `"shadow": true` when the mode is on, alongside
  the standard fields: `request_id`, `primitive`, `candidate_count`, `route`,
  `deferred`, `status`, `latency_ms`, `model_version`, `error_code`,
  `timestamp`, `confidence`, `max_probability`, `automation`, `config_hash`.
- Telemetry never contains request text or candidate labels — the raw state
  text is not logged, in shadow mode as anywhere else.
- `GET /readyz` serves the release join keys as response headers only when all
  readiness components latch: `X-Veya-Model-Hash` (checkpoint SHA-256) and
  `X-Veya-Policy-Hash` (selective policy SHA-256). A manifest mismatch keeps
  `/readyz` at 503 for the life of the process (fail-closed, G18).

The deployment topology is unchanged from production: one pod runs the audited
model service (`web/model_server.py` → `veya/server.py`, bound to
`127.0.0.1:8080` inside the container) and the gateway (`web/server.py`, port
8000, the only exposed surface).

## Prerequisites

- Production K8s cluster with routing capability (Istio/Envoy mesh, or an
  application layer that can duplicate requests).
- Veya production image built and pushed to the registry
  (`Dockerfile`; deployment procedure in [`docs/OPERATIONS.md`](OPERATIONS.md)).
  The shadow overlay inherits the base deployment and overrides the image per
  environment.
- Production traffic source that can mirror or split requests. Mirrored
  requests must carry `Content-Type: application/json` and a body that passes
  `web/contracts.py` validation (same contract as production traffic).
- A plan for ground-truth labeling of collected records (`ground_truth` /
  `correct` joins, see Step 5) and, for the full blind verdict, a minted
  `veya-freeze-record-v1` freeze record (`research/veyabench/freeze_record.py`).

## Step 1: Deploy the shadow instance

```bash
kubectl apply -k k8s/shadow-overlay/
kubectl rollout status deployment/shadow-veya
```

What the overlay (`k8s/shadow-overlay/kustomization.yaml`) changes on top of
the base stack:

- `namePrefix: shadow-` renames every resource — Deployment/Service/HPA
  `shadow-veya`, ConfigMap `shadow-veya-config`. Service DNS inside the
  cluster: `shadow-veya` (ClusterIP, port 80 → container port 8000).
- the ConfigMap key `shadow-mode` is forced to `"1"`; the base Deployment
  sources `VEYA_SHADOW_MODE` from it.
- selector labels are patched to `app.kubernetes.io/name: shadow-veya` on the
  Deployment, its pod template, and the Service, so shadow and production
  routing stay strictly disjoint even when both stacks coexist in one cluster.
- 1 replica (mirrored traffic, not production capacity) and HPA bounded to
  1–2 replicas.

## Step 2: Configure traffic routing

Route a **copy** of production traffic to the shadow Service `shadow-veya`.
Production responses are untouched; the shadow response is discarded by the
routing layer.

### Option A: Istio/Envoy traffic mirroring

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: veya-shadow-mirror
spec:
  hosts:
    - veya.example.com          # the production host fronting the veya Service
  http:
    - route:
        - destination:
            host: veya          # production Service (k8s/service.yaml)
            port:
              number: 80
      mirror:
        host: shadow-veya       # shadow Service (k8s/shadow-overlay namePrefix)
        port:
          number: 80
      mirrorPercentage:
        value: 5.0              # start small; see rate budget below
```

Raw Envoy: a `request_mirror_policy` on the route pointing at the
`outbound|<namespace>|shadow-veya|80||` cluster is equivalent.

### Option B: Application-level request duplication

The production application (or an edge proxy it controls) sends the same JSON
body to both the production gateway and the shadow Service, discards the
shadow response, and must treat any shadow error as non-authoritative
(fire-and-forget; never block or fail production on a shadow result).

### Routing caveats

- **Origin header**: the gateway rejects `POST /api/infer` with a foreign
  `Origin` header ("Cross-origin model requests are not allowed").
  Browser traffic mirrored by a mesh keeps its production `Origin`; either
  strip the `Origin` header on the mirror leg or allowlist the production
  origin on the shadow deployment only:
  `kubectl set env deployment/shadow-veya VEYA_ALLOWED_ORIGINS=https://veya.example.com`
- **Rate limit**: the gateway admits `CAPACITY = 12` weight per `WINDOW_S = 60`
  seconds per source IP (`web/server.py`). One typical request (≤ 19
  candidates) weighs 1; weight grows as `1 + ceil((candidates - 19) / 20)`,
  capped at 12. Mirrored requests arrive from the mirror's IP — per
  production pod with sidecar mirroring — so size `mirrorPercentage` to the
  budget: at most ~12 typical requests/minute per source address. Excess
  mirrored requests get HTTP 429 (`error_code: rate_limited`), which is safe
  for production but is recorded as a non-collected record.
- **Concurrency and timeouts**: the shadow gateway runs `MAX_INFLIGHT = 2`
  inferences and `VEYA_REQUEST_TIMEOUT` (default 30 s) per request; mirrored
  bursts above capacity get 503 `model_busy`. Mirror timeouts should not be
  shorter than production latency plus 5 s, or aborted mirrors are logged as
  client-side drops.
- **Correlation**: `request_id` is generated server-side per request
  (`secrets.token_hex(8)`), so the production request and its shadow copy have
  different ids. Option B can capture both; for Option A, join ground truth to
  shadow records by the shadow telemetry `request_id` (returned in the
  response body on success, `X-Request-Id` header on errors) and correlate
  production labels approximately on (timestamp, primitive, candidate_count).
  Label joins are exact only when the duplication layer records the id pair.

## Step 3: Verify shadow health

```bash
kubectl get pods -l app.kubernetes.io/name=shadow-veya
kubectl get svc shadow-veya

# from any pod in the same namespace:
curl -s http://shadow-veya/readyz

# or from the workstation:
kubectl port-forward svc/shadow-veya 8080:80 &
curl -s -D - http://127.0.0.1:8080/readyz -o /tmp/readyz_body.json
```

Expected: `/readyz` returns 200 with body
`{"status": "ready", "components": {...all true...}}` and the release join
headers:

```
X-Veya-Model-Hash: <checkpoint_sha256>
X-Veya-Policy-Hash: <selective_policy_sha256>
```

Cold start is ~34 s (manifest verification + 2.2 GB checkpoint load) and can
exceed a minute under CPU load; the startup probe allows 120 s. Liveness is
`/healthz`, startup is `/startupz`. If `/readyz` stays 503 after the startup
budget, the pod failed hash verification and will not serve — check
`kubectl logs deployment/shadow-veya -c veya`.

## Step 4: Monitor collection

Telemetry lands on the pod's stdout (container `veya`). Inspect and count in
one step:

```bash
kubectl logs -l app.kubernetes.io/name=shadow-veya -c veya | grep '"shadow":true'
kubectl logs -l app.kubernetes.io/name=shadow-veya -c veya | grep -c '"shadow":true'
```

Container logs are ephemeral; ship them to the collection store during the run
(e.g., append to a file/object store periodically), because Step 6 deletes the
pods:

```bash
kubectl logs -l app.kubernetes.io/name=shadow-veya -c veya | grep '"shadow":true' >> shadow_2026W40.jsonl
```

Expected collection rate: `production QPS × mirrorPercentage`, bounded by the
gateway's rate limit (`12` weight/minute per source IP, see Step 2). A quiet
service mirrors slowly; plan collection across whole periods (the drift
monitor partitions by `week` by default) so each period has enough records for
its statistics.

Required sample size (`research/veyabench/sample_size.py`, Wilson-interval
inversion; consumed by `blind_verdict.py`'s sample-size check). Run from the
repository root:

```bash
.venv/bin/python -c "from research.veyabench.sample_size import required_n; print(required_n(0.02, 593/3000, 0))"
# {'epsilon': 0.02, 'confidence': 0.95, ..., 'max_errors': 0,
#  'anticipated_coverage': 0.197666..., 'accepted': 189, 'total': 957,
#  'wilson_ucb_at_accepted': 0.019920295377281655}
```

- `epsilon=0.02`, `confidence=0.95` are the frozen risk parameters.
- `accepted` is coverage-independent: **189 accepted decisions** (zero
  accepted errors) clear the risk gate; 188 do not (fail-closed).
- `total = ceil(accepted / c_min)` scales with the freeze record's `c_min`:
  957 total at `c_min = 593/3000` (0.1977), 956 at the rounded 0.1977; with up
  to 3 observed accepted errors, 437 accepted / 2211 total. Recompute against
  the actual pre-registered `c_min` before freezing the collection target.
- The shipped policy has `operating_epsilon: "disabled"`
  (`artifacts/release/selective_policy.json`), so the deployment itself accepts
  nothing at runtime — all records carry `automation: "defer"` (shortlist
  route) or `null` (packed route). Acceptance for the gates is computed
  **offline** in Step 5 by applying a policy copy at a candidate operating
  point; the collected pool is still the `total` figure above.

## Step 5: Analyze results

### Assemble records

One record per mirrored request, in the
[`shadow_schema.json`](../artifacts/release/shadow_schema.json) shape. Four
sources join:

1. the telemetry line (all runtime fields),
2. `X-Veya-Model-Hash` / `X-Veya-Policy-Hash` captured from `/readyz` (per
   release; `config_hash` already rides inside the telemetry line),
3. the served prediction (response body: `decision`, `answer`,
   `probabilities`),
4. the eventual ground-truth label (`ground_truth`, `correct`).

Required record fields (schema `record_schema`): `timestamp`, `request_id`,
`model_hash`, `policy_hash`, `config_hash`, `primitive`, `prediction`,
`confidence`, `automation`, `selective_decision`, `latency_ms`,
`ground_truth` (null until labeled), `correct` (null until labeled),
`slice_metadata` (`candidate_count`, `route`), `slice_tag`,
`incumbent_outcome` (null until known).

### Transform for drift_monitor (required)

`research/veyabench/drift_monitor.py` reads `candidate_count` and `route` at
the **top level** of each record (they feed the candidate-count and
input-length distributions), while the schema nests them under
`slice_metadata`. Flatten before calling the monitor, and join labels:

```python
import json

def flatten_for_drift(record, ground_truth=None, correct=None):
    """shadow_schema.json record -> drift_monitor.py input."""
    out = {k: v for k, v in record.items() if k != "slice_metadata"}
    out["candidate_count"] = record["slice_metadata"]["candidate_count"]
    out["route"] = record["slice_metadata"]["route"]
    out["ground_truth"] = ground_truth
    out["correct"] = correct
    return out
```

Skipping the flatten silently loses data: an un-flattened record yields
`candidate_count_distribution {"null": n}` and `input_length_distribution
null`, while the flattened record yields the true distributions (both
behaviors are pinned by `tests/test_shadow_deployment.py`).

Labeling rule: set `correct` to `true`/`false` for every labeled record.
`drift_monitor.period_stats` counts an **error** for every record with
`automation == "answer"` whose `correct` is falsy — an accepted record left
unlabeled (`correct: null`) counts as an error as soon as any record in the
same period carries ground truth.

### Drift report

```bash
.venv/bin/python -c "import json; from research.veyabench.drift_monitor import drift_report, drift_triggers; records = json.load(open('shadow_data.json')); report = drift_report(records, period='week'); print(json.dumps(report, indent=2)); print(json.dumps(drift_triggers(report, epsilon=0.02), indent=2))"
```

- `records` is the JSON array of flattened records above; `period` is
  `day`/`week`/`month`.
- Report format `veya-drift-report-v1`. Triggers (fail-closed):
  `risk_breach` (period `risk_ucb95 > epsilon`; needs labeled accepted
  records), `config_mismatch` (two release hashes in one period),
  `coverage_drop` (below 0.8 × first-period baseline), `distribution_shift`
  (confidence-histogram KS test, p < 0.01). Alert semantics:
  [`docs/RUNBOOK.md`](RUNBOOK.md) § "Drift Alert Triggered".

### Selective gates

The shipped policy defers all automation (`operating_epsilon: "disabled"`), so
gate runs need a shadow-only policy copy with an operating point enabled
(thresholds already frozen in the artifact: `0.02` → `0.9796500892257723`,
`0.05` → `0.9541679372957224`, `0.10` → `0.896261827143334`):

```bash
python3 - <<'EOF'
import json
policy = json.load(open("artifacts/release/selective_policy.json"))
policy["operating_epsilon"] = "0.02"   # one of "0.02" / "0.05" / "0.10"
json.dump(policy, open("shadow_policy_0.02.json", "w"), indent=2)
EOF

.venv/bin/python -c "import json; from research.veyabench.selective_gates import gate_report; cases = json.load(open('shadow_cases.json')); policy = json.load(open('shadow_policy_0.02.json')); print(json.dumps(gate_report(cases, policy, 0.02, 0.95, {'coverage': 0.05, 'accuracy': 0.30}), indent=2))"
```

`shadow_cases.json` carries one case per record following the blind case
contract (`correct` required on every case; `probabilities` drive the
threshold score, `slice_tag` selects clean vs shifted slices).

### Full verdict

The end-to-end `veya-shadow-evaluation-v1` report is composed by
`research/veyabench/blind_verdict.py` (`run_blind_verdict`, verdict layer
format `veya-blind-verdict-v1`) with annexes in `veya-drift-report-v1`,
`veya-selective-gates-v1`, and `veya-contamination-report-v1`. The exact
command template is `analysis_commands.full_verdict` in
[`shadow_schema.json`](../artifacts/release/shadow_schema.json); the config
needs `freeze_record`, `cases`, `training_texts`, and `policy`.

## Step 6: Teardown

Preserve collected telemetry **before** deleting the deployment (logs live and
die with the pods), then remove the overlay:

```bash
kubectl logs -l app.kubernetes.io/name=shadow-veya -c veya | grep '"shadow":true' > shadow_final_$(date +%Y%m%d).jsonl

kubectl delete -k k8s/shadow-overlay/
```

`kubectl delete -k k8s/shadow-overlay/` removes exactly the shadow resources
(`shadow-veya` Deployment/Service/HPA and `shadow-veya-config`). Production
resources are untouched: the disjoint `app.kubernetes.io/name: shadow-veya`
selector value guarantees no overlap.

## Verified dry run (2026-09-30)

Everything below was executed against this repository — the closest the
repository alone can get to the production procedure, since no cluster or live
traffic exists yet.

**Focused end-to-end test** (`tests/test_shadow_deployment.py`, runs the real
gateway process in shadow mode against a model-service double):

```bash
.venv/bin/python -m pytest tests/test_shadow_deployment.py -v
# 13 passed — covers: X-Veya-Shadow header on success / absent when the mode is
# off, telemetry fields incl. "shadow": true and max_probability capture, the
# schema record join, the flatten transform, drift compatibility (un-flattened
# records lose the distributions), config_mismatch on a mid-period release
# change, and the sample-size cliff (189 accepted clears epsilon=0.02; 188
# fires risk_breach).
```

**Real-stack dry run** (audited model service + gateway in shadow mode, same
commands as this document, on the repository host):

```bash
VEYA_MODEL_PORT=8089 .venv/bin/python -m web.model_server &
VEYA_SHADOW_MODE=1 VEYA_WEB_HOST=127.0.0.1 VEYA_WEB_PORT=3011 \
  VEYA_MODEL_URL=http://127.0.0.1:8089 .venv/bin/python -m web.server &
```

Observed:

- `/readyz` → `{"status": "ready", "components": {...all true...}}` with
  `X-Veya-Model-Hash: 891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
  and
  `X-Veya-Policy-Hash: 80e7be582c9c890cf73da701cbd86a71c6d908331186ee68f1021763011cd620`
  (model cold start 63 s warm filesystem cache; 211 s on first load).
- Two requests (`POST /api/infer`, the `web/contracts.py` shape), both
  HTTP 200 with `X-Veya-Shadow: true`. Telemetry lines:

```json
{"request_id":"072d19bbcc6704db","primitive":"choice","candidate_count":2,"route":"packed","deferred":false,"status":200,"latency_ms":781,"model_version":"0.3.5","error_code":null,"timestamp":"2026-09-29T19:50:20.446350+00:00","confidence":0.7651,"max_probability":0.9616,"automation":null,"config_hash":"cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de","shadow":true}
{"request_id":"c0c3ebed8f9dd6a3","primitive":"choice","candidate_count":26,"route":"shortlist_k8_packed_rerank","deferred":false,"status":200,"latency_ms":627,"model_version":"0.3.5","error_code":null,"timestamp":"2026-09-29T19:50:22.091247+00:00","confidence":0.0388,"max_probability":0.2321,"automation":"defer","config_hash":"cd1e20296811ed80e4e400ab52ee812bf927e4e9b879b758656542d72410e5de","shadow":true}
```

- Flattened and fed to `drift_monitor.drift_report(..., period='week')`:
  partition `2026-W40`, `n=2`, `accepted_n=0`, `deferred_n=1`, `coverage=0.0`,
  `risk=null` (unlabeled), `mean_latency_ms=704.0`,
  `candidate_count_distribution {"2": 1, "26": 1}`, single `config_hash`,
  `drift_triggers(..., epsilon=0.02) == []`. The same records un-flattened
  yield `candidate_count_distribution {"null": 2}` — the transform in Step 5
  is what makes the pipeline read the records correctly.
- `automation` semantics match the shipped policy: the packed-route record has
  `automation: null`, the shortlist-route record `automation: "defer"` —
  nothing is accepted at runtime while `operating_epsilon` is `"disabled"`.
- `max_probability` is captured from the raw model-service payload for
  telemetry (`web/server.py`, `_telemetry_max_probability`) while the client
  payload never exposes it — shadow records get the score the client contract
  strips.

## Status: READY vs BLOCKED_EXTERNAL

### READY (in-repository, verified)

- **Shadow image and mode**: `VEYA_SHADOW_MODE` telemetry flag,
  `X-Veya-Shadow` header, `max_probability` capture
  (`web/server.py`; pinned by `tests/test_shadow_deployment.py`,
  `tests/test_telemetry_max_probability.py`, `tests/test_web_gateway.py`).
- **K8s overlay**: `k8s/shadow-overlay/kustomization.yaml` (disjoint
  `shadow-veya` resources, single replica, HPA 1–2, `shadow-mode` forced to
  `"1"`; structure verified against the base manifests).
- **Telemetry capture**: one JSON line per request, `"shadow": true`, all
  record-schema runtime fields (dry run above).
- **Analysis pipeline**: `research/veyabench/drift_monitor.py`,
  `selective_gates.py`, `blind_verdict.py`, `sample_size.py`,
  `freeze_record.py`, `calibration_report.py` — all exercised by the dry run
  and the test suite; the flatten transform is documented and pinned.
- **Sample-size requirements**: 189 accepted / 957 total at
  `c_min = 593/3000` (re-verified by execution, see Step 4).
- **Procedure**: this document, including the routing options and caveats.

### BLOCKED_EXTERNAL (per `artifacts/release/gap_audit.json` G20)

- **Live traffic routing credentials/infrastructure**: no production traffic
  exists yet, so there is nothing to mirror. The mirror/split source (mesh
  VirtualService or application duplication), its egress identities, and the
  `Origin`/rate-limit accommodations in Step 2 must be provisioned in the
  production environment.
- **Production cluster access**: deploying `k8s/shadow-overlay/`, observing
  rollouts, and scraping pod logs require a live cluster with `kubectl` —
  operations in Steps 1, 3, 4, and 6 run only there.
- **Ground-truth labeling feed**: `ground_truth` / `correct` joins come from
  the production escalation outcome (Step 5); the per-period risk statistics
  stay `null` until records are labeled, and the full blind verdict needs the
  labeled cases plus a minted `veya-freeze-record-v1` record.

Until those exist, G20 remains `BLOCKED_EXTERNAL` by design:
`artifacts/release/shadow_schema.json` carries the same status, and
`docs/PRODUCTION_READINESS.md` lists the matching next action.
