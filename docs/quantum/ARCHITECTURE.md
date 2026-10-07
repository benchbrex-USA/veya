# Optional quantum calibration

Status: **experimental**. This path is not part of the audited Veya release. Unit tests do not establish calibration quality, and they do not authorize unattended hardware control.

## Roles

| Model | Role | Runtime |
|---|---|---|
| `nvidia/ising-calibration-1.5-31b` | Multimodal calibration specialist | Required only when quantum mode is enabled |
| Veya Choice, Noul, and Score | Bounded decision | Required for `/decide` |
| Critic NIM models | Text critique and disagreement | Research only, off unless listed |
| `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | Free-text safety classification | Off unless `VEYA_QUANTUM_SAFETY_GUARD=1`. Not a physics judge |

Generalist models are not mandatory. The runtime does not send plots to a model whose registry entry has image input turned off. Ising is the specialist allowed to see calibration images.

## Flow

```text
API gateway
    |
    v
Quantum API
    |
    +-- measured series, fit metadata, instrument metadata
    |
    +-- plot image
            |
            v
        Ising (external NIM)
            |
            v
        QuantumObservation (validated or rejected)
            |
            v
        Veya system-one questions
            |
            v
        policy gate --> defer / human review
```

`/api/quantum/analyze` stops after a validated observation. `/api/quantum/decide` continues to Veya. The two stages stay separate inside one process: a failed Ising call does not invent a diagnosis, and a failed Veya call does not promote Ising's recommended action into the decision.

The policy object in this build always sets `eligible_for_execution` to false and `hardware_control` to false. Agreement between Ising, critics, and Veya does not open that gate. A validated automation policy does not exist yet.

## Operating modes

1. **Minimal.** Ising, then Veya. This is the primary path.
2. **Research ensemble.** Ising, then the configured text critics, then disagreement metadata, then Veya. Set `mode` to `research_ensemble` and `VEYA_QUANTUM_CRITIC_MODELS` to an allowlisted list.
3. **Teacher factory.** Offline only. `quantum.research.assemble_teacher_record` stores teacher output and human ground truth as different fields. The HTTP API rejects this mode. Nothing trains on production traffic.

## Configuration

The caller cannot pass a model id or an upstream URL. The gateway reads:

- `VEYA_QUANTUM_ENABLED`
- `VEYA_EXTERNAL_MODEL_POLICY` (`disabled`, `self_hosted`, `hosted`)
- `VEYA_NIM_BASE_URL`
- `VEYA_NIM_API_KEY`
- `VEYA_ISING_MODEL`
- `VEYA_QUANTUM_CRITIC_MODELS`
- `VEYA_QUANTUM_ICL_K` (0 through 8, default 0)
- `VEYA_QUANTUM_CACHE` and `VEYA_QUANTUM_CACHE_ENTRIES`
- `VEYA_QUANTUM_SAFETY_GUARD`
- `VEYA_NIM_CONNECT_TIMEOUT`, `VEYA_NIM_READ_TIMEOUT`, `VEYA_NIM_MAX_RETRIES`, `VEYA_ISING_MAX_TOKENS`

`hosted` requires HTTPS and a key. `self_hosted` may use HTTP for an operator-chosen lab address. Link-local metadata hosts, URL userinfo, and non-HTTP schemes are rejected. A partial configuration stays disabled.

Hosted mode sends plots and the structured evidence in the prompt to that endpoint. Self-hosted mode is the setting for proprietary lab data. Disabled mode makes no outbound call.

## Sample request

```json
{
  "experiment_family": "rabi",
  "project_id": "lab-a",
  "measurements": {
    "series": [{"name": "excited_state", "x": [0, 20, 40], "y": [0.1, 0.4, 0.2], "x_unit": "ns", "y_unit": "arb"}],
    "metrics": {"contrast": 0.3}
  },
  "fit": {"available": false, "r2": null, "parameters": {}, "residuals_summary": ""},
  "images": [{"mime": "image/png", "data_base64": "<png>"}]
}
```

Measured values, fit values, Ising fields, and human-verified fields stay in different objects on the response. Parameter numbers are accepted only as JSON numbers. The text `approximately 5 GHz` does not become `5`.

Veya questions are fixed for this surface: `experiment_status` (choice), `fit_reliable` (noul), `severity` (score), `next_action` (choice), and `automation_eligible` (noul). Next-action candidates depend on the experiment family. Unknown families use a short conservative set.

`scores.external_model_score` is provider logprob metadata when the endpoint returns it. `scores.veya_probability` comes only from Veya. `scores.policy_decision` is the gate. Those three are not interchangeable.

## Limits and failures

Images: at most two, PNG or JPEG, at most 1.5 MB each, dimensions at most 4096, and a 4 MB request cap on the quantum routes. The existing `/api/infer` cap is unchanged. Series length, metadata size, example count, and upstream response size are capped. Timeouts and retries are capped. Upstream errors are returned as short codes. Response bodies and credentials are not copied into the client error or the log line.

If Ising times out, rate-limits, or returns text that is not the schema, the observation is null and the defer reason is `external_model_unavailable` or `invalid_observation`. Decide may still ask Veya using the measured record, and the gate remains closed. If Veya is down, the defer reason is `veya_unavailable` and Ising is not substituted as the authority.

## In-context examples

`VEYA_QUANTUM_ICL_K` bounds retrieval from 0 to 8. Example lookup is scoped to `project_id`. Another project's record is a miss, not a result. Example images are not forwarded; the prompt receives verified text fields. The cache key, when `VEYA_QUANTUM_CACHE=1`, includes the model, prompt version, schema version, image hashes, evidence hash, example ids, and project id.

## What this build does not do

- It does not load or modify `models/production/model.safetensors`.
- It does not change pinned runtime hashes or release manifests.
- It does not restore a website.
- It does not run hardware.
- It does not treat a critic or a safety model as a scientific referee.
- It does not train from Ising output.

Research notes for a future non-autoregressive support conditioner and a compact multimodal student are in `SUPPORT_CONDITIONING.md` and `NATIVE_MULTIMODAL.md`. They are not implemented in the served model.
