---
kind: logging_system
name: Minimal stdlib logging in the web gateway only
category: logging_system
scope:
    - '**'
source_files:
    - web/server.py
    - veya/server.py
---

## What system/approach is used

The repository uses **Python's stdlib `logging` module** exclusively for runtime output. There is no third-party logger (no `loguru`, `structlog`, `sentry-sdk`, etc.), no log-level configuration file, and no structured-log framework. The only place that configures a logger is the web gateway (`web/server.py`). The core decision engine package (`veya/`) has no logging calls at all.

## Key files and packages

- `web/server.py` — the sole producer of log records. It creates a logger named `"veya.web"` and emits one kind of record via `_safe_telemetry`.
- `veya/server.py` — the local HTTP inference server. It deliberately suppresses `BaseHTTPRequestHandler.log_message` (overridden to return) so the stdlib access log is silent; it contains no `logger.info`/`logger.error` calls.

No other `.py` file in the repo imports `logging` or calls `logger.*`.

## Architecture and conventions

### Logger identity and initialization

At module load time in `web/server.py`:

```python
logger = logging.getLogger("veya.web")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
```

This means:
- The logger name is fixed as `veya.web`.
- A single `StreamHandler` writing to `sys.stderr` is attached once.
- The formatter is a bare `%(message)s` — no timestamp, level, or module prefix is emitted by default.
- `propagate = False` isolates this logger from the root logger.
- Level is set to `INFO`; nothing below INFO is emitted.

### Log format

All telemetry goes through `_safe_telemetry(request_id, primitive, candidate_count, route, deferred, status, started, model_version, error_code)` which serializes a dict with `json.dumps(..., separators=(",", ":"))` and logs it as an `INFO` message. The resulting line on stderr is a compact JSON object with these fields:

| Field | Meaning |
|---|---|
| `request_id` | `secrets.token_hex(8)` generated per request |
| `primitive` | `choice`, `score`, `noul`, or `systemone` |
| `candidate_count` | number of criteria (or sum over systemone questions) |
| `route` | upstream response field `route` (e.g. `packed`) |
| `deferred` | upstream response field `deferred` |
| `status` | HTTP status code returned to the client |
| `latency_ms` | wall-clock ms measured with `time.monotonic()` |
| `model_version` | propagated from upstream `/version` when available |
| `error_code` | internal error token such as `rate_limited`, `model_busy`, `timeout`, `invalid_upstream_response` |

### Sinks

There is exactly one sink: `logging.StreamHandler` → `stderr`. There is no file sink, no syslog/UDP/HTTP sink, no rotation, and no way to reconfigure the logger at runtime. Consumers must attach their own handlers to the `veya.web` logger if they need richer output (the `if not logger.handlers:` guard leaves room for external configuration before the module finishes loading).

### Suppressed HTTP access logs

Both `veya/server.py` and `web/server.py` override `BaseHTTPRequestHandler.log_message` to do nothing, suppressing Python's built-in access log lines. This is intentional: the only structured output is the explicit telemetry JSON.

## Conventions and constraints

- **Only the web gateway logs.** The `veya/` package (agent, router, reranker, shortlist, calibration, manifest, release_bundle, email, lang, presets, selective, server) contains zero logging calls. All diagnostic output lives in `web/server.py`.
- **Log level is INFO.** The logger is explicitly set to `logging.INFO`; no DEBUG/WARNING/ERROR levels are used anywhere in the codebase.
- **Structured payload, plain text line.** Each log event is a single line containing a compact JSON object produced by `json.dumps(..., separators=(",", ":"))`. There is no envelope around it — the entire line is the JSON.
- **No sensitive data in logs.** The telemetry omits request bodies, state, user input, and bearer tokens. `request_id` is random hex and carries no PII.
- **Stdlib-only dependency.** No extra package is required for logging; the project depends only on Python stdlib for this concern.
- **No log-level CLI flag or env var.** Unlike other runtime knobs (`VEYA_MODEL_URL`, `VEYA_WEB_PORT`, `VEYA_PUBLIC_ORIGIN`, `VEYA_ALLOWED_ORIGINS`, `VEYA_MODEL_TOKEN`), there is no environment variable controlling verbosity.
- **Handler attachment is idempotent.** The `if not logger.handlers:` guard allows a caller to install additional handlers on the same logger before the first import completes, but the default StreamHandler is installed automatically on first use.