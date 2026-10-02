---
kind: error_handling
name: 'Error Handling in Veya: Request/Upstream Contract Errors and Standard Exceptions'
category: error_handling
scope:
    - '**'
source_files:
    - web/contracts.py
    - web/server.py
    - veya/server.py
    - veya/agent.py
    - veya/common.py
    - veya/calibration.py
---

## Approach

Veya uses a layered, exception-based error model with no third-party error framework. Errors are raised as Python exceptions and converted to HTTP responses at the boundary layers:

- **Library layer** (`veya/*`): raises standard built-in exceptions (`ValueError`, `FileNotFoundError`) plus domain-specific subclasses of `ValueError`.
- **HTTP gateway layer** (`web/server.py`): catches those exceptions and maps them to JSON responses with a uniform `{decision, defer_reason, detail?}` shape.
- **Public contract layer** (`web/contracts.py`): defines two dedicated exception types for request validation vs. upstream response validation, both subclassing `ValueError`.

There is no centralized error registry, no error codes enum, no `try/except` middleware wrapping every call — instead, each handler explicitly catches the exceptions it knows about.

## Key Files

- `web/contracts.py` — public API contract validators; defines `RequestError` and `UpstreamContractError` (both `ValueError` subclasses) and all request/response validation helpers.
- `web/server.py` — FastAPI-style web server that imports `RequestError` / `UpstreamContractError`, validates Content-Type/CORS/Content-Length, proxies to the model service, and converts errors to HTTP 400/413/5xx JSON responses.
- `veya/server.py` — stdlib `http.server` fallback server; defines its own `RequestTooLarge(ValueError)` and catches `(AttributeError, KeyError, TypeError, ValueError)` to return `{"decision": "invalid", "defer_reason": "invalid_schema", "detail": ...}` with HTTP 400.
- `veya/agent.py` — core inference runtime; raises `ValueError` for schema mismatches, `FileNotFoundError` for missing checkpoints, and catches `RuntimeError` / `torch.cuda.OutOfMemoryError` to fall back from GPU to CPU.
- `veya/common.py`, `veya/calibration.py` — raise `ValueError` for invalid configuration values (e.g., temperature bounds, calibration format).

## Architecture and Conventions

### Two-tier exception taxonomy in the web layer

`web/contracts.py` separates client-facing validation errors from upstream-response validation errors:

| Exception | Base class | Meaning |
|---|---|---|
| `RequestError` | `ValueError` | The caller sent an invalid request (bad primitive, duplicate keys, oversized state, unsupported question type, etc.). |
| `UpstreamContractError` | `ValueError` | The model service returned a response that violates the audited contract (missing probability distribution, probabilities not summing to one, unexpected label, answer absent from distribution, etc.). |

Both are caught by `web/server.py`'s top-level handlers and mapped to HTTP responses:
- `RequestError` → HTTP 400 with `{"decision": "invalid", "defer_reason": ..., "detail": str(exc)}`.
- `UpstreamContractError` → HTTP 502 (Bad Gateway), because it signals a broken upstream model service rather than a bad client request.

This distinction is enforced by the validator functions themselves: `_state`, `_text`, `_criteria`, `validate`, `_probabilities`, `_answer`, and `validate_response` only raise one of these two types — never bare `ValueError` — so callers can distinguish client vs. upstream faults.

### Uniform JSON error envelope

The HTTP servers normalize all errors into a small set of fields:

- `decision`: usually `"invalid"` for client errors or `"defer"` for model deferrals.
- `defer_reason`: machine-readable reason string (`"invalid_json"`, `"request_too_large"`, `"not_found"`, `"hypothesis_not_loaded"`, `"invalid_schema"`, `"model_deferred"`, …).
- `detail`: optional human-readable message (only attached when the server has the underlying exception text).

This is visible in `veya/server.py`'s `_send` calls on lines 178, 181, 216, 233 and in `web/server.py`'s error branches around line 544–626.

### GPU memory resilience

`Agent.__init__` and `Agent.system_one` both wrap device placement / forward passes in `try/except (RuntimeError, torch.cuda.OutOfMemoryError)` blocks. On OOM they print a warning, switch `self.device` to CPU, reset dtype to float32, and retry. This is the only place in the codebase where a hardware failure is treated as recoverable rather than fatal.

### Validation-by-exception pattern

Every input-validation helper follows the same shape: validate a single field and `raise` a descriptive `RequestError` / `ValueError`. There is no return-code convention — validation failures are signaled purely via exceptions. The `validate` function in `web/contracts.py` composes these helpers and returns a normalized dict only on success.

### No panics, no global error hooks

Python has no panic/recover semantics. The codebase does not use `sys.excepthook`, Sentry, structured logging frameworks, or a central error bus. Errors propagate up the stack until a handler catches them.

## Conventions and Constraints

1. **Client-side request validation must raise `RequestError`**, not a bare `ValueError`. Enforced by the design of `web/contracts.py`, which is the sole source of request validation logic used by the website gateway.
2. **Upstream model response validation must raise `UpstreamContractError`**, not a bare `ValueError`. Enforced by the `_probabilities`, `_number`, `_optional_text`, `_answer`, and `validate_response` helpers, which are the only places upstream payloads are accepted.
3. **All HTTP error responses go through `_send(code, payload)`** in `veya/server.py` and the equivalent handler in `web/server.py`; raw `self.send_response` calls outside these paths are limited to health/version endpoints.
4. **Schema validation errors surface as HTTP 400 with `defer_reason="invalid_schema"`**; malformed JSON surfaces as `defer_reason="invalid_json"`; oversized requests as `defer_reason="request_too_large"` with HTTP 413. These mappings are hard-coded in the POST handlers of both servers.
5. **GPU OOM is recovered, not re-raised**: both `Agent.__init__` and `Agent.system_one` catch `torch.cuda.OutOfMemoryError` and retry on CPU. Other `RuntimeError`s are re-raised.
6. **Missing checkpoint files raise `FileNotFoundError`** with a message naming the missing artifact (`rl_agent_config.json`, `model.safetensors`, subfolder). This is the only file-system error type used in `Agent.__init__`.
7. **Configuration/value errors raise `ValueError`** with a message describing the violated invariant (temperature bounds, calibration format, question schema, criteria cardinality). There is no custom exception hierarchy under `veya/` beyond the two in `web/`.
8. **No `except Exception:` except in non-critical compatibility code**: the broad `except Exception: pass` in `_fix_tokenizer_config` (line 51) is intentionally silent to tolerate legacy tokenizer configs; other broad `except Exception:` blocks appear only in agent initialization paths that should degrade gracefully.