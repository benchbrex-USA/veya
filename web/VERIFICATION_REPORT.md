# Model website verification report

## Verdict

PASS WITH GAPS. The local website performs real inference through the validated gateway. Public multi-instance deployment and formal accessibility review remain separate gates.

## Claims checked

| Claim | Evidence | Result |
|---|---|---|
| Browser does not call model service or contain a model token | `model-client.mjs` uses only `/api/*`; `VEYA_MODEL_TOKEN` read only in `web/server.py` | PASS |
| All four primitives reach the real checkpoint | Automated browser assertions for Choice, Noul, Score, SystemOne | PASS |
| Choice probabilities and route are truthful | Live Choice sum 0.9999, packed route; 20-way route showed eight shortlisted candidates | PASS |
| Unsupported input defers, and outage does not fabricate results | Browser unsupported-language and stopped-model outage assertions | PASS |
| Boundary rejects bad requests/responses and handles failures | Gateway tests: malformed JSON, duplicate/near-duplicate IDs, oversized input, malformed upstream, 5xx, timeout, 429, upstream Retry-After, cross-origin | PASS |
| Rate limiting returns retry guidance | HTTP test for weighted high-N limit and `Retry-After`; browser rate-limit UI showed Retry-After countdown | PASS WITH GAP |
| Telemetry omits raw state/candidates | Captured structured log assertions | PASS |

## Commands / harnesses

```bash
python3 -m pytest -q tests/test_web_gateway.py tests/test_server_security.py  # 12 passed
node --test tests/test_model_client.mjs  # 6 passed
python3 -m ruff check web tests/test_web_gateway.py
node --check web/static/site.js
node --check web/static/model-client.mjs
python3 -m py_compile web/server.py web/contracts.py web/model_server.py
```

Browser E2E used the Codex in-app browser's Playwright locators against the running local model and gateway. Assertions covered chosen candidate, probability sum, technical route/version, Noul, Score, SystemOne, unsupported language, high-N shortlist, service outage, and browser rate-limit recovery. The final restarted-model smoke returned billing with a 0.9999 probability sum.

## Important observations

The audited manifest serves English only. Choice uses a top-8 learned shortlist at 20+ candidates; the experimental hypothesis scorer is not loaded. Score is categorical. SystemOne responds atomically; partial success is not part of the current API. The local model is a research release, not a production reliability guarantee.

## Gaps

The process-local limiter must be replaced or backed by a shared ingress limiter before multi-instance public deployment. No public host, TLS termination, or independent production model release was provided. Primitive selection was rechecked in the browser accessibility tree after correcting its group/pressed semantics. Full screen-reader, automated WCAG, and load testing were not run.

## Confidence

MEDIUM: strong local contract, failure, and real-model browser evidence; public infrastructure and model reliability remain unverified.
