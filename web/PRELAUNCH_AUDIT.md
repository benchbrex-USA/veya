# Veya website forensic pre-launch audit

## Production verdict

**Blocked for public production release.** The local website and real-model path work, but the served model is a research release, the website directory and its tests are untracked in Git, and no public deployment, shared limiter, or browser/CWV certification was available to verify. This audit does not promote model reliability.

## Issues found and fixed

- At 320 px, the horizontally scrolling primitive selector obscured SystemOne. It now shows all four primitives in a two-row grid; verified in the rendered browser.
- All six pages shared generic server-rendered metadata and had no canonical/sitemap path. Each route now gets its own title and description. An explicit HTTPS `VEYA_PUBLIC_ORIGIN` enables canonical and absolute OG URLs plus sitemap/robots; the local preview is intentionally disallowed from indexing. Public binding without that origin now fails startup.
- Unknown public routes returned raw JSON. They now render a branded, `noindex` 404 page; API/static 404s remain JSON.
- Every asset had `no-store`, forcing a fetch on every page visit. Static assets now cache for five minutes; HTML and API responses remain `no-store`.
- Routine client disconnects produced server tracebacks during the responsive sweep. The production server now suppresses only broken-pipe/connection-reset tracebacks.

## Checks run

- **Git:** `git status --short` shows `web/` and `tests/test_web_gateway.py` untracked, with substantial pre-existing modified/untracked research files. No commit or staging was performed.
- **Build:** no website build/bundler or public deployment manifest exists. The site runs directly from Python and six static assets totaling 2,591,870 bytes, including the supplied 2.5 MB lockup PNG. `pyproject.toml` packages `veya` and `veya.experimental`, not `web`; a Python wheel is not a website deployment artifact.
- **Static:** `python3 -m ruff check web tests/test_web_gateway.py` passed; `node --check` on both website JS modules passed; `py_compile` on the three web Python modules passed. `ruff check .` failed with 91 findings in research notebooks/scripts outside the website.
- **Tests:** final `python3 -m pytest -q` passed: 69 tests, one skipped, six subtests. `node --test tests/test_model_client.mjs` passed: six tests. The gateway tests exercised 400/404/429/5xx, timeout, model outage, malformed upstream response, request boundaries, malicious candidate text, response validation, metadata, SEO and cache headers.
- **Routes:** all six public routes rendered an H1 and valid assets. Automated DOM sweep of six routes at 320, 375, 430, 768, 1024, 1280, 1440 and 1920 px found no horizontal overflow. A first-load image observation at 320 px resolved after page readiness; all three logo images then reported complete with intrinsic width 500.
- **Browser:** visual inspection of desktop hero, 320/375 px hero/workspace, narrow result metadata, and 404 page. The in-app Chromium browser showed no console errors on the benchmark page. Safari/WebKit and Firefox were not available for testing.
- **Accessibility:** automated DOM checks on six routes found one main, one H1, a skip link, and no unnamed form controls/buttons/progress elements. Manual keyboard pass reached and activated the skip link; the model primitive controls expose pressed state. Forced reduced-motion mode reported a 0.01 ms hero animation and no hidden reveal content. No axe/WCAG or screen-reader audit was run; contrast was not instrumented.
- **Security:** browser assets contain no model token or mock adapter. Same-origin gateway, strict request/response schemas, 32 KB request limit, candidate limits, CSP/security headers, safe metadata logging, two-inflight cap and per-process rate limit were inspected and tested. Real local 429 returned structured JSON plus `Retry-After: 47`. Public ingress/authentication and multi-instance abuse control remain unverified.
- **Claims:** visible benchmark figures matched the sealed CLINC150 validation artifact: Recall@8 35.6%, end-to-end accuracy 34.9%, ECE 26.7%. Copy keeps the research-release scope, disabled automation, and categorical Score semantics. The served manifest does not load the experimental hypothesis scorer.
- **Performance:** no WebGL/canvas or external font load was found; static payload totals 77,980 bytes. Core Web Vitals (LCP, INP, CLS), 60 fps traces, slow-device and network-throttled profiles were not measured; the available browser automation did not expose performance timing APIs.

## Final real-model smoke

Against the restarted local gateway and audited model checkpoint:

| Case | HTTP | Result |
|---|---:|---|
| Choice, three candidates | 200 | `billing`, `packed`, probability sum 1.0000 |
| Choice, twenty candidates | 200 | `billing`, `shortlist_k8_packed_rerank`, eight shortlisted, sum 1.0001 |
| Noul | 200 | true probability 0.967 |
| Categorical Score | 200 | expected value 1.0171, distribution sum 1.0000 |
| SystemOne | 200 | `department` and `refund` IDs preserved |
| Unsupported Hindi state | 200 | `defer`, reason `unsupported_language` |
| Gateway rate limit | 429 | structured body and `Retry-After: 47` |

Fault injection for upstream 5xx, timeout and invalid responses used a controlled HTTP test service, not the production inference path. The production site has no mock fallback.

## Remaining blockers

1. The model release verdict in `docs/audit/release_verdict.md` says this checkpoint is not trustworthy as a release candidate; Noul and Score remain research only.
2. The website is untracked and has no reproducible public deployment artifact/configuration. No production URL, TLS, health orchestration, source-map policy, or public smoke target was provided.
3. The limiter is process-local and no public authentication/shared ingress policy was verified.
4. Safari/Firefox, automated WCAG/screen-reader, contrast, Core Web Vitals, slow-device, and load testing remain unverified.
5. Repository-wide lint fails outside the website scope.

## Brand follow-up

The user-supplied transparent lockup and mark replaced the previous website assets in the header, footer, hero, favicon, and social image. The public gateway now emits `model: "Veya Core"`; the internal checkpoint identifier is not exposed to the browser. The supplied lockup adds about 2.5 MB to the uncached static payload; image optimization remains a performance opportunity.
