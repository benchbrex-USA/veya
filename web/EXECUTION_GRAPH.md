# Website execution graph

Status is based on observable evidence from this working tree. The website is a local research experience; no production deployment is claimed.

| ID | Objective | Dependencies | Acceptance | Status | Evidence / blocker |
|---|---|---|---|---|---|
| WEB-G0 | Inspect repository and served model | none | Real API and release facts identified | complete | `veya/server.py`, audited `docs/audit/release_verdict.md`, manifest |
| WEB-G1 | Integrate canonical identity | G0 | Supplied mark used without redrawing | complete | supplied `web/static/veya-lockup.png`, `veya-symbol.png` |
| WEB-G2 | Information architecture | G0 | All navigation destinations contain content | complete | Five routes verified in browser |
| WEB-G3 | Design system | G1 | Palette, type, layout and components render | complete | `web/static/site.css`; mobile and desktop browser review |
| WEB-G4 | Motion and reduced motion | G3 | Purposeful entry motion, reduced motion override | complete | CSS keyframes and `prefers-reduced-motion` rule |
| WEB-G5 | Shell and hero | G2-G4 | Responsive navigation and homepage | complete | Browser homepage screenshot and accessibility tree |
| WEB-G6 | Playground and primitives | G0,G5 | Choice, Noul, Score, SystemOne run real requests | complete | Live browser responses for all four primitives |
| WEB-G7 | Probability, defer, high-N | G6 | Truthful values, unsupported defer, 20-way Choice | complete | Native progress values, language gate and shortlist response verified in browser |
| WEB-G8 | Gateway and security | G0,G6 | Schema bounds, timeouts, rate limit, safe error mapping | complete | `web/server.py`; 5 targeted tests passed |
| WEB-G9 | Architecture and benchmark pages | G0,G2 | Architecture matches audit; metrics from sealed artifact | complete | Browser architecture/benchmark views; `/api/benchmark` reads sealed result |
| WEB-G10 | Developer guidance | G8 | Real request example and local run instructions | complete | `/developers`, `web/README.md` |
| WEB-G11 | Accessibility and responsive QA | G5-G7 | Keyboard semantics, live results, mobile/desktop layout | active | Browser AX and viewport checks passed; full WCAG and screen-reader audit remains |
| WEB-G12 | Production release | G0-G11 | Public hosting, TLS, shared rate limit, release gate | blocked | No deployment target, domain, or public model infrastructure provided; audited model is research release |

## Cinematic experience pass

| ID | Objective | Dependencies | Acceptance | Status | Evidence / blocker |
|---|---|---|---|---|---|
| MOTION-G0 | Audit rendered product | none | Homepage and playground visually reviewed | complete | Browser screenshots of existing mobile hero and workspace |
| MOTION-G1 | Visual hierarchy | G0 | Headline, model path and CTA remain clear | complete | Mobile and desktop rendered review |
| MOTION-G2 | Typography and grid | G1 | Editorial hierarchy without viewport overflow | complete | Browser width checks at 375, 768, 1024, 1440 |
| MOTION-G3 | Hero choreography | G1,G2 | Points and axes resolve into canonical mark | complete | Rendered mobile and desktop hero reviewed |
| MOTION-G4 | Symbol primitives | G3 | Geometry maps to Veya points and axes | complete | Hero and processing symbol observed in browser |
| MOTION-G5 | Section transitions | G2 | Reveals follow reading order without scroll hijack | complete | Homepage scroll inspection; IntersectionObserver only |
| MOTION-G6 | Playground motion | G4 | Loading and result states remain truthful | complete | Live Choice result and native progress values inspected |
| MOTION-G7 | Architecture animation | G4 | Branches reflect audited serving path | complete | Architecture map rendered; `/version` metadata surfaced |
| MOTION-G8 | Mobile simplification | G3-G7 | No pointer effects or expensive scene on touch | complete | 375px mobile review and pointer media gate |
| MOTION-G9 | Reduced motion | G3-G7 | Direct canonical mark, no ambient sequence | complete | `?motion=reduce` rendered at 375px; OS media query shares equivalent CSS |
| MOTION-G10 | Performance | G3-G9 | No heavy motion runtime or scroll work | active | No motion dependency added. Uncompressed CSS 13,341→24,672 bytes; JS 23,645→27,408 bytes. CPU/LCP/INP profiling and slow-device simulation unavailable in this browser runtime. |
| MOTION-G11 | Visual regression QA | G3-G9 | Review key viewport and result states | complete | Homepage mobile/desktop, architecture, Choice result, reduced mode observed; width checks show no horizontal overflow |

## Real model experience pass

| ID | Objective | Dependencies | Acceptance | Status | Evidence / blocker |
|---|---|---|---|---|---|
| MODELWEB-G0 | Discover runtime contract | none | Actual handlers and release inspected | complete | `veya/server.py`, `veya/agent.py`, `veya/release_bundle.py`, manifest |
| MODELWEB-G1 | Typed client | G0 | One browser client serializes all primitives | complete | `web/static/model-client.mjs`; 6 Node tests |
| MODELWEB-G2 | Runtime schemas | G0 | Requests and responses checked at boundary | complete | `web/contracts.py`; malformed upstream and partial SystemOne tests |
| MODELWEB-G3 | Secure gateway | G1,G2 | Same-origin validation, model proxy, no browser credential | complete | `web/server.py`; HTTP integration tests and source inspection |
| MODELWEB-G4 | Choice | G3 | Real Choice and distribution, metadata | complete | Browser E2E: billing, sum 0.9999, packed route |
| MODELWEB-G5 | Noul | G3 | Real proposition probability | complete | Browser E2E: 84.5% true |
| MODELWEB-G6 | Score | G3 | Categorical expected score and submitted labels | complete | Browser E2E: 1.80/2 with level probabilities |
| MODELWEB-G7 | SystemOne | G3 | Question IDs preserved across results | complete | Browser E2E: three associated results; contract test rejects partial IDs |
| MODELWEB-G8 | Error/defer/unsupported | G2,G3 | Explicit outcomes, no mock fallback | complete | Browser unsupported-language defer and stopped-model outage; HTTP timeout/5xx tests |
| MODELWEB-G9 | Version | G0,G3 | Release and route metadata in technical panel | complete | Browser drawer showed version 0.3.5 and audited release |
| MODELWEB-G10 | Abuse/rate limit | G3 | Bounded inputs, weighted limit, concurrency cap | complete | HTTP tests for 413, 429 with Retry-After, high-N cost, cross-origin rejection |
| MODELWEB-G11 | Privacy-safe telemetry | G3 | No raw user text in log | complete | Integration test captures logged metadata and asserts raw input absent |
| MODELWEB-G12 | Accessibility | G4-G8 | Labels, live result, keyboard-reachable controls | active | Browser accessibility tree checked after primitive selector semantics fix; formal screen-reader/automated WCAG audit pending |
| MODELWEB-G13 | Integration/E2E | G4-G12 | Browser pathways and HTTP failure paths pass | complete | Automated browser assertions plus 11 Python and 6 Node tests |
| MODELWEB-G14 | Production smoke | G13 | Real model through final gateway after restart | complete | Audited model restarted; final browser Choice result billing, probability sum 0.9999 |
