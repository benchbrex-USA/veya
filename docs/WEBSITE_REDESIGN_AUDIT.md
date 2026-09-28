# Website redesign audit · 2026-09-23

## Starting point

The existing website had 15 registered public routes, a documentation sidebar/search/TOC, native ES modules, a real same-origin inference gateway, and a shared Choice/Noul/Score/SystemOne playground. The prior homepage was documentation-led and visually compact. `WEBSITE_ARCHITECTURE.md` remains the ownership map.

## Design direction

The redesign uses an editorial research story around the supplied Veya mark, `#03008B` blue, and `#FDFBF2` ivory. Homepage sections own the large type and visual pacing; documentation and the playground retain their utility layout. The mark's four points and axes provide the hero motion; there is no WebGL, font download, or third-party animation package.

## Evidence sources

- The served architecture and release status were checked against `artifacts/release/manifest.json` and the active `/api/version` response.
- The four displayed benchmark values are served from `artifacts/release/blind_clinc150_val_result.json` through `/api/benchmark`.
- The 98.03% conditional figure is explicitly scoped to the 1,068 cases where the target reached the shortlist. Blind end-to-end accuracy is 34.9% across all 3,000 examples. Automation remains disabled.

## Rendered QA

The homepage was visually inspected at 1440, 1280, 1024, 768, 430, 375, and 320 pixels. Hero, comparison, principles, live form, evidence, architecture, developer example, FAQ, and footer were viewed in the browser. All 15 public routes were loaded at 320 pixels: one H1 each, no broken images, no horizontal document overflow, and no browser console errors in that pass. The 320-pixel documentation drawer and benchmark page were inspected. Reduced-motion mode rendered the final mark without entrance/reveal animation.

The shared live instrument returned real local service responses for Choice, 20-candidate Choice, Noul, Score, SystemOne, and unsupported-language deferral. The homepage Choice example returned `billing`; high-N Choice showed eight shortlisted candidates from 20 submitted.

## Performance check

One local warm-cache Chromium run with a temporary, subsequently removed `PerformanceObserver` probe measured LCP 108 ms, INP 48 ms, CLS 0 at 1440 pixels. At 320 pixels it measured LCP 156 ms, INP 48 ms, CLS 0. These are local development measurements, not production field data or throttled-device results. The homepage HTML response measured 0.000638 s TTFB locally. Homepage-specific CSS is 14,831 bytes, composition JavaScript 11,920 bytes, narrative data 4,020 bytes; the optimized official lockup is 56,537 bytes and the symbol is 23,976 bytes. There is no baseline Web Vitals capture from before the redesign, so no before/after speed claim is made.

## Remaining validation boundary

Safari/Firefox, throttled devices, and production field Web Vitals were not measured in this local pass. A setuptools license metadata deprecation warning appeared during a successful wheel build. The broader repository has pre-existing uncommitted and untracked work; this redesign did not reset it.

## Visual refinement pass

The generic circular hero orbit was replaced with a CSS/SVG decision field. Four decorative alternatives trace toward the exact supplied Veya symbol, which remains an image asset and is never deformed. The field enters once; reduced motion shows the resolved state immediately. The same symbol now marks homepage section labels, documentation categories, the standalone playground header, and its empty result state. Choice probability bars were given a stronger ruled visual treatment across the live instrument. Dead pointer-follow animation code was removed from the page bootstrap.

The homepage now includes an artifact-backed, proportional breakdown of 3,000 sealed blind examples into 1,932 retrieval misses, 21 scoring misses after retrieval, and 1,047 correct end-to-end decisions. Those counts are read from `/api/benchmark` and withheld if the aggregate fails validation. The new route inspector compares the served packed and K=8 shortlist paths, then displays route metadata from the last real Choice response. The hero CTA loads and runs a real Choice example through the existing gateway.

This pass was inspected in the live browser at 320, 375, 768, 1024, and desktop widths. At 320, 375, 768, 1024, and 1440 pixels, document width matched viewport width. Forced reduced motion had no hero-field or path animation and used normal scrolling. Local Choice and 20-candidate Choice both returned actual model responses; the route inspector changed from packed to `shortlist_k8_packed_rerank` based on the latter response. The documented Web Vitals above predate this visual refinement and should not be read as a new measurement.
