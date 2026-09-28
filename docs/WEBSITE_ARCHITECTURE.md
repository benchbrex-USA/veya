# Veya website architecture

The website is a small Python HTTP gateway plus browser-native ES modules. It has no frontend framework or bundler. `python3 -m web.server` serves the HTML shell and static modules; the browser calls only the same-origin gateway. The gateway calls the separate Veya model service. There is no production mock fallback.

## Ownership map

| Concern | Primary file |
| --- | --- |
| Route metadata, ordering, primary navigation, descriptions | `web/static/site-config.json` |
| HTML shell, logo placement, static imports | `web/static/index.html` |
| Brand and playground styles | `web/static/site.css` |
| Documentation tokens, layout, and code | `web/static/docs.css` |
| Research homepage typography, sections, and breakpoints | `web/static/home.css` |
| Route bootstrap and global navigation | `web/static/site.js` |
| Documentation layout, sidebar, TOC, search, live metadata | `web/static/docs.mjs` |
| Documentation text and page sections | `web/static/content.mjs` |
| Homepage prose and section data | `web/static/home-content.mjs` |
| Homepage composition | `web/static/home.mjs` |
| Interactive Choice route schematic and observed metadata | `web/static/architecture-explorer.mjs`, `web/static/architecture-explorer.css` |
| Request and quickstart examples | `web/static/examples.mjs` |
| Shared badges, callouts, code tabs, tables, flows, metrics | `web/static/ui.mjs` |
| Live primitive forms and result display | `web/static/playground.mjs` |
| Browser model client and response parsing | `web/static/model-client.mjs` |
| Gateway, SEO shell, rate limit, timeout, static serving | `web/server.py` |
| Gateway schemas | `web/contracts.py` |
| Model service | `web/model_server.py` |
| Official brand image assets | `web/static/veya-lockup.png`, optimized display derivative `veya-lockup-display.png`, `web/static/veya-symbol.png` |

The supplied PNG logos are currently used. A 700px display derivative reduces the rendered lockup transfer from 2.4 MB to 55 KB; the original remains in the repository for source fidelity and social metadata. The earlier supplied SVG path is absent from the current workspace; use the official SVG directly if it is restored, without tracing or redrawing it. The lockup is intentionally kept as a single asset instead of recreating its lettering in CSS.

## Stable contracts

`site-config.json` is the route registry. Each page has `href`, `title`, `description`, `section`, `order`, `kind`, and `navLabel`. `primaryNavigation` holds the focused header ordering; `footerNavigation` holds the fuller footer list. The server uses the same registry for route resolution, titles, descriptions, and sitemap. Documentation sidebar and search use it too.

Documentation content uses structured page objects in `content.mjs`: `{ category, heading, summary, status, sections }`. A section has a title and typed blocks. `renderSection` in `ui.mjs` currently accepts `p`, `list`, `code`, `codeTabs`, `callout`, `flow`, `table`, `links`, and `metrics`. Reuse a block or add a focused variant; do not fork a page renderer. Content strings are escaped before insertion into HTML.

The model path is browser → `VeyaClient` → `/api/infer` → gateway validation/rate limit/timeout → Veya service. `/api/version` and `/api/health` expose bounded operational metadata. `/api/benchmark` serves an audited aggregate artifact. The homepage and benchmark page read Recall@8, conditional accuracy, end-to-end accuracy, and ECE from that endpoint. `playground.mjs` mounts the same live experience on the homepage and `/experience`; it does not invent reasoning or model output.

`site.css` defines base brand colors (`--blue`, `--ivory`, etc.) and shared result graphics. `home.css` owns the homepage's editorial section system, the CSS/SVG decision-field hero, and responsive treatment. `architecture-explorer.css` owns the Choice route schematic. `docs.css` defines the documentation tokens for surfaces, type spacing, sidebar and content widths, border/radius, and transition speeds. Edit the appropriate token for a shared style change. The documentation layout has a sidebar, article, and on-page TOC at desktop widths; the TOC hides on medium screens; the sidebar becomes a button-controlled drawer on mobile. Reduced motion disables nonessential transitions. The hero resolves to the unmodified supplied symbol asset; decorative paths are separate SVG/CSS geometry.

## Routine changes

| Wanted change | Primary location |
| --- | --- |
| Brand color | `site.css` base token, then check dependent `docs.css` token |
| Logo | Official asset in `web/static/`, then `index.html` references |
| Header/footer navigation item | `site-config.json` `primaryNavigation` or `footerNavigation`, and target page metadata |
| Homepage copy | `home-content.mjs` for data; `home.mjs` for composed text |
| Homepage layout | `home.css`, then check 320px through desktop widths |
| Documentation text | `content.mjs` page entry |
| Benchmark number | Audited aggregate artifact behind `/api/benchmark`; do not type it into several pages |
| Model status | Audited release manifest/service metadata; update `content.mjs` only after checking evidence |
| API endpoint or schema | `contracts.py`, `server.py`, `model-client.mjs`, focused tests |
| Sidebar width | `docs.css` `--docs-sidebar-width` |
| Code theme | `.docs-code` rules in `docs.css` |
| Playground preset | `examples.mjs` `playgroundExamples` |
| Request example | `examples.mjs` |

To add a page: add one metadata entry in `site-config.json`, add one structured page in `content.mjs`, and add the route to `primaryNavigation` or `footerNavigation` only if it belongs there. The server derives its route and sitemap from the registry. Check the rendered page and navigation afterward.

The framework choice, route registry, content architecture, token layer, client/gateway separation, and keyboard/focus patterns are protected foundations. Ordinary content or styling requests should not replace them.
