/**
 * Request handling for the customer Worker `veya` (veya.<subdomain>.workers.dev).
 *
 * `veya` used to reverse-proxy everything to the web/server.py gateway on a Mac
 * (Tailscale Funnel). It now serves the same public contract from Cloudflare:
 *
 *   - the website (web/static) from Workers static assets, with the gateway's
 *     page routing, headers, robots/sitemap and 404 page;
 *   - GET /api/benchmark from the release artifact baked in at build time;
 *   - POST /api/infer, /api/health and /api/version backed by the production
 *     Worker `veya-production` through a service binding (VEYA_API). Inference
 *     calls /v1/decisions with a dedicated server-side project key
 *     (VEYA_SITE_KEY, project "veya-public-site"); browsers never see a key.
 *
 * Gateway behaviour kept: 32 KB request cap (counted on the streamed bytes),
 * application/json only, same-origin check (+ VEYA_ALLOWED_ORIGINS), weighted
 * per-IP rate limit (12 units / 60 s), the {code, detail, error, error_code,
 * request_id} error envelope, Retry-After, and every upstream failure is JSON.
 *
 * This module has no Workers-only imports so it is unit tested under Node
 * (tests/test_veya_site_edge.mjs). The deployed entrypoint is worker/veya-edge.mjs.
 */

export const MAX_REQUEST_BYTES = 32_768;
export const MAX_RESPONSE_BYTES = 524_288;
export const RATE_WINDOW_S = 60;
export const RATE_CAPACITY = 12;
export const DEFAULT_INFER_TIMEOUT_MS = 25_000;
export const DEFAULT_PROBE_TIMEOUT_MS = 3_000;
export const VERSION_CACHE_MS = 60_000;
export const UPSTREAM_ORIGIN = "https://veya-production.internal";
export const SUPPORTED_PRIMITIVES = ["choice", "noul", "score"];

const STATIC_SUFFIXES = new Set(["css", "js", "mjs", "json", "png", "svg"]);
const CONTENT_TYPES = {
  html: "text/html; charset=utf-8",
  css: "text/css; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  mjs: "text/javascript; charset=utf-8",
  json: "application/json; charset=utf-8",
  svg: "image/svg+xml",
  png: "image/png",
};

/** Same stable machine codes as web/server.py, plus the upstream codes we can see. */
export const ERROR_CODES = {
  invalid_input: "VALIDATION_ERROR",
  request_too_large: "VALIDATION_ERROR",
  model_rejected: "VALIDATION_ERROR",
  unsupported_primitive: "VALIDATION_ERROR",
  unsupported_language: "UNSUPPORTED_LANGUAGE",
  model_busy: "OVERLOADED",
  rate_limited: "RATE_LIMITED",
  timeout: "TIMEOUT",
  model_unavailable: "INTERNAL_ERROR",
  model_error: "INTERNAL_ERROR",
  invalid_upstream_response: "INTERNAL_ERROR",
  benchmark_unavailable: "INTERNAL_ERROR",
  method_not_allowed: "NOT_FOUND",
  not_found: "NOT_FOUND",
};

/** web/server.py headers_common(). */
export const SECURITY_HEADERS = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "strict-origin-when-cross-origin",
  "x-frame-options": "DENY",
  "strict-transport-security": "max-age=31536000; includeSubDomains",
  "permissions-policy": "camera=(), microphone=(), geolocation=()",
  "content-security-policy":
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
};

export class BodyTooLarge extends Error {}
export class InvalidLength extends Error {}
export class Timeout extends Error {}

function headersWith(cacheControl, extra) {
  const headers = new Headers(extra);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  headers.set("cache-control", cacheControl);
  return headers;
}

export function requestId() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** JSON response with the gateway's error envelope on every status >= 400. */
export function sendJson(status, payload, options = {}) {
  let body = payload;
  if (status >= 400) {
    const legacy = String(payload.code || "error");
    body = {
      ...payload,
      error: String(payload.detail || legacy),
      error_code: ERROR_CODES[legacy] || "INTERNAL_ERROR",
    };
  }
  const headers = headersWith("no-store", { "content-type": "application/json; charset=utf-8" });
  if (status >= 400 && typeof body.request_id === "string") headers.set("x-request-id", body.request_id);
  if (options.retryAfter !== undefined && options.retryAfter !== null) headers.set("retry-after", String(options.retryAfter));
  return new Response(JSON.stringify(body), { status, headers });
}

function withoutBody(response, method) {
  if (method !== "HEAD") return response;
  return new Response(null, { status: response.status, headers: response.headers });
}

/** Read at most `limit` bytes, checking Content-Length and then the stream itself. */
export async function readBodyLimited(request, limit = MAX_REQUEST_BYTES) {
  const declared = request.headers.get("content-length");
  if (declared !== null && declared !== "") {
    if (!/^\d+$/.test(declared)) throw new InvalidLength();
    if (Number(declared) > limit) throw new BodyTooLarge();
  }
  if (!request.body) return new Uint8Array(0);
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      try {
        await reader.cancel();
      } catch {
        // client already gone
      }
      throw new BodyTooLarge();
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

export async function withTimeout(ms, work) {
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Timeout()), ms);
  });
  try {
    return await Promise.race([work(), deadline]);
  } finally {
    clearTimeout(timer);
  }
}

function bounded(raw, fallback, min, max) {
  const value = Number(raw);
  if (raw === undefined || raw === null || raw === "" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

/** web/server.py _weight(): large Choice sets cost more rate-limit units. */
export function weight(payload) {
  let count = 0;
  try {
    if (payload.primitive === "choice") {
      const criteria = payload.criteria;
      count = Array.isArray(criteria) ? criteria.length : criteria && typeof criteria === "object" ? Object.keys(criteria).length : 0;
    } else if (payload.primitive === "systemone" && payload.questions && typeof payload.questions === "object") {
      for (const question of Object.values(payload.questions)) {
        if (question && question.type === "choice" && question.criteria && typeof question.criteria === "object") {
          count += Array.isArray(question.criteria) ? question.criteria.length : Object.keys(question.criteria).length;
        }
      }
    }
  } catch {
    count = 0;
  }
  return Math.min(RATE_CAPACITY, 1 + Math.ceil(Math.max(0, count - 19) / 20));
}

/**
 * Per-IP limiter. Uses the Workers rate-limit binding (VEYA_IP_RATE, 12 per 60 s,
 * one unit per call) when bound; otherwise an in-isolate sliding window with the
 * gateway's exact algorithm (tests, local dev). Returns seconds to retry, or 0.
 */
export function createLimiter(now = () => Date.now()) {
  const hits = new Map();
  function local(key, cost) {
    const at = now() / 1000;
    if (hits.size > 4096) {
      for (const [client, events] of hits) {
        if (!events.length || at - events[events.length - 1][0] >= RATE_WINDOW_S) hits.delete(client);
      }
    }
    const events = hits.get(key) || [];
    while (events.length && at - events[0][0] >= RATE_WINDOW_S) events.shift();
    const used = events.reduce((sum, [, units]) => sum + units, 0);
    if (used + cost > RATE_CAPACITY) {
      hits.set(key, events);
      return events.length ? Math.max(1, Math.ceil(RATE_WINDOW_S - (at - events[0][0]))) : RATE_WINDOW_S;
    }
    events.push([at, cost]);
    hits.set(key, events);
    return 0;
  }
  return async function charge(env, key, cost) {
    if (cost <= 0) return 0;
    const binding = env && env.VEYA_IP_RATE;
    if (binding && typeof binding.limit === "function") {
      for (let unit = 0; unit < cost; unit += 1) {
        const result = await binding.limit({ key });
        if (result && result.success === false) return RATE_WINDOW_S;
      }
      return 0;
    }
    return local(key, cost);
  };
}

export function pagesFrom(siteConfig) {
  const pages = new Map();
  for (const page of (siteConfig && siteConfig.pages) || []) pages.set(page.href, [page.title, page.description]);
  return pages;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");
}

/** web/server.py _page_html(). */
export function pageHtml(indexHtml, pages, path, publicOrigin = "") {
  const [title, description] = pages.get(path);
  let page = indexHtml.replaceAll("__VEYA_TITLE__", escapeHtml(title)).replaceAll("__VEYA_DESCRIPTION__", escapeHtml(description));
  let seo = "";
  if (publicOrigin) {
    const canonical = escapeHtml(publicOrigin + path);
    const image = escapeHtml(publicOrigin + "/static/veya-lockup.png");
    seo = `<link rel="canonical" href="${canonical}"><meta property="og:url" content="${canonical}">`;
    page = page.replace('content="/static/veya-lockup.png"', `content="${image}"`);
  }
  return page.replace("<!--VEYA_SEO-->", seo);
}

/** web/server.py GET /api/benchmark payload. */
export function benchmarkPayload(audit) {
  const choice = audit.choice;
  return {
    dataset: audit.dataset,
    n: audit.n,
    candidate_count: audit.candidate_count,
    release: audit.release,
    split_status: audit.status,
    architecture: audit.architecture,
    recall_at_8: choice.retrieval_recall_at_8.rate,
    conditional_accuracy: choice.conditional_accuracy_given_hit.rate,
    accuracy: choice.end_to_end_accuracy.rate,
    ece: choice.ece_end_to_end,
    retrieval_misses: choice.retrieval_misses,
    shortlist_hits: choice.retrieval_recall_at_8.count,
    correct_on_hit: choice.conditional_accuracy_given_hit.count,
    scoring_misses: choice.reranker_misses_with_target_present,
  };
}

const NOT_FOUND_PAGE =
  '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
  '<meta name="viewport" content="width=device-width,initial-scale=1">' +
  '<meta name="robots" content="noindex">' +
  "<title>Page not found — Veya</title>" +
  '<link rel="icon" href="/static/veya-symbol.png">' +
  '<link rel="stylesheet" href="/static/site.css"></head>' +
  '<body><main id="main"><section class="page-hero">' +
  '<span class="eyebrow">404 / Veya</span><h1>Page not found.</h1>' +
  "<p>This address is not part of the Veya site.</p>" +
  '<a class="button" href="/">Return home ↗</a>' +
  "</section></main></body></html>";

/** Upstream (veya-production) error code -> public gateway status/code. */
export function mapUpstreamFailure(status, body, retryHeader) {
  const code = body && typeof body.code === "string" ? body.code : "";
  const retry = bounded(retryHeader, null, 1, 3600);
  if (status === 429) {
    return { status: 429, code: "rate_limited", detail: "Too many model requests. Try again shortly.", retryAfter: retry || 60 };
  }
  if (status === 400 || status === 413 || status === 422) {
    return { status, passthrough: true };
  }
  if (code === "model_busy") {
    return { status: 503, code: "model_busy", detail: "Model is busy. Try again shortly.", retryAfter: retry || 2 };
  }
  if (status === 504 || code === "upstream_timeout") {
    return { status: 504, code: "timeout", detail: "The model did not answer in time. Try again shortly." };
  }
  // 401/403/404 mean this Worker is misconfigured (key, binding); never leak that detail.
  return { status: 503, code: "model_unavailable", detail: "Model service unavailable. Please retry shortly.", retryAfter: retry || 5 };
}

export function createHandler(site = {}, options = {}) {
  const indexHtml = site.indexHtml || "";
  const pages = pagesFrom(site.siteConfig);
  const benchmark = (() => {
    try {
      return site.benchmark ? benchmarkPayload(site.benchmark) : null;
    } catch {
      return null;
    }
  })();
  const charge = options.limiter || createLimiter(options.now);
  const log = options.log || ((entry) => console.log(JSON.stringify(entry)));
  let versionCache = { at: 0, payload: null };

  function upstream(env) {
    return env && env.VEYA_API && typeof env.VEYA_API.fetch === "function" ? env.VEYA_API : null;
  }

  async function probe(env) {
    const api = upstream(env);
    if (!api) return { ok: false, status: null, latency_ms: 0, error: "binding_missing" };
    const started = Date.now();
    try {
      const response = await withTimeout(bounded(env.VEYA_PROBE_TIMEOUT_MS, DEFAULT_PROBE_TIMEOUT_MS, 250, 30_000), () =>
        api.fetch(new Request(`${UPSTREAM_ORIGIN}/healthz`, { method: "GET" })),
      );
      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      return {
        ok: response.status === 200 && Boolean(body) && body.status === "ok",
        status: response.status,
        latency_ms: Date.now() - started,
        release: body && typeof body.release === "string" ? body.release : null,
      };
    } catch (error) {
      return { ok: false, status: null, latency_ms: Date.now() - started, error: error instanceof Timeout ? "timeout" : "error" };
    }
  }

  async function edgeHealth(request, env) {
    const result = await probe(env);
    return withoutBody(
      sendJson(result.ok ? 200 : 503, {
        edge: "ok",
        status: result.ok ? "ok" : "degraded",
        upstream: { service: "veya-production", ...result },
        site_key_configured: Boolean(env && env.VEYA_SITE_KEY),
        rate_limit: env && env.VEYA_IP_RATE ? "binding" : "isolate",
        ...(result.ok ? {} : { code: "model_unavailable", detail: "veya-production did not answer the health probe." }),
      }),
      request.method,
    );
  }

  async function apiHealth(request, env) {
    const result = await probe(env);
    if (!result.ok) return withoutBody(sendJson(503, { status: "unavailable", code: "model_unavailable" }), request.method);
    return withoutBody(
      sendJson(200, { status: "ok", backend: "cloudflare", upstream: "veya-production", release: result.release }),
      request.method,
    );
  }

  async function apiVersion(request, env) {
    const now = Date.now();
    if (versionCache.payload && now - versionCache.at < VERSION_CACHE_MS) {
      return withoutBody(sendJson(200, versionCache.payload), request.method);
    }
    const api = upstream(env);
    try {
      if (!api) throw new Error("binding missing");
      const response = await withTimeout(bounded(env.VEYA_PROBE_TIMEOUT_MS, DEFAULT_PROBE_TIMEOUT_MS, 250, 30_000), () =>
        api.fetch(new Request(`${UPSTREAM_ORIGIN}/v1/version`, { method: "GET" })),
      );
      if (response.status !== 200) throw new Error("upstream status");
      const source = await response.json();
      if (!source || typeof source.release !== "string") throw new Error("invalid version");
      const pinned = env.VEYA_RELEASE_ID && env.VEYA_RELEASE_ID === source.release;
      const payload = {
        model: "Veya Core",
        model_version: pinned && env.VEYA_MODEL_VERSION ? env.VEYA_MODEL_VERSION : source.release,
        release: source.release,
        checkpoint: typeof source.checkpoint_sha256 === "string" ? source.checkpoint_sha256 : undefined,
        architecture_version: pinned && env.VEYA_ARCHITECTURE_VERSION ? env.VEYA_ARCHITECTURE_VERSION : undefined,
        supported_languages: ["en"],
        supported_primitives: SUPPORTED_PRIMITIVES,
        release_status: source.release_claims && typeof source.release_claims === "object" ? source.release_claims : undefined,
      };
      versionCache = { at: now, payload: JSON.parse(JSON.stringify(payload)) };
      return withoutBody(sendJson(200, versionCache.payload), request.method);
    } catch {
      return withoutBody(sendJson(503, { status: "unavailable", code: "model_unavailable" }), request.method);
    }
  }

  async function staticAsset(request, env, name) {
    const suffix = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "";
    if (!name || name.includes("..") || name.startsWith("/") || !/^[A-Za-z0-9._\-/]+$/.test(name) || !STATIC_SUFFIXES.has(suffix)) {
      return sendJson(404, { code: "not_found" });
    }
    if (!env || !env.ASSETS || typeof env.ASSETS.fetch !== "function") return sendJson(404, { code: "not_found" });
    const asset = await env.ASSETS.fetch(new Request(new URL(`/${name}`, request.url), { method: "GET" }));
    if (asset.status !== 200) return sendJson(404, { code: "not_found" });
    const image = suffix === "png" || suffix === "svg";
    const headers = headersWith(image ? "public, max-age=300" : "no-store", { "content-type": CONTENT_TYPES[suffix] });
    const etag = asset.headers.get("etag");
    if (etag) headers.set("etag", etag);
    return new Response(request.method === "HEAD" ? null : asset.body, { status: 200, headers });
  }

  function html(status, body, method) {
    const headers = headersWith("no-store", { "content-type": CONTENT_TYPES.html });
    return new Response(method === "HEAD" ? null : body, { status, headers });
  }

  function robots(env, path, method) {
    const origin = String((env && env.VEYA_PUBLIC_ORIGIN) || "").replace(/\/+$/, "");
    if (path === "/robots.txt") {
      const body = origin ? `User-agent: *\nAllow: /\nSitemap: ${origin}/sitemap.xml\n` : "User-agent: *\nDisallow: /\n";
      return new Response(method === "HEAD" ? null : body, { status: 200, headers: headersWith("no-store", { "content-type": "text/plain; charset=utf-8" }) });
    }
    if (!origin) return withoutBody(sendJson(404, { code: "not_found" }), method);
    const body =
      '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' +
      [...pages.keys()].map((route) => `<url><loc>${escapeHtml(origin + route)}</loc></url>`).join("") +
      "</urlset>";
    return new Response(method === "HEAD" ? null : body, { status: 200, headers: headersWith("no-store", { "content-type": "application/xml; charset=utf-8" }) });
  }

  async function infer(request, env, ctx) {
    const id = requestId();
    const started = Date.now();
    const ip = request.headers.get("cf-connecting-ip") || "unknown";
    const fields = { event: "infer", request_id: id, primitive: null, status: 0, code: null, upstream_status: null };
    const done = (response) => {
      fields.status = response.status;
      fields.latency_ms = Date.now() - started;
      log(fields);
      return response;
    };
    let raw;
    try {
      raw = await readBodyLimited(request, MAX_REQUEST_BYTES);
    } catch (error) {
      if (error instanceof BodyTooLarge) {
        fields.code = "request_too_large";
        return done(sendJson(413, { code: "request_too_large", detail: "Request exceeds 32 KB.", request_id: id }));
      }
      fields.code = "invalid_input";
      return done(sendJson(400, { code: "invalid_input", detail: "Invalid Content-Length.", request_id: id }));
    }
    const invalid = (detail) => {
      fields.code = "invalid_input";
      return done(sendJson(400, { code: "invalid_input", detail, request_id: id }));
    };
    if (raw.byteLength === 0) return invalid("Request body is required.");
    const contentType = (request.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
    if (contentType !== "application/json") return invalid("Content-Type must be application/json.");
    const origin = request.headers.get("origin");
    const host = request.headers.get("host") || new URL(request.url).host;
    const allowed = new Set([`http://${host}`, `https://${host}`]);
    for (const item of String((env && env.VEYA_ALLOWED_ORIGINS) || "").split(",")) if (item.trim()) allowed.add(item.trim());
    if (origin && !allowed.has(origin)) return invalid("Cross-origin model requests are not allowed.");
    const limited = (retryAfter) => {
      fields.code = "rate_limited";
      return done(
        sendJson(429, { code: "rate_limited", detail: "Too many model requests. Try again shortly.", retry_after: retryAfter, request_id: id }, { retryAfter }),
      );
    };
    let retryAfter = await charge(env, ip, 1);
    if (retryAfter) return limited(retryAfter);
    let payload;
    try {
      payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
    } catch {
      return invalid("Request body must be valid JSON.");
    }
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return invalid("Request body must be a JSON object.");
    fields.primitive = typeof payload.primitive === "string" ? payload.primitive.slice(0, 20) : null;
    const cost = weight(payload);
    retryAfter = cost > 1 ? await charge(env, ip, cost - 1) : 0;
    if (retryAfter) return limited(retryAfter);
    const api = upstream(env);
    if (!api || !env.VEYA_SITE_KEY) {
      fields.code = "model_unavailable";
      return done(sendJson(503, { code: "model_unavailable", detail: "Model service unavailable. Please retry shortly.", request_id: id }, { retryAfter: 5 }));
    }
    let response;
    try {
      response = await withTimeout(bounded(env.VEYA_INFER_TIMEOUT_MS, DEFAULT_INFER_TIMEOUT_MS, 1_000, 120_000), () =>
        api.fetch(
          new Request(`${UPSTREAM_ORIGIN}/v1/decisions`, {
            method: "POST",
            headers: { authorization: `Bearer ${env.VEYA_SITE_KEY}`, "content-type": "application/json", "x-request-id": id },
            body: raw,
          }),
        ),
      );
    } catch (error) {
      if (error instanceof Timeout) {
        fields.code = "timeout";
        return done(sendJson(504, { code: "timeout", detail: "The model did not answer in time. Try again shortly.", request_id: id }));
      }
      fields.code = "model_unavailable";
      return done(sendJson(503, { code: "model_unavailable", detail: "Model service unavailable. Please retry shortly.", request_id: id }, { retryAfter: 5 }));
    }
    fields.upstream_status = response.status;
    fields.replica = response.headers.get("x-veya-replica");
    let text = "";
    let body = null;
    try {
      text = await response.text();
      if (text.length > MAX_RESPONSE_BYTES) throw new Error("too large");
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    if (response.status >= 200 && response.status < 300) {
      if (!body || typeof body !== "object" || Array.isArray(body)) {
        fields.code = "invalid_upstream_response";
        return done(sendJson(502, { code: "invalid_upstream_response", detail: "The model returned an invalid response.", request_id: id }));
      }
      fields.code = null;
      return done(new Response(text, { status: 200, headers: headersWith("no-store", { "content-type": "application/json; charset=utf-8" }) }));
    }
    const mapped = mapUpstreamFailure(response.status, body, response.headers.get("retry-after"));
    if (mapped.passthrough && body && typeof body === "object" && !Array.isArray(body)) {
      fields.code = typeof body.code === "string" ? body.code.slice(0, 40) : "invalid_input";
      const out = { ...body, code: body.code || "invalid_input", detail: body.detail || "The model rejected this input." };
      if (typeof out.request_id !== "string") out.request_id = id;
      return done(sendJson(mapped.status, out));
    }
    if (mapped.passthrough) {
      fields.code = "model_rejected";
      return done(sendJson(400, { code: "model_rejected", detail: "The model rejected this input.", request_id: id }));
    }
    fields.code = mapped.code;
    fields.upstream_code = body && typeof body.code === "string" ? body.code.slice(0, 40) : null;
    const out = { code: mapped.code, detail: mapped.detail, request_id: id };
    if (mapped.retryAfter) out.retry_after = mapped.retryAfter;
    return done(sendJson(mapped.status, out, { retryAfter: mapped.retryAfter }));
  }

  return async function handle(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    try {
      if (method === "POST") {
        if (path === "/api/infer") return await infer(request, env, ctx);
        return sendJson(404, { code: "not_found" });
      }
      if (method !== "GET" && method !== "HEAD") return sendJson(405, { code: "method_not_allowed", detail: "Method not allowed." });
      if (path === "/healthz") return withoutBody(sendJson(200, { status: "alive" }), method);
      if (path === "/__edge_health") return await edgeHealth(request, env);
      if (path === "/readyz" || path === "/startupz") {
        const result = await probe(env);
        return withoutBody(sendJson(result.ok ? 200 : 503, result.ok ? { status: "ready" } : { status: "not_ready", code: "model_unavailable" }), method);
      }
      if (path === "/robots.txt" || path === "/sitemap.xml") return robots(env, path, method);
      if (path === "/api/benchmark") {
        return withoutBody(benchmark ? sendJson(200, benchmark) : sendJson(503, { code: "benchmark_unavailable" }), method);
      }
      if (path === "/api/health") return await apiHealth(request, env);
      if (path === "/api/version") return await apiVersion(request, env);
      if (pages.has(path)) return html(200, pageHtml(indexHtml, pages, path, String((env && env.VEYA_PUBLIC_ORIGIN) || "").replace(/\/+$/, "")), method);
      if (path.startsWith("/static/")) return withoutBody(await staticAsset(request, env, path.slice("/static/".length)), method);
      if (path.startsWith("/api/")) return withoutBody(sendJson(404, { code: "not_found" }), method);
      return html(404, NOT_FOUND_PAGE, method);
    } catch {
      return sendJson(500, { code: "model_error", detail: "Unexpected edge error.", request_id: requestId() });
    }
  };
}
