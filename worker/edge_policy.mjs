/**
 * Edge checks for the Cloudflare Worker. This module does not call a container.
 *
 * POST /api/infer accepts only the demo credential. /__admin/version accepts
 * only the infrastructure credential. The isolate-local Map is a fallback for
 * local tests where the Cloudflare rate-limit binding is absent. It uses the
 * same demo identity, not the client IP. When the binding is present, only
 * that binding applies.
 */

export const INSTANCE_NAME = "veya-core";
export const INFER_MAX_BODY_BYTES = 32_768;
export const INFER_RATE_LIMIT = 20;
export const HEAVY_RATE_LIMIT = 6;
export const RATE_LIMIT = INFER_RATE_LIMIT;
export const RATE_WINDOW_MS = 60_000;
export const DEMO_IDENTITY = "veya-demo";

const PUBLIC_GET = new Set([
  "/__edge_health",
  "/healthz",
  "/readyz",
  "/startupz",
  "/version",
  "/api/version",
]);

const METHODS = new Set(["GET", "HEAD", "POST", "OPTIONS"]);

const HOP = [
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  // Set only by the production Worker after API-key authentication. A client
  // must never be able to choose the in-container rate-limit bucket.
  "x-veya-tenant",
];

export const SECURITY_HEADERS = {
  // HTTPS only for two years, including api./app./docs. subdomains. `preload` is
  // deliberately omitted until the apex is ready for the HSTS preload list.
  "strict-transport-security": "max-age=63072000; includeSubDomains",
  "cross-origin-opener-policy": "same-origin",
  "cross-origin-resource-policy": "same-origin",
  "content-security-policy":
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  "x-content-type-options": "nosniff",
  "referrer-policy": "strict-origin-when-cross-origin",
  "permissions-policy": "camera=(), microphone=(), geolocation=()",
  "x-frame-options": "DENY",
  "x-robots-tag": "noindex",
};

export function edgeHealth() {
  return json({ edge: "ok", backend: "cloudflare-container" }, 200);
}

/**
 * Container idle timeout. Configured per environment through the
 * VEYA_SLEEP_AFTER var (staging 15m, production 1h); "<n>s|m|h" between one
 * minute and 24 hours. Anything else falls back to the historical 8h.
 */
export const DEFAULT_SLEEP_AFTER = "8h";

export function sleepAfterFor(env) {
  const raw = env && typeof env.VEYA_SLEEP_AFTER === "string" ? env.VEYA_SLEEP_AFTER.trim() : "";
  const match = /^([1-9][0-9]{0,4})([smh])$/.exec(raw);
  if (!match) return DEFAULT_SLEEP_AFTER;
  const seconds = Number(match[1]) * { s: 1, m: 60, h: 3600 }[match[2]];
  if (seconds < 60 || seconds > 86_400) return DEFAULT_SLEEP_AFTER;
  return raw;
}

/** Seconds in a "<n>s|m|h" expression (the sleepAfter format); NaN if malformed. */
export function sleepAfterSeconds(expr) {
  const match = /^([0-9]+)([smh])$/.exec(String(expr));
  return match ? Number(match[1]) * { s: 1, m: 60, h: 3600 }[match[2]] : Number.NaN;
}

/**
 * How long the container may report requests in flight without any request
 * starting or finishing before the count is treated as leaked (see
 * worker/body_hygiene.mjs). Far above the 15 s edge deadline. VEYA_INFLIGHT_STALE_MS,
 * 30 s to 1 h, default 2 min.
 */
export const DEFAULT_INFLIGHT_STALE_MS = 120_000;

export function inflightStaleMs(env) {
  const raw = env && env.VEYA_INFLIGHT_STALE_MS;
  const value = Number(raw);
  if (raw === undefined || raw === null || raw === "" || !Number.isFinite(value)) return DEFAULT_INFLIGHT_STALE_MS;
  return Math.min(3_600_000, Math.max(30_000, Math.floor(value)));
}

/**
 * Optional container env passed through from Worker vars. Only VEYA_MAX_INFLIGHT
 * (gateway admission limit per replica, 1..8) today; release pins are never
 * taken from here.
 */
export function containerEnvOverrides(env) {
  const out = {};
  const raw = env && typeof env.VEYA_MAX_INFLIGHT === "string" ? env.VEYA_MAX_INFLIGHT.trim() : "";
  if (/^[1-8]$/.test(raw)) out.VEYA_MAX_INFLIGHT = raw;
  return out;
}

export function guardRequest(request, env, buckets, now = Date.now()) {
  const url = new URL(request.url);
  if (!METHODS.has(request.method)) {
    return json({ code: "method_not_allowed", detail: "Method not allowed." }, 405);
  }
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: SECURITY_HEADERS });
  }
  const length = contentLength(request);
  if (length !== null && length > INFER_MAX_BODY_BYTES) {
    return json({ code: "request_too_large", detail: "Request exceeds the size limit." }, 413);
  }
  if (isPublic(request.method, url.pathname)) {
    if (url.pathname === "/__edge_health") return edgeHealth();
    return null;
  }
  const token = expectedToken(env, url.pathname);
  if (typeof token !== "string" || token.length < 16) {
    return json(
      { code: "auth_not_configured", detail: "Inference authentication is not configured." },
      503,
    );
  }
  if (!bearerMatches(request.headers.get("authorization"), token)) {
    return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
  }
  if (!cloudflareLimiter(env)) {
    const limited = rateLimit(limitIdentity(url.pathname), buckets, now, INFER_RATE_LIMIT);
    if (limited) return limited;
  }
  return null;
}

export function limitIdentity(pathname) {
  return pathname === "/api/infer" ? DEMO_IDENTITY : "veya-admin";
}

export function notReady(status) {
  if (status === 200) return null;
  return json(
    { code: "not_ready", detail: "Veya Core is not ready." },
    503,
    { "retry-after": "5" },
  );
}

export function forwardHeaders(request, requestId) {
  const headers = new Headers(request.headers);
  for (const name of HOP) headers.delete(name);
  headers.delete("authorization");
  headers.set("accept-encoding", "identity");
  headers.set("x-request-id", requestId);
  return headers;
}

export function requestId(request) {
  const current = request.headers.get("x-request-id") || "";
  if (/^[A-Za-z0-9._:-]{8,80}$/.test(current)) return current;
  return crypto.randomUUID();
}

export function failure(requestIdValue, status, code) {
  return json(
    { code, detail: "The Veya container did not respond.", request_id: requestIdValue },
    status,
  );
}

export function stamp(response) {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function isPublic(method, pathname) {
  return (method === "GET" || method === "HEAD") && PUBLIC_GET.has(pathname);
}

function expectedToken(env, pathname) {
  if (!env) return undefined;
  return pathname === "/api/infer" ? env.VEYA_DEMO_TOKEN : env.VEYA_EDGE_TOKEN;
}

function contentLength(request) {
  const raw = request.headers.get("content-length");
  if (raw === null || raw === "") return null;
  if (!/^\d+$/.test(raw)) return INFER_MAX_BODY_BYTES + 1;
  return Number(raw);
}

function bearerMatches(header, expected) {
  if (typeof header !== "string" || !header.startsWith("Bearer ")) return false;
  const presented = header.slice("Bearer ".length);
  if (presented.length !== expected.length) return false;
  let diff = 0;
  for (let index = 0; index < presented.length; index += 1) {
    diff |= presented.charCodeAt(index) ^ expected.charCodeAt(index);
  }
  return diff === 0;
}

function cloudflareLimiter(env) {
  const binding = env && env.VEYA_RATE_INFERENCE;
  return Boolean(binding && typeof binding.limit === "function");
}

function rateLimit(key, buckets, now, limit) {
  const bucket = buckets.get(key) || { start: now, count: 0 };
  if (now - bucket.start >= RATE_WINDOW_MS) {
    bucket.start = now;
    bucket.count = 0;
  }
  bucket.count += 1;
  buckets.set(key, bucket);
  if (bucket.count > limit) {
    return json({ code: "rate_limited", detail: "Too many requests." }, 429, {
      "retry-after": "60",
    });
  }
  return null;
}

export async function requestWeight(request) {
  const url = new URL(request.url);
  if (isPublic(request.method, url.pathname)) return "public";
  if (request.method !== "POST" || url.pathname !== "/api/infer") return "inference";
  try {
    const body = JSON.parse(await request.clone().text());
    if (body && body.primitive && body.primitive !== "choice") return "heavy";
    const criteria = body && body.criteria;
    if (
      criteria &&
      typeof criteria === "object" &&
      !Array.isArray(criteria) &&
      Object.keys(criteria).length >= 20
    ) {
      return "heavy";
    }
  } catch {
    return "inference";
  }
  return "inference";
}

export async function edgeRateLimit(request, env) {
  const weight = await requestWeight(request);
  if (weight === "public") return null;
  const key = limitIdentity(new URL(request.url).pathname);
  if (weight === "heavy") {
    const heavy = await consumeLimit(env && env.VEYA_RATE_HEAVY, key);
    if (heavy) return heavy;
  }
  return consumeLimit(env && env.VEYA_RATE_INFERENCE, key);
}

export function adminVersion(env, build = {}) {
  const metadata = (env && env.CF_VERSION_METADATA) || {};
  const dirty = build.sourceDirty ?? (env && env.VEYA_SOURCE_DIRTY);
  return json(
    {
      worker_version_id: metadata.id || null,
      source_commit: build.sourceCommit || (env && env.VEYA_SOURCE_COMMIT) || null,
      source_dirty: dirty === true || dirty === "true",
      container_image_digest: env && env.VEYA_IMAGE_DIGEST ? env.VEYA_IMAGE_DIGEST : null,
      release: build.release || (env && env.VEYA_RELEASE_ID) || null,
      manifest_sha256: build.manifestSha256 || (env && env.VEYA_MANIFEST_SHA256) || null,
      checkpoint_sha256: build.checkpointSha256 || (env && env.VEYA_CHECKPOINT_SHA256) || null,
    },
    200,
  );
}

async function consumeLimit(binding, key) {
  if (!binding || typeof binding.limit !== "function") return null;
  const result = await binding.limit({ key });
  if (result && result.success === false) {
    return json({ code: "rate_limited", detail: "Too many requests." }, 429, {
      "retry-after": "60",
    });
  }
  return null;
}

function json(body, status, extra) {
  const headers = new Headers(extra);
  headers.set("content-type", "application/json");
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  return new Response(JSON.stringify(body), { status, headers });
}
