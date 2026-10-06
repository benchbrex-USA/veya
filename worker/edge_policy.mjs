/**
 * Edge checks for the Cloudflare Worker. This module does not call a container.
 */

export const INSTANCE_NAME = "veya-core";
export const MAX_BODY_BYTES = 4 * 1024 * 1024;
export const RATE_LIMIT = 60;
export const RATE_WINDOW_MS = 60_000;

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
];

export function edgeHealth() {
  return json({ edge: "ok", backend: "cloudflare-container" }, 200);
}

export function guardRequest(request, env, buckets, now = Date.now()) {
  const url = new URL(request.url);
  if (!METHODS.has(request.method)) {
    return json({ code: "method_not_allowed", detail: "Method not allowed." }, 405);
  }
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204 });
  }
  const length = contentLength(request);
  if (length !== null && length > MAX_BODY_BYTES) {
    return json({ code: "request_too_large", detail: "Request exceeds the size limit." }, 413);
  }
  if (isPublic(request.method, url.pathname)) {
    if (url.pathname === "/__edge_health") return edgeHealth();
    return null;
  }
  const token = env && env.VEYA_EDGE_TOKEN;
  if (typeof token !== "string" || token.length < 16) {
    return json(
      { code: "auth_not_configured", detail: "Inference authentication is not configured." },
      503,
    );
  }
  if (!bearerMatches(request.headers.get("authorization"), token)) {
    return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
  }
  const limited = rateLimit(clientKey(request), buckets, now);
  if (limited) return limited;
  return null;
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

function isPublic(method, pathname) {
  return (method === "GET" || method === "HEAD") && PUBLIC_GET.has(pathname);
}

function contentLength(request) {
  const raw = request.headers.get("content-length");
  if (raw === null || raw === "") return null;
  if (!/^\d+$/.test(raw)) return MAX_BODY_BYTES + 1;
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

function clientKey(request) {
  return request.headers.get("cf-connecting-ip") || "local";
}

function rateLimit(key, buckets, now) {
  const bucket = buckets.get(key) || { start: now, count: 0 };
  if (now - bucket.start >= RATE_WINDOW_MS) {
    bucket.start = now;
    bucket.count = 0;
  }
  bucket.count += 1;
  buckets.set(key, bucket);
  if (bucket.count > RATE_LIMIT) {
    return json({ code: "rate_limited", detail: "Too many requests." }, 429, {
      "retry-after": "60",
    });
  }
  return null;
}

export async function requestWeight(request) {
  const url = new URL(request.url);
  if (isPublic(request.method, url.pathname)) return "public";
  if (url.pathname.startsWith("/api/quantum") || url.pathname === "/api/experts/route") {
    return "heavy";
  }
  if (request.method !== "POST") return "inference";
  const length = contentLength(request);
  if (length !== null && length > 65536) return "heavy";
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
  const key = clientKey(request);
  if (weight === "heavy") {
    const heavy = await consumeLimit(env && env.VEYA_RATE_HEAVY, key);
    if (heavy) return heavy;
  }
  return consumeLimit(env && env.VEYA_RATE_INFERENCE, key);
}

export function adminVersion(env) {
  const metadata = (env && env.CF_VERSION_METADATA) || {};
  return json(
    {
      worker_version_id: metadata.id || null,
      source_commit: env && env.VEYA_SOURCE_COMMIT ? env.VEYA_SOURCE_COMMIT : null,
      source_dirty: env && env.VEYA_SOURCE_DIRTY === "true",
      container_image_digest: env && env.VEYA_IMAGE_DIGEST ? env.VEYA_IMAGE_DIGEST : null,
      release: env && env.VEYA_RELEASE_ID ? env.VEYA_RELEASE_ID : null,
      manifest_sha256: env && env.VEYA_MANIFEST_SHA256 ? env.VEYA_MANIFEST_SHA256 : null,
      checkpoint_sha256: env && env.VEYA_CHECKPOINT_SHA256 ? env.VEYA_CHECKPOINT_SHA256 : null,
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
  return new Response(JSON.stringify(body), { status, headers });
}
