/**
 * Isolated production-candidate edge.
 *
 * Two identical Veya Core replicas share one R2 checkpoint. Customer calls use
 * D1 API keys. Specialist models are not enabled.
 */

import { authenticate, bearer } from "./api_keys.mjs";
import {
  INFER_MAX_BODY_BYTES,
  SECURITY_HEADERS,
  adminVersion,
  failure,
  forwardHeaders,
  requestId,
  stamp,
} from "./edge_policy.mjs";
import { fetchFromPool } from "./pool.mjs";
import { RELEASE_ID, SOURCE_COMMIT, SOURCE_DIRTY, MANIFEST_SHA256, CHECKPOINT_SHA256 } from "./build-meta.mjs";

export { ContainerProxy, VeyaContainer } from "./veya-container.mjs";

const BUILD = {
  sourceCommit: SOURCE_COMMIT,
  sourceDirty: SOURCE_DIRTY,
  release: RELEASE_ID,
  manifestSha256: MANIFEST_SHA256,
  checkpointSha256: CHECKPOINT_SHA256,
};

const PUBLIC = new Set(["/healthz", "/readyz", "/startupz", "/version", "/api/version", "/__edge_health", "/v1/version", "/v1/models"]);

function json(body, status, extra) {
  const headers = new Headers(extra);
  headers.set("content-type", "application/json");
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  return new Response(JSON.stringify(body), { status, headers });
}

function models() {
  return json({
    release: RELEASE_ID,
    checkpoint_sha256: CHECKPOINT_SHA256,
    enabled: ["veya-core"],
    specialists: [],
    benchbrex_tuned: false,
    release_claims: {
      english_choice: "RESEARCH RELEASE",
      english_noul: "RESEARCH ONLY",
      english_score_categorical: "RESEARCH ONLY",
      automation: "DISABLED",
    },
  }, 200);
}

async function recordUsage(env, identity, fields) {
  if (!env.VEYA_DB) return false;
  await env.VEYA_DB.prepare(
    `INSERT INTO usage_ledger (
      id, organization_id, project_id, api_key_id, endpoint, primitive, task,
      selected_model, route, http_status, latency_ms, release, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).bind(
    crypto.randomUUID(),
    identity.organizationId,
    identity.projectId,
    identity.keyId,
    fields.endpoint,
    fields.primitive || null,
    fields.task || null,
    fields.selectedModel || "veya-core",
    fields.route || null,
    fields.status,
    fields.latencyMs,
    RELEASE_ID,
    new Date().toISOString(),
  ).run();
  if (env.VEYA_USAGE && typeof env.VEYA_USAGE.writeDataPoint === "function") {
    env.VEYA_USAGE.writeDataPoint({
      indexes: [identity.organizationId],
      blobs: [identity.projectId, identity.keyId, fields.endpoint, fields.primitive || fields.task || "", fields.route || "", String(fields.status)],
      doubles: [fields.latencyMs],
    });
  }
  return true;
}

async function limited(env, keyId, heavy) {
  const binding = heavy ? env.VEYA_RATE_HEAVY : env.VEYA_RATE_INFERENCE;
  if (!binding || typeof binding.limit !== "function") return null;
  const result = await binding.limit({ key: keyId });
  if (result && result.success === false) {
    return json({ code: "rate_limited", detail: "Too many requests." }, 429, { "retry-after": "60" });
  }
  return null;
}

export default {
  async fetch(request, env) {
    const id = requestId(request);
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: SECURITY_HEADERS });
    if (request.method === "GET" && url.pathname === "/__admin/version") {
      const token = env && env.VEYA_EDGE_TOKEN;
      const presented = bearer(request.headers.get("authorization"));
      if (!token || presented.length !== token.length) return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
      let diff = 0;
      for (let index = 0; index < presented.length; index += 1) diff |= presented.charCodeAt(index) ^ token.charCodeAt(index);
      if (diff !== 0) return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
      return adminVersion(env, BUILD);
    }
    if (request.method === "GET" && (url.pathname === "/v1/models" || url.pathname === "/v1/version")) return models();
    if (request.method === "GET" && url.pathname === "/__edge_health") {
      return json({ edge: "ok", backend: "cloudflare-container", replicas: 2 }, 200);
    }
    if ((request.method === "GET" || request.method === "HEAD") && PUBLIC.has(url.pathname)) {
      return proxy(request, env, id, null);
    }
    if (request.method !== "POST" || (url.pathname !== "/v1/decisions" && url.pathname !== "/v1/responses")) {
      return json({ code: "not_found", detail: "Unknown production route." }, 404);
    }
    const length = request.headers.get("content-length");
    if (length && Number(length) > INFER_MAX_BODY_BYTES) {
      return json({ code: "request_too_large", detail: "Request exceeds the size limit." }, 413);
    }
    const scope = url.pathname === "/v1/responses" ? "responses:invoke" : "decisions:invoke";
    const identity = await authenticate(env.VEYA_DB, bearer(request.headers.get("authorization")), new Date(), scope);
    if (!identity) return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
    let payload = {};
    try {
      payload = await request.clone().json();
    } catch {
      return json({ code: "invalid_input", detail: "Request body must be JSON." }, 400);
    }
    if (payload.model || payload.base_url) {
      return json({ code: "invalid_input", detail: "Caller cannot select a model or URL." }, 400);
    }
    if (url.pathname === "/v1/responses") {
      if (payload.task === "bounded_decision") {
        return json({ code: "wrong_endpoint", detail: "Bounded decisions use /v1/decisions." }, 400);
      }
      return json({
        code: "specialist_not_enabled",
        detail: "No production specialist is enabled for this task.",
        benchbrex_tuned: false,
      }, 403);
    }
    if (payload.primitive === "systemone") {
      return json({ code: "unsupported_primitive", detail: "SystemOne is not served by this release." }, 400);
    }
    const heavy = payload.primitive && payload.primitive !== "choice";
    const block = await limited(env, identity.keyId, heavy);
    if (block) return block;
    const started = Date.now();
    const infer = new Request(new URL("/api/infer", url.origin), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const response = await proxy(infer, env, id, identity);
    await recordUsage(env, identity, {
      endpoint: "/v1/decisions",
      primitive: payload.primitive || "",
      status: response.status,
      latencyMs: Date.now() - started,
      route: response.headers.get("x-veya-replica"),
    }).catch(() => undefined);
    return response;
  },
};

async function proxy(request, env, id, identity) {
  if (!env || !env.VEYA_CONTAINER) return failure(id, 503, "container_unavailable");
  try {
    const outboundHeaders = forwardHeaders(request, id);
    const result = await fetchFromPool(env.VEYA_CONTAINER, async (stub) => {
      if (new URL(request.url).pathname === "/api/infer") {
        const probe = await stub.fetch(new Request(new URL("/readyz", request.url), { method: "GET" }));
        if (probe.status !== 200) return probe;
      }
      return stub.fetch(new Request(request, { headers: outboundHeaders }));
    }, { accept: (response) => response && response.status < 500 });
    const headers = new Headers(result.response.headers);
    headers.set("x-veya-replica", String(result.index));
    if (identity) headers.set("x-veya-project", identity.projectId);
    return stamp(new Response(result.response.body, { status: result.response.status, headers }));
  } catch {
    return failure(id, 503, "container_unavailable");
  }
}

