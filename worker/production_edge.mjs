/**
 * Request handling for the isolated production Worker `veya-production`.
 *
 * This module never imports the Containers runtime, so every branch is unit
 * tested with plain stubs (tests/test_veya_production_edge.mjs). The deployed
 * entrypoint is worker/veya-production.mjs.
 *
 * Guarantees:
 *  - /healthz and /__edge_health are answered at the edge and never wake a container.
 *  - Container-touching probes (/readyz, /startupz, /version, /api/version) and
 *    every /__admin/* route require the admin bearer secret VEYA_EDGE_TOKEN.
 *  - Inference bodies are capped on the bytes actually streamed (chunked
 *    uploads included), not only on Content-Length.
 *  - A D1 failure during key lookup is a structured 503, not a raw 500.
 *  - Container calls have a deadline (VEYA_INFER_TIMEOUT_MS, default 15 s);
 *    expiry is a structured 504.
 *  - Decision calls are idempotent model inference, so each replica attempt has
 *    its own timeout (VEYA_ATTEMPT_TIMEOUT_MS, default 6 s) and a failed attempt
 *    (5xx, timeout, throw) is retried once on the other replica, always inside
 *    the overall deadline. A replica that timed out or errored is "suspect" for
 *    VEYA_SUSPECT_MS (default 10 s) and is tried second.
 *  - When every replica sheds load (gateway 503 model_busy), the caller gets a
 *    structured 503 `model_busy` with Retry-After instead of a generic error.
 *  - Usage is written through ctx.waitUntil after the response is ready.
 *  - Every response carries HSTS and the standard security headers.
 */

import { authenticate, bearer } from "./api_keys.mjs";
import {
  INFER_MAX_BODY_BYTES,
  SECURITY_HEADERS,
  adminVersion,
  forwardHeaders,
  requestId,
  stamp,
} from "./edge_policy.mjs";
import { discard, fetchTracker } from "./body_hygiene.mjs";
import { REPLICA_COUNT, fetchFromPool, replicaName, stubFor } from "./pool.mjs";

export const DEFAULT_INFER_TIMEOUT_MS = 15_000;
export const DEFAULT_PROBE_TIMEOUT_MS = 5_000;
export const DEFAULT_READY_CACHE_MS = 15_000;
export const DEFAULT_ATTEMPT_TIMEOUT_MS = 6_000;
export const DEFAULT_SUSPECT_MS = 10_000;
export const BUSY_RETRY_AFTER_S = 2;
/** Paths whose upstream call is side-effect free and may be retried on another replica. */
export const IDEMPOTENT_INFERENCE = new Set(["/v1/decisions"]);
export const TENANT_HEADER = "x-veya-tenant";

const EDGE_ONLY = new Set(["/healthz", "/__edge_health"]);
const ADMIN_PROBES = new Set(["/readyz", "/startupz", "/version", "/api/version"]);
const INFERENCE = new Set(["/v1/decisions", "/v1/responses"]);
const ADMIN_REPLICA = /^\/__admin\/replicas\/(\d+)\/(stop|restart)$/;

export class BodyTooLarge extends Error {}
export class DeadlineExceeded extends Error {}

export function json(body, status, extra) {
  const headers = new Headers(extra);
  headers.set("content-type", "application/json");
  headers.set("cache-control", "no-store");
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
  return new Response(JSON.stringify(body), { status, headers });
}

function bounded(raw, fallback, min, max) {
  const value = Number(raw);
  if (raw === undefined || raw === null || raw === "" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

export function inferTimeoutMs(env) {
  return bounded(env && env.VEYA_INFER_TIMEOUT_MS, DEFAULT_INFER_TIMEOUT_MS, 1_000, 120_000);
}

export function probeTimeoutMs(env) {
  return bounded(env && env.VEYA_PROBE_TIMEOUT_MS, DEFAULT_PROBE_TIMEOUT_MS, 500, 30_000);
}

export function readyCacheMs(env) {
  return bounded(env && env.VEYA_READY_CACHE_MS, DEFAULT_READY_CACHE_MS, 0, 60_000);
}

/**
 * Per-attempt budget. Always leaves at least 1 s of the overall deadline for the
 * retry on the other replica, so attempt + retry never exceed the deadline.
 */
export function attemptTimeoutMs(env, deadlineMs = inferTimeoutMs(env)) {
  const ceiling = Math.max(500, deadlineMs - 1_000);
  return Math.min(ceiling, bounded(env && env.VEYA_ATTEMPT_TIMEOUT_MS, DEFAULT_ATTEMPT_TIMEOUT_MS, 500, 120_000));
}

export function suspectMs(env) {
  return bounded(env && env.VEYA_SUSPECT_MS, DEFAULT_SUSPECT_MS, 0, 120_000);
}

/** Constant-time check of the admin bearer secret. */
export function adminAuthorized(request, env) {
  const token = env && env.VEYA_EDGE_TOKEN;
  if (typeof token !== "string" || token.length === 0) return false;
  const presented = bearer(request.headers.get("authorization"));
  if (presented.length !== token.length) return false;
  let diff = 0;
  for (let index = 0; index < presented.length; index += 1) {
    diff |= presented.charCodeAt(index) ^ token.charCodeAt(index);
  }
  return diff === 0;
}

/**
 * Read at most `limit` bytes. The declared Content-Length is checked first, then
 * the stream is counted as it arrives, so a chunked upload without a length is
 * cut off as soon as it crosses the cap.
 */
export async function readBodyLimited(request, limit = INFER_MAX_BODY_BYTES) {
  const declared = request.headers.get("content-length");
  if (declared !== null && declared !== "") {
    if (!/^\d+$/.test(declared) || Number(declared) > limit) throw new BodyTooLarge();
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
        // the client may already have gone away
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

/**
 * Run `work(signal)` against an AbortController deadline. The race guarantees the
 * caller gets DeadlineExceeded on time even if the underlying stub call cannot be
 * cancelled; `work` checks `signal.aborted` so no further replica is attempted.
 */
export async function withDeadline(ms, work) {
  const controller = new AbortController();
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new DeadlineExceeded());
    }, ms);
  });
  try {
    return await Promise.race([work(controller.signal), deadline]);
  } finally {
    clearTimeout(timer);
  }
}

function upstreamFailure(id, status, code, detail) {
  return json({ code, detail, request_id: id }, status, status === 503 ? { "retry-after": "5" } : undefined);
}

function timeoutFailure(id) {
  return upstreamFailure(id, 504, "upstream_timeout", "Veya Core did not answer before the deadline.");
}

function unavailable(id) {
  return upstreamFailure(id, 503, "container_unavailable", "The Veya container did not respond.");
}

function overloaded(id) {
  return json(
    {
      code: "model_busy",
      detail: "Every Veya Core replica is at capacity. Retry after the indicated delay.",
      retry_after: BUSY_RETRY_AFTER_S,
      request_id: id,
    },
    503,
    { "retry-after": String(BUSY_RETRY_AFTER_S) },
  );
}

/** Mark how many replica attempts a response (success or error) took. */
export function withAttempts(response, attempts) {
  response.headers.set("x-veya-attempts", String(attempts));
  return response;
}

/** Gateway load shedding: 503 with body code "model_busy" (web/server.py). */
export async function isLoadShed(response) {
  if (!response || response.status !== 503) return false;
  try {
    const body = await response.clone().json();
    return Boolean(body) && body.code === "model_busy";
  } catch {
    return false;
  }
}

/** Replica order: random, but replicas marked suspect go last. */
export function orderReplicas(count, cache, random = Math.random) {
  const first = Math.floor(random() * count);
  const order = Array.from({ length: count }, (_, offset) => (first + offset) % count);
  const healthy = order.filter((index) => !cache.suspect(index));
  const suspect = order.filter((index) => cache.suspect(index));
  return healthy.concat(suspect);
}

/**
 * Per-isolate readiness cache: replica index -> epoch ms until which a 200 from
 * /readyz is trusted. A failed or 5xx inference clears the entry.
 */
export function createReadyCache(now = () => Date.now()) {
  const entries = new Map();
  const suspects = new Map();
  return {
    fresh(index) {
      const until = entries.get(index);
      return typeof until === "number" && until > now();
    },
    mark(index, ttl) {
      if (ttl > 0) entries.set(index, now() + ttl);
      suspects.delete(index);
    },
    clear(index) {
      entries.delete(index);
    },
    /** With `ms`, mark the replica suspect for that long; without, report whether it is suspect. */
    suspect(index, ms) {
      if (ms === undefined) {
        const until = suspects.get(index);
        return typeof until === "number" && until > now();
      }
      if (ms > 0) suspects.set(index, now() + ms);
      return true;
    },
  };
}

const sharedReadyCache = createReadyCache();

export function createHandler(build, options = {}) {
  const readyCache = options.readyCache || sharedReadyCache;
  const random = options.random;

  function models() {
    return json({
      release: build.release,
      checkpoint_sha256: build.checkpointSha256,
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

  function edgeHealth(request) {
    const response = json({ status: "ok", edge: "ok", backend: "cloudflare-container", replicas: REPLICA_COUNT, release: build.release }, 200);
    if (request.method === "HEAD") return new Response(null, { status: 200, headers: response.headers });
    return response;
  }

  async function proxyProbe(request, env, ctx, id) {
    if (!env || !env.VEYA_CONTAINER) return unavailable(id);
    const headers = forwardHeaders(request, id);
    const tracker = fetchTracker();
    try {
      const result = await withDeadline(probeTimeoutMs(env), (signal) => fetchFromPool(
        env.VEYA_CONTAINER,
        (stub) => {
          if (signal.aborted) throw new DeadlineExceeded();
          return tracker.fetch(stub, new Request(request.url, { method: request.method, headers }));
        },
        { accept: (response) => response && response.status < 500, random },
      ));
      const out = new Headers(result.response.headers);
      out.set("x-veya-replica", String(result.index));
      return stamp(new Response(result.response.body, { status: result.response.status, headers: out }));
    } catch (error) {
      // Whatever is still on its way back is never used.
      background(ctx, tracker.release());
      if (error instanceof DeadlineExceeded) return timeoutFailure(id);
      return unavailable(id);
    }
  }

  /**
   * One replica attempt: readiness probe (unless cached) then the inference call,
   * both inside `budgetMs`. Returns { kind, response } where kind is
   * ok | busy | failed | timeout | error; only "ok" carries a response (the
   * caller owns its body). Every other container response is read or cancelled
   * here, including ones that arrive after the budget ran out (body_hygiene.mjs).
   */
  async function attemptReplica(ctx, stub, index, infer, headers, body, budgetMs, ttl) {
    const tracker = fetchTracker();
    try {
      return await withDeadline(budgetMs, async (signal) => {
        if (!readyCache.fresh(index)) {
          const probe = await tracker.fetch(stub, new Request(new URL("/readyz", infer.url), { method: "GET" }));
          // Only the status matters; release the body so the probe is not left in flight.
          await discard(probe);
          if (probe.status !== 200) {
            readyCache.clear(index);
            return { kind: "failed", status: probe.status, response: null };
          }
          readyCache.mark(index, ttl);
        }
        if (signal.aborted) throw new DeadlineExceeded();
        const response = await tracker.fetch(stub, new Request(infer.url, { method: "POST", headers, body }));
        if (signal.aborted) {
          // The budget expired while the answer was on its way; nobody will read it.
          await discard(response);
          throw new DeadlineExceeded();
        }
        if (response.status < 500) return { kind: "ok", status: response.status, response };
        const shed = await isLoadShed(response);
        await discard(response);
        if (shed) return { kind: "busy", status: response.status, response: null };
        readyCache.clear(index);
        return { kind: "failed", status: response.status, response: null };
      });
    } catch (error) {
      readyCache.clear(index);
      // A probe or inference call may still answer after the deadline: cancel it then.
      background(ctx, tracker.release());
      return { kind: error instanceof DeadlineExceeded ? "timeout" : "error", response: null };
    }
  }

  async function proxyInference(infer, env, ctx, id, identity, options = {}) {
    if (!env || !env.VEYA_CONTAINER) return unavailable(id);
    const headers = forwardHeaders(infer, id);
    headers.set("content-type", "application/json");
    headers.set(TENANT_HEADER, identity.keyId);
    const body = await infer.text();
    const ttl = readyCacheMs(env);
    const deadlineMs = inferTimeoutMs(env);
    const perAttempt = attemptTimeoutMs(env, deadlineMs);
    const cooldown = suspectMs(env);
    const retry = options.retry !== false;
    const started = Date.now();
    const order = orderReplicas(REPLICA_COUNT, readyCache, random).slice(0, retry ? 2 : 1);
    const attempts = [];
    const pending = [];
    try {
      const result = await withDeadline(deadlineMs, async (signal) => {
        for (let n = 0; n < order.length; n += 1) {
          if (signal.aborted) throw new DeadlineExceeded();
          const index = order[n];
          const remaining = deadlineMs - (Date.now() - started);
          if (remaining <= 0) throw new DeadlineExceeded();
          // The last attempt may use whatever is left of the overall deadline.
          const budget = n === order.length - 1 ? remaining : Math.min(perAttempt, remaining);
          const stub = stubFor(env.VEYA_CONTAINER, index);
          const running = attemptReplica(ctx, stub, index, infer, headers, body, budget, ttl);
          pending.push(running);
          const outcome = await running;
          attempts.push({ index, kind: outcome.kind });
          if (outcome.kind === "ok") return { index, response: outcome.response };
          if (outcome.kind !== "busy") readyCache.suspect(index, cooldown);
        }
        return null;
      });
      if (result) {
        const out = new Headers(result.response.headers);
        out.set("x-veya-replica", String(result.index));
        out.set("x-veya-project", identity.projectId);
        out.set("x-veya-attempts", String(attempts.length));
        return stamp(new Response(result.response.body, { status: result.response.status, headers: out }));
      }
      if (attempts.length > 0 && attempts.every((attempt) => attempt.kind === "busy")) {
        return withAttempts(overloaded(id), attempts.length);
      }
      if (attempts.length > 0 && attempts.every((attempt) => attempt.kind === "timeout")) {
        return withAttempts(timeoutFailure(id), attempts.length);
      }
      return withAttempts(unavailable(id), attempts.length);
    } catch (error) {
      // The overall deadline won: an attempt still running may yet return an
      // "ok" response that nobody will read. Cancel it when it lands.
      background(ctx, Promise.all(pending.map((running) => running.then((outcome) => discard(outcome && outcome.response)))));
      const response = error instanceof DeadlineExceeded ? timeoutFailure(id) : unavailable(id);
      return withAttempts(response, pending.length);
    }
  }

  async function replicaReport(env, ctx, id) {
    if (!env || !env.VEYA_CONTAINER) return unavailable(id);
    const replicas = [];
    for (let index = 0; index < REPLICA_COUNT; index += 1) {
      const stub = stubFor(env.VEYA_CONTAINER, index);
      const entry = { index, name: replicaName(index), ready_status: null, state: null };
      const tracker = fetchTracker();
      try {
        const probe = await withDeadline(probeTimeoutMs(env), () => tracker.fetch(
          stub,
          new Request("https://veya.internal/readyz", { method: "GET" }),
        ));
        entry.ready_status = probe.status;
        await discard(probe);
      } catch (error) {
        entry.ready_status = error instanceof DeadlineExceeded ? "timeout" : "error";
        background(ctx, tracker.release());
      }
      if (typeof stub.getState === "function") {
        try {
          entry.state = await withDeadline(probeTimeoutMs(env), () => stub.getState());
        } catch {
          entry.state = "unknown";
        }
      }
      replicas.push(entry);
    }
    return json({ replicas, request_id: id }, 200);
  }

  async function replicaAction(env, ctx, id, index, action) {
    if (!env || !env.VEYA_CONTAINER) return unavailable(id);
    if (!Number.isInteger(index) || index < 0 || index >= REPLICA_COUNT) {
      return json({ code: "unknown_replica", detail: "Replica index is out of range.", request_id: id }, 404);
    }
    const stub = stubFor(env.VEYA_CONTAINER, index);
    readyCache.clear(index);
    try {
      if (action === "stop") {
        if (typeof stub.stop !== "function") throw new TypeError("stop unsupported");
        await stub.stop();
        return json({ replica: index, name: replicaName(index), action: "stop", accepted: true, request_id: id }, 202);
      }
      if (typeof stub.destroy !== "function" || typeof stub.start !== "function") {
        throw new TypeError("restart unsupported");
      }
      await stub.destroy();
      const starting = stub.start();
      if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(Promise.resolve(starting).catch(() => undefined));
      else Promise.resolve(starting).catch(() => undefined);
      return json({ replica: index, name: replicaName(index), action: "restart", accepted: true, request_id: id }, 202);
    } catch (error) {
      if (error instanceof TypeError) {
        return json({ code: "not_supported", detail: "This Containers runtime does not expose the lifecycle call.", request_id: id }, 501);
      }
      return json({ code: "replica_action_failed", detail: "The replica did not accept the lifecycle call.", request_id: id }, 502);
    }
  }

  async function recordUsage(env, identity, fields) {
    if (env && env.VEYA_USAGE && typeof env.VEYA_USAGE.writeDataPoint === "function") {
      env.VEYA_USAGE.writeDataPoint({
        indexes: [identity.organizationId],
        // blob1..blob6 keep their existing meaning; blob7 (replica) is new.
        blobs: [
          identity.projectId,
          identity.keyId,
          fields.endpoint,
          fields.primitive || fields.task || "",
          fields.route || "",
          String(fields.status),
          fields.replica == null ? "" : String(fields.replica),
        ],
        doubles: [fields.latencyMs],
      });
    }
    if (!env || !env.VEYA_DB) return false;
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
      build.release,
      new Date().toISOString(),
    ).run();
    return true;
  }

  async function decisionRoute(response) {
    if (!response || !response.ok) return null;
    try {
      const text = await response.text();
      if (text.length > 1_000_000) return null;
      const body = JSON.parse(text);
      const route = body && (body.route || (body.result && body.result.route));
      return typeof route === "string" && route.length <= 80 ? route : null;
    } catch {
      return null;
    }
  }

  function background(ctx, promise) {
    const guarded = Promise.resolve(promise).catch(() => undefined);
    if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(guarded);
    return guarded;
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

  async function inference(request, env, ctx, id, url) {
    const declared = request.headers.get("content-length");
    if (declared !== null && declared !== "" && (!/^\d+$/.test(declared) || Number(declared) > INFER_MAX_BODY_BYTES)) {
      return json({ code: "request_too_large", detail: "Request exceeds the size limit.", request_id: id }, 413);
    }
    const scope = url.pathname === "/v1/responses" ? "responses:invoke" : "decisions:invoke";
    let identity;
    try {
      identity = await authenticate(env && env.VEYA_DB, bearer(request.headers.get("authorization")), new Date(), scope);
    } catch {
      return json(
        { code: "auth_unavailable", detail: "API key verification is temporarily unavailable.", request_id: id },
        503,
        { "retry-after": "5" },
      );
    }
    if (!identity) return json({ code: "unauthorized", detail: "Authentication is required.", request_id: id }, 401);
    let payload;
    try {
      const raw = await readBodyLimited(request, INFER_MAX_BODY_BYTES);
      payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
    } catch (error) {
      if (error instanceof BodyTooLarge) {
        return json({ code: "request_too_large", detail: "Request exceeds the size limit.", request_id: id }, 413);
      }
      return json({ code: "invalid_input", detail: "Request body must be JSON.", request_id: id }, 400);
    }
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      return json({ code: "invalid_input", detail: "Request body must be a JSON object.", request_id: id }, 400);
    }
    if (payload.model || payload.base_url) {
      return json({ code: "invalid_input", detail: "Caller cannot select a model or URL.", request_id: id }, 400);
    }
    if (url.pathname === "/v1/responses") {
      if (payload.task === "bounded_decision") {
        return json({ code: "wrong_endpoint", detail: "Bounded decisions use /v1/decisions.", request_id: id }, 400);
      }
      return json({
        code: "specialist_not_enabled",
        detail: "No production specialist is enabled for this task.",
        benchbrex_tuned: false,
        request_id: id,
      }, 403);
    }
    if (payload.primitive === "systemone") {
      return json({ code: "unsupported_primitive", detail: "SystemOne is not served by this release.", request_id: id }, 400);
    }
    const heavy = Boolean(payload.primitive && payload.primitive !== "choice");
    let block;
    try {
      block = await limited(env, identity.keyId, heavy);
    } catch {
      return json({ code: "rate_limit_unavailable", detail: "Rate limiting is temporarily unavailable.", request_id: id }, 503, { "retry-after": "5" });
    }
    if (block) return block;
    const started = Date.now();
    const infer = new Request(new URL("/api/infer", url.origin), {
      method: "POST",
      headers: { "content-type": "application/json", "x-request-id": id },
      body: JSON.stringify(payload),
    });
    const response = await proxyInference(infer, env, ctx, id, identity, { retry: IDEMPOTENT_INFERENCE.has(url.pathname) });
    const latencyMs = Date.now() - started;
    const replica = response.headers.get("x-veya-replica");
    const copy = response.ok ? response.clone() : null;
    background(ctx, (async () => {
      const route = await decisionRoute(copy);
      await recordUsage(env, identity, {
        endpoint: "/v1/decisions",
        primitive: payload.primitive || "",
        status: response.status,
        latencyMs,
        route,
        replica,
      });
    })());
    return response;
  }

  return async function handle(request, env, ctx) {
    const id = requestId(request);
    const url = new URL(request.url);
    const { pathname } = url;
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: SECURITY_HEADERS });
    if ((request.method === "GET" || request.method === "HEAD") && EDGE_ONLY.has(pathname)) return edgeHealth(request);
    if (request.method === "GET" && (pathname === "/v1/models" || pathname === "/v1/version")) return models();
    if (pathname.startsWith("/__admin/") || ADMIN_PROBES.has(pathname)) {
      if (!adminAuthorized(request, env)) {
        return json({ code: "unauthorized", detail: "Authentication is required.", request_id: id }, 401);
      }
      if (request.method === "GET" && pathname === "/__admin/version") return stamp(adminVersion(env, build));
      if (request.method === "GET" && pathname === "/__admin/replicas") return replicaReport(env, ctx, id);
      const action = ADMIN_REPLICA.exec(pathname);
      if (action && request.method === "POST") return replicaAction(env, ctx, id, Number(action[1]), action[2]);
      if ((request.method === "GET" || request.method === "HEAD") && ADMIN_PROBES.has(pathname)) {
        return proxyProbe(request, env, ctx, id);
      }
      return json({ code: "not_found", detail: "Unknown admin route.", request_id: id }, 404);
    }
    if (request.method === "POST" && INFERENCE.has(pathname)) return inference(request, env, ctx, id, url);
    return json({ code: "not_found", detail: "Unknown production route.", request_id: id }, 404);
  };
}
