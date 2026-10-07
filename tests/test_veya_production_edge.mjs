import assert from "node:assert/strict";
import test from "node:test";

import { sha256Hex } from "../worker/api_keys.mjs";
import { INFER_MAX_BODY_BYTES, forwardHeaders } from "../worker/edge_policy.mjs";
import {
  BodyTooLarge,
  DEFAULT_INFER_TIMEOUT_MS,
  TENANT_HEADER,
  createHandler,
  createReadyCache,
  inferTimeoutMs,
  readBodyLimited,
  readyCacheMs,
} from "../worker/production_edge.mjs";

const ADMIN = "admin-secret-for-tests-0123456789";
const KEY = "vk_test_customer_key";
const BUILD = { release: "0.3.5-audit-baseline.1", checkpointSha256: "c".repeat(64), manifestSha256: "m".repeat(64) };

function req(path, { method = "GET", headers = {}, body, duplex } = {}) {
  const init = { method, headers, body };
  if (duplex) init.duplex = duplex;
  return new Request("https://api.veyadynamics.com" + path, init);
}

async function keyDb({ fail = false } = {}) {
  const hash = await sha256Hex(KEY);
  const writes = [];
  return {
    writes,
    prepare(sql) {
      return {
        bind(...args) {
          return {
            async first() {
              if (fail) throw new Error("D1_ERROR: network connection lost");
              if (args[0] !== hash) return null;
              return {
                id: "key_1", organization_id: "org_1", project_id: "proj_1", key_hash: hash,
                scopes: JSON.stringify(["decisions:invoke"]), expires_at: null, revoked_at: null,
              };
            },
            async run() {
              writes.push({ sql, args });
              return { success: true };
            },
          };
        },
      };
    },
  };
}

function containerBinding(behaviour = {}) {
  const calls = [];
  const lifecycle = [];
  const binding = {
    calls,
    lifecycle,
    idFromName(name) { return name; },
    get(name) {
      return {
        name,
        async fetch(request) {
          const path = new URL(request.url).pathname;
          calls.push({ name, path, method: request.method, tenant: request.headers.get(TENANT_HEADER) });
          if (behaviour.fetch) return behaviour.fetch(name, path, request);
          if (path === "/readyz") return new Response("{}", { status: 200 });
          return Response.json({ route: "packed", answer: "a" });
        },
        async stop() { lifecycle.push([name, "stop"]); },
        async destroy() { lifecycle.push([name, "destroy"]); },
        async start() { lifecycle.push([name, "start"]); },
        async getState() { return { status: "healthy" }; },
      };
    },
  };
  return binding;
}

function context() {
  const pending = [];
  return { pending, waitUntil(promise) { pending.push(promise); } };
}

async function env(overrides = {}) {
  return {
    VEYA_EDGE_TOKEN: ADMIN,
    VEYA_DB: await keyDb(),
    VEYA_CONTAINER: containerBinding(),
    ...overrides,
  };
}

function decision(body = { primitive: "choice", criteria: { a: "x", b: "y" } }, headers = {}) {
  return req("/v1/decisions", {
    method: "POST",
    headers: { authorization: "Bearer " + KEY, "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

function chunkedBody(totalBytes, chunk = 4096) {
  let sent = 0;
  return new ReadableStream({
    pull(controller) {
      if (sent >= totalBytes) return controller.close();
      const size = Math.min(chunk, totalBytes - sent);
      sent += size;
      controller.enqueue(new TextEncoder().encode("x".repeat(size)));
    },
  });
}

// ---- body limit on streamed bytes --------------------------------------------

test("readBodyLimited counts streamed bytes when there is no content-length", async () => {
  const small = req("/v1/decisions", { method: "POST", body: chunkedBody(1000), duplex: "half" });
  assert.equal(small.headers.get("content-length"), null);
  assert.equal((await readBodyLimited(small, INFER_MAX_BODY_BYTES)).byteLength, 1000);
  const large = req("/v1/decisions", { method: "POST", body: chunkedBody(INFER_MAX_BODY_BYTES + 1), duplex: "half" });
  await assert.rejects(readBodyLimited(large, INFER_MAX_BODY_BYTES), BodyTooLarge);
});

test("a chunked request over 32 KB is rejected with 413 even without content-length", async () => {
  const handle = createHandler(BUILD);
  const e = await env();
  const response = await handle(req("/v1/decisions", {
    method: "POST",
    headers: { authorization: "Bearer " + KEY, "content-type": "application/json" },
    body: chunkedBody(40 * 1024),
    duplex: "half",
  }), e, context());
  assert.equal(response.status, 413);
  assert.equal((await response.json()).code, "request_too_large");
  assert.equal(e.VEYA_CONTAINER.calls.length, 0);
});

test("a lying or malformed content-length is refused before auth", async () => {
  const handle = createHandler(BUILD);
  for (const length of [String(INFER_MAX_BODY_BYTES + 1), "12abc"]) {
    const response = await handle(req("/v1/decisions", { method: "POST", headers: { "content-length": length }, body: "{}" }), await env(), context());
    assert.equal(response.status, 413);
  }
});

// ---- health endpoints ----------------------------------------------------------

test("/healthz and /__edge_health are edge-only and never touch a container", async () => {
  const handle = createHandler(BUILD);
  const e = await env();
  for (const path of ["/healthz", "/__edge_health"]) {
    const response = await handle(req(path), e, context());
    assert.equal(response.status, 200);
    assert.equal((await response.json()).edge, "ok");
    const head = await handle(req(path, { method: "HEAD" }), e, context());
    assert.equal(head.status, 200);
  }
  assert.equal(e.VEYA_CONTAINER.calls.length, 0);
});

test("container-touching probes require the admin secret", async () => {
  const handle = createHandler(BUILD);
  const e = await env();
  for (const path of ["/readyz", "/startupz", "/version", "/api/version"]) {
    const anonymous = await handle(req(path), e, context());
    assert.equal(anonymous.status, 401, path);
    const customer = await handle(req(path, { headers: { authorization: "Bearer " + KEY } }), e, context());
    assert.equal(customer.status, 401, path);
  }
  assert.equal(e.VEYA_CONTAINER.calls.length, 0);
  const admin = await handle(req("/readyz", { headers: { authorization: "Bearer " + ADMIN } }), e, context());
  assert.equal(admin.status, 200);
  assert.ok(admin.headers.get("x-veya-replica") !== null);
});

test("admin routes fail closed when the admin secret is not configured", async () => {
  const handle = createHandler(BUILD);
  const response = await handle(req("/__admin/version", { headers: { authorization: "Bearer " } }), await env({ VEYA_EDGE_TOKEN: undefined }), context());
  assert.equal(response.status, 401);
});

// ---- D1 failure ------------------------------------------------------------------

test("a D1 failure during key lookup is a structured 503, not a raw 500", async () => {
  const handle = createHandler(BUILD);
  const e = await env({ VEYA_DB: await keyDb({ fail: true }) });
  const response = await handle(decision(), e, context());
  assert.equal(response.status, 503);
  const body = await response.json();
  assert.equal(body.code, "auth_unavailable");
  assert.ok(body.request_id);
  assert.equal(response.headers.get("retry-after"), "5");
  assert.equal(e.VEYA_CONTAINER.calls.length, 0);
});

test("an unknown key is 401", async () => {
  const handle = createHandler(BUILD);
  const response = await handle(decision(undefined, { authorization: "Bearer nope" }), await env(), context());
  assert.equal(response.status, 401);
});

// ---- usage via waitUntil -----------------------------------------------------------

test("usage is written through ctx.waitUntil with the decision route, not the replica", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const points = [];
  const e = await env({ VEYA_USAGE: { writeDataPoint(point) { points.push(point); } } });
  const ctx = context();
  const response = await handle(decision(), e, ctx);
  assert.equal(response.status, 200);
  assert.equal(e.VEYA_DB.writes.length, 0, "the ledger insert must not block the response");
  assert.equal(ctx.pending.length, 1);
  await Promise.all(ctx.pending);
  assert.equal(e.VEYA_DB.writes.length, 1);
  const args = e.VEYA_DB.writes[0].args;
  assert.equal(args[8], "packed", "usage_ledger.route is the decision route");
  assert.equal(args[9], 200);
  assert.equal(points[0].blobs[4], "packed");
  assert.equal(points[0].blobs[6], "0", "replica is recorded separately");
  assert.deepEqual(await response.json(), { route: "packed", answer: "a" });
});

test("a usage write failure never changes the customer response", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env();
  e.VEYA_DB.prepare = ((original) => (sql) => (sql.includes("INSERT") ? { bind() { return { run() { throw new Error("D1 down"); } }; } } : original(sql)))(e.VEYA_DB.prepare.bind(e.VEYA_DB));
  const ctx = context();
  const response = await handle(decision(), e, ctx);
  assert.equal(response.status, 200);
  await Promise.all(ctx.pending);
});

// ---- inference timeout -------------------------------------------------------------

test("inference timeout defaults to 15 s and is configurable within bounds", () => {
  assert.equal(DEFAULT_INFER_TIMEOUT_MS, 15_000);
  assert.equal(inferTimeoutMs({}), 15_000);
  assert.equal(inferTimeoutMs({ VEYA_INFER_TIMEOUT_MS: "2500" }), 2_500);
  assert.equal(inferTimeoutMs({ VEYA_INFER_TIMEOUT_MS: "5" }), 1_000);
  assert.equal(inferTimeoutMs({ VEYA_INFER_TIMEOUT_MS: "999999" }), 120_000);
  assert.equal(inferTimeoutMs({ VEYA_INFER_TIMEOUT_MS: "abc" }), 15_000);
});

test("a hung container returns a structured 504 at the deadline", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env({
    VEYA_INFER_TIMEOUT_MS: "1000",
    VEYA_CONTAINER: containerBinding({
      fetch(name, path) {
        if (path === "/readyz") return new Response("{}", { status: 200 });
        return new Promise(() => {});
      },
    }),
  });
  const started = Date.now();
  const response = await handle(decision(), e, context());
  const elapsed = Date.now() - started;
  assert.equal(response.status, 504);
  const body = await response.json();
  assert.equal(body.code, "upstream_timeout");
  assert.ok(body.request_id);
  assert.ok(elapsed >= 900 && elapsed < 3000, String(elapsed));
});

test("both replicas down is a structured 503", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env({ VEYA_CONTAINER: containerBinding({ fetch() { throw new Error("down"); } }) });
  const response = await handle(decision(), e, context());
  assert.equal(response.status, 503);
  assert.equal((await response.json()).code, "container_unavailable");
});

// ---- readiness cache ---------------------------------------------------------------

test("readiness is probed once per replica per cache window, and cleared on a 5xx", async () => {
  let now = 1_000_000;
  const cache = createReadyCache(() => now);
  let fail = false;
  const binding = containerBinding({
    fetch(name, path) {
      if (path === "/readyz") return new Response("{}", { status: 200 });
      return fail ? new Response("{}", { status: 500 }) : Response.json({ route: "packed" });
    },
  });
  const handle = createHandler(BUILD, { readyCache: cache, random: () => 0 });
  const e = await env({ VEYA_CONTAINER: binding });
  await handle(decision(), e, context());
  await handle(decision(), e, context());
  const probes = () => binding.calls.filter((call) => call.path === "/readyz" && call.name === "instance-0").length;
  assert.equal(probes(), 1);
  now += readyCacheMs({}) + 1;
  await handle(decision(), e, context());
  assert.equal(probes(), 2);
  fail = true;
  await handle(decision(), e, context());
  assert.equal(cache.fresh(0), false);
});

// ---- tenant header -------------------------------------------------------------------

test("the Worker sets the tenant header from the API key and clients cannot spoof it", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env();
  await handle(decision(undefined, { [TENANT_HEADER]: "spoofed" }), e, context());
  const infer = e.VEYA_CONTAINER.calls.find((call) => call.path === "/api/infer");
  assert.equal(infer.tenant, "key_1");
  const stripped = forwardHeaders(req("/api/infer", { headers: { [TENANT_HEADER]: "spoofed" } }), "req-12345678");
  assert.equal(stripped.get(TENANT_HEADER), null);
});

// ---- security headers ---------------------------------------------------------------

test("every response carries HSTS and the standard security headers", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env();
  const responses = [
    await handle(req("/healthz"), e, context()),
    await handle(req("/v1/models"), e, context()),
    await handle(req("/nope"), e, context()),
    await handle(req("/readyz"), e, context()),
    await handle(decision(), e, context()),
    await handle(req("/v1/decisions", { method: "OPTIONS" }), e, context()),
  ];
  for (const response of responses) {
    assert.match(response.headers.get("strict-transport-security") || "", /max-age=\d{8,}; includeSubDomains/);
    assert.equal(response.headers.get("x-content-type-options"), "nosniff");
    assert.equal(response.headers.get("x-frame-options"), "DENY");
    assert.ok(response.headers.get("content-security-policy"));
    assert.ok(response.headers.get("referrer-policy"));
  }
});

// ---- failover drill -------------------------------------------------------------------

test("admin can stop or restart one replica; everyone else cannot", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env();
  const denied = await handle(req("/__admin/replicas/0/stop", { method: "POST", headers: { authorization: "Bearer " + KEY } }), e, context());
  assert.equal(denied.status, 401);
  assert.deepEqual(e.VEYA_CONTAINER.lifecycle, []);
  const auth = { authorization: "Bearer " + ADMIN };
  const stop = await handle(req("/__admin/replicas/1/stop", { method: "POST", headers: auth }), e, context());
  assert.equal(stop.status, 202);
  assert.deepEqual(e.VEYA_CONTAINER.lifecycle, [["instance-1", "stop"]]);
  const ctx = context();
  const restart = await handle(req("/__admin/replicas/0/restart", { method: "POST", headers: auth }), e, ctx);
  assert.equal(restart.status, 202);
  await Promise.all(ctx.pending);
  assert.deepEqual(e.VEYA_CONTAINER.lifecycle.slice(1), [["instance-0", "destroy"], ["instance-0", "start"]]);
  const outOfRange = await handle(req("/__admin/replicas/2/stop", { method: "POST", headers: auth }), e, context());
  assert.equal(outOfRange.status, 404);
  const report = await handle(req("/__admin/replicas", { headers: auth }), e, context());
  const body = await report.json();
  assert.equal(body.replicas.length, 2);
  assert.deepEqual(body.replicas.map((item) => item.ready_status), [200, 200]);
});

test("traffic fails over to the healthy replica after one is stopped", async () => {
  const stopped = new Set(["instance-0"]);
  const binding = containerBinding({
    fetch(name, path) {
      if (stopped.has(name)) throw new Error("container stopped");
      return path === "/readyz" ? new Response("{}", { status: 200 }) : Response.json({ route: "packed" });
    },
  });
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const response = await handle(decision(), await env({ VEYA_CONTAINER: binding }), context());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
});

// ---- unchanged product rules ----------------------------------------------------------

test("product rules are unchanged: no model selection, no SystemOne, responses not enabled", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache() });
  const e = await env();
  assert.equal((await handle(decision({ model: "x" }), e, context())).status, 400);
  assert.equal((await handle(decision({ primitive: "systemone" }), e, context())).status, 400);
  assert.equal((await handle(decision([1, 2]), e, context())).status, 400);
  const limited = await handle(decision(), await env({ VEYA_RATE_INFERENCE: { async limit() { return { success: false }; } } }), context());
  assert.equal(limited.status, 429);
});
