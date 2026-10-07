// Every container response the edge does not forward must be read or cancelled.
// @cloudflare/containers keeps a request "in flight" until its body is piped to
// the end, and an in-flight request keeps renewing sleepAfter (body_hygiene.mjs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { sha256Hex } from "../worker/api_keys.mjs";
import { discard, discardLate, fetchTracker } from "../worker/body_hygiene.mjs";
import { fetchFromPool } from "../worker/pool.mjs";
import { createHandler, createReadyCache } from "../worker/production_edge.mjs";

const KEY = "vk_test_customer_key";
const ADMIN = "a".repeat(40);
const BUILD = { release: "r", checkpointSha256: "c".repeat(64), manifestSha256: "m".repeat(64) };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function keyDb() {
  const hash = await sha256Hex(KEY);
  return {
    prepare() {
      return {
        bind(...args) {
          return {
            async first() {
              if (args[0] !== hash) return null;
              return { id: "key_1", organization_id: "org_1", project_id: "proj_1", key_hash: hash,
                scopes: JSON.stringify(["decisions:invoke"]), expires_at: null, revoked_at: null };
            },
            async run() { return { success: true }; },
          };
        },
      };
    },
  };
}

/** A response whose body records whether it was read to the end or cancelled. */
function tracked(registry, label, status, payload, headers = {}) {
  const rec = { label, status, settled: null };
  registry.push(rec);
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  let sent = false;
  const body = new ReadableStream({
    pull(controller) {
      if (!sent) {
        sent = true;
        controller.enqueue(bytes);
      } else {
        rec.settled ??= "read";
        controller.close();
      }
    },
    cancel() {
      rec.settled ??= "cancelled";
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status, headers: { "content-type": "application/json", ...headers } });
}

/**
 * behaviour[name][path] -> { status, payload?, delay? }. Default: readyz 200, infer 200.
 * Every response handed out is recorded in `issued`.
 */
function pool(behaviour = {}) {
  const issued = [];
  const calls = [];
  return {
    issued,
    calls,
    idFromName(name) { return name; },
    get(name) {
      return {
        async fetch(request) {
          const path = new URL(request.url).pathname;
          calls.push({ name, path, at: Date.now() });
          const spec = (behaviour[name] && behaviour[name][path]) || { status: 200 };
          if (spec.delay) await sleep(spec.delay);
          const payload = spec.payload || (path === "/readyz" ? { status: "ready" } : { route: "packed", served_by: name });
          return tracked(issued, `${name}${path}`, spec.status, payload, spec.headers);
        },
      };
    },
  };
}

function context() {
  const tasks = [];
  return { tasks, waitUntil(promise) { tasks.push(promise); } };
}

async function environment(behaviour, overrides = {}) {
  return {
    VEYA_EDGE_TOKEN: ADMIN,
    VEYA_DB: await keyDb(),
    VEYA_CONTAINER: pool(behaviour),
    VEYA_ATTEMPT_TIMEOUT_MS: "500",
    VEYA_INFER_TIMEOUT_MS: "1000",
    VEYA_PROBE_TIMEOUT_MS: "500",
    VEYA_SUSPECT_MS: "0",
    ...overrides,
  };
}

function decision() {
  return new Request("https://api.example/v1/decisions", {
    method: "POST",
    headers: { authorization: "Bearer " + KEY, "content-type": "application/json" },
    body: JSON.stringify({ primitive: "choice", criteria: { a: "x", b: "y" } }),
  });
}

/** Read the client response, let late container answers land, flush waitUntil work. */
async function settle(response, ctx, lateMs = 0) {
  const text = await response.text();
  if (lateMs) await sleep(lateMs);
  await Promise.all(ctx.tasks);
  await sleep(5);
  return text;
}

function assertAllReleased(issued) {
  assert.ok(issued.length > 0, "no container response was issued");
  const open = issued.filter((rec) => rec.settled === null);
  assert.deepEqual(open, [], `container responses left unread: ${open.map((rec) => rec.label).join(", ")}`);
}

const first = () => 0; // replica order: instance-0 first

test("readiness probe bodies are released; the forwarded inference body is read by the client", async () => {
  const env = await environment();
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(decision(), env, ctx);
  assert.equal(response.status, 200);
  await settle(response, ctx);
  const probe = env.VEYA_CONTAINER.issued.find((rec) => rec.label === "instance-0/readyz");
  assert.equal(probe.settled, "cancelled");
  assert.equal(env.VEYA_CONTAINER.issued.find((rec) => rec.label === "instance-0/api/infer").settled, "read");
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("a failed readiness probe and a retried 5xx attempt are cancelled before the retry", async () => {
  for (const behaviour of [
    { "instance-0": { "/readyz": { status: 503, payload: { status: "loading" } } } },
    { "instance-0": { "/api/infer": { status: 500, payload: { code: "internal" } } } },
    { "instance-0": { "/api/infer": { status: 502, payload: { code: "bad_gateway" } } } },
  ]) {
    const env = await environment(behaviour);
    const ctx = context();
    const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
    const response = await handle(decision(), env, ctx);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("x-veya-replica"), "1");
    assert.equal(response.headers.get("x-veya-attempts"), "2");
    await settle(response, ctx);
    assertAllReleased(env.VEYA_CONTAINER.issued);
  }
});

test("load-shed 503 bodies are released whether the retry succeeds or every replica is busy", async () => {
  const shed = { status: 503, payload: { code: "model_busy", retry_after: 5 }, headers: { "retry-after": "5" } };
  let env = await environment({ "instance-0": { "/api/infer": shed } });
  let ctx = context();
  let handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  let response = await handle(decision(), env, ctx);
  assert.equal(response.status, 200);
  await settle(response, ctx);
  assertAllReleased(env.VEYA_CONTAINER.issued);

  env = await environment({ "instance-0": { "/api/infer": shed }, "instance-1": { "/api/infer": shed } });
  ctx = context();
  handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  response = await handle(decision(), env, ctx);
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("retry-after"), "2");
  assert.equal(response.headers.get("x-veya-attempts"), "2");
  assert.equal(JSON.parse(await settle(response, ctx)).code, "model_busy");
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("an inference answer that arrives after the attempt budget is cancelled when it lands", async () => {
  const env = await environment({ "instance-0": { "/api/infer": { status: 200, delay: 900 } } });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(decision(), env, ctx);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
  await settle(response, ctx, 700);
  const late = env.VEYA_CONTAINER.issued.find((rec) => rec.label === "instance-0/api/infer");
  assert.ok(late, "the slow replica eventually answered");
  assert.equal(late.settled, "cancelled");
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("a readiness probe that answers after the budget is cancelled when it lands", async () => {
  const env = await environment({ "instance-0": { "/readyz": { status: 200, delay: 900 } } });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(decision(), env, ctx);
  assert.equal(response.status, 200);
  await settle(response, ctx, 700);
  assert.equal(env.VEYA_CONTAINER.issued.find((rec) => rec.label === "instance-0/readyz").settled, "cancelled");
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("when the overall deadline wins, both late answers are cancelled and the 504 says 2 attempts", async () => {
  const slow = { "/api/infer": { status: 200, delay: 1200 } };
  const env = await environment({ "instance-0": slow, "instance-1": slow });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const started = Date.now();
  const response = await handle(decision(), env, ctx);
  assert.ok(Date.now() - started < 1400, "deadline honoured");
  assert.equal(response.status, 504);
  assert.equal(response.headers.get("x-veya-attempts"), "2");
  await settle(response, ctx, 1300);
  assert.equal(env.VEYA_CONTAINER.issued.filter((rec) => rec.label.endsWith("/api/infer")).length, 2);
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("error responses carry x-veya-attempts", async () => {
  const fail = { "/api/infer": { status: 500, payload: { code: "internal" } } };
  const env = await environment({ "instance-0": fail, "instance-1": fail });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(decision(), env, ctx);
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("x-veya-attempts"), "2");
  assert.equal(JSON.parse(await settle(response, ctx)).code, "container_unavailable");
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("admin replica report releases probe bodies, including probes that answer late", async () => {
  const env = await environment({ "instance-1": { "/readyz": { status: 200, delay: 800 } } });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(new Request("https://api.example/__admin/replicas", {
    headers: { authorization: "Bearer " + ADMIN },
  }), env, ctx);
  assert.equal(response.status, 200);
  const report = JSON.parse(await settle(response, ctx, 500));
  assert.deepEqual(report.replicas.map((replica) => replica.ready_status), [200, "timeout"]);
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("admin probe passthrough releases a rejected replica's body and forwards the other", async () => {
  const env = await environment({ "instance-0": { "/readyz": { status: 500, payload: { status: "broken" } } } });
  const ctx = context();
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: first });
  const response = await handle(new Request("https://api.example/readyz", {
    headers: { authorization: "Bearer " + ADMIN },
  }), env, ctx);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
  await settle(response, ctx);
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("fetchFromPool cancels responses it rejects", async () => {
  const issued = [];
  const binding = { idFromName: (name) => name, get: (name) => ({ name }) };
  const result = await fetchFromPool(binding, (stub) => tracked(issued, stub.name, stub.name === "instance-0" ? 503 : 200, {}), {
    random: first,
  });
  assert.equal(result.index, 1);
  await result.response.text();
  assert.deepEqual(issued.map((rec) => rec.settled), ["cancelled", "read"]);
});

test("discard helpers tolerate null, bodiless, read and locked responses", async () => {
  await discard(null);
  await discard(new Response(null, { status: 204 }));
  const read = new Response("x");
  await read.text();
  await discard(read);
  const locked = new Response("y");
  locked.body.getReader();
  await discard(locked);
  await discardLate(Promise.reject(new Error("network")));
  const issued = [];
  const tracker = fetchTracker();
  const stub = { fetch: () => tracked(issued, "t", 200, {}) };
  const keep = await tracker.fetch(stub, new Request("https://x/a"));
  await tracker.fetch(stub, new Request("https://x/b"));
  await tracker.release(keep);
  assert.deepEqual(issued.map((rec) => rec.settled), [null, "cancelled"]);
  await keep.text();
  assert.equal(issued[0].settled, "read");
});

test("the readiness cache never pings containers on its own", async () => {
  let clock = 1_000_000;
  const cache = createReadyCache(() => clock);
  const env = await environment({}, { VEYA_READY_CACHE_MS: "15000" });
  const handle = createHandler(BUILD, { readyCache: cache, random: first });
  // No requests: nothing reaches a container, however long we wait.
  await sleep(50);
  clock += 10 * 60_000;
  assert.equal(env.VEYA_CONTAINER.calls.length, 0);
  // Two requests inside the cache window: one probe.
  for (let i = 0; i < 2; i += 1) {
    const ctx = context();
    await settle(await handle(decision(), env, ctx), ctx);
  }
  assert.equal(env.VEYA_CONTAINER.calls.filter((call) => call.path === "/readyz").length, 1);
  // Window expires with no traffic: still no calls until the next request, which probes once.
  clock += 60_000;
  await sleep(20);
  assert.equal(env.VEYA_CONTAINER.calls.length, 3);
  const ctx = context();
  await settle(await handle(decision(), env, ctx), ctx);
  assert.equal(env.VEYA_CONTAINER.calls.filter((call) => call.path === "/readyz").length, 2);
  assertAllReleased(env.VEYA_CONTAINER.issued);
});

test("worker code has no background timers that could keep containers awake", () => {
  for (const file of ["production_edge.mjs", "veya-container.mjs", "veya-production.mjs", "pool.mjs", "edge_policy.mjs", "body_hygiene.mjs"]) {
    const source = readFileSync(new URL(`../worker/${file}`, import.meta.url), "utf8");
    assert.doesNotMatch(source, /setInterval\s*\(/, `${file} uses setInterval`);
    assert.doesNotMatch(source, /\bschedule(Every)?\s*\(/, `${file} schedules container work`);
    // The only timer is the per-request deadline in withDeadline, cleared in finally.
    const timers = source.match(/setTimeout\s*\(/g) || [];
    assert.ok(timers.length <= 1, `${file} has ${timers.length} setTimeout calls`);
    if (timers.length) assert.match(source, /clearTimeout\(timer\)/);
  }
});
