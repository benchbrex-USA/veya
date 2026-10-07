import assert from "node:assert/strict";
import test from "node:test";

import { sha256Hex } from "../worker/api_keys.mjs";
import {
  BUSY_RETRY_AFTER_S,
  DEFAULT_ATTEMPT_TIMEOUT_MS,
  IDEMPOTENT_INFERENCE,
  attemptTimeoutMs,
  createHandler,
  createReadyCache,
  isLoadShed,
  orderReplicas,
  suspectMs,
} from "../worker/production_edge.mjs";

const KEY = "vk_test_customer_key";
const BUILD = { release: "r", checkpointSha256: "c".repeat(64), manifestSha256: "m".repeat(64) };

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

const never = () => new Promise(() => {});
const busy = () => Response.json({ code: "model_busy", detail: "Model is busy.", retry_after: 5 }, { status: 503, headers: { "retry-after": "5" } });
const ok = (name) => Response.json({ route: "packed", answer: "a", served_by: name });

/** behaviour: { "instance-0": (path) => Response|Promise, ... }; readyz is 200 unless overridden. */
function binding(behaviour) {
  const calls = [];
  return {
    calls,
    idFromName(name) { return name; },
    get(name) {
      return {
        async fetch(request) {
          const path = new URL(request.url).pathname;
          calls.push({ name, path });
          const handler = behaviour[name];
          if (path === "/readyz" && !(handler && handler.readyz)) return new Response("{}", { status: 200 });
          if (path === "/readyz") return handler.readyz();
          return handler ? handler(path) : ok(name);
        },
      };
    },
  };
}

async function env(behaviour, overrides = {}) {
  return { VEYA_EDGE_TOKEN: "x".repeat(32), VEYA_DB: await keyDb(), VEYA_CONTAINER: binding(behaviour), ...overrides };
}

function decision() {
  return new Request("https://api.example/v1/decisions", {
    method: "POST",
    headers: { authorization: "Bearer " + KEY, "content-type": "application/json" },
    body: JSON.stringify({ primitive: "choice", criteria: { a: "x", b: "y" } }),
  });
}

const ctx = () => ({ waitUntil() {} });
const inferCalls = (e) => e.VEYA_CONTAINER.calls.filter((c) => c.path === "/api/infer");

test("attempt timeout defaults to 6 s and always leaves room for the retry inside the deadline", () => {
  assert.equal(DEFAULT_ATTEMPT_TIMEOUT_MS, 6_000);
  assert.equal(attemptTimeoutMs({}), 6_000);
  assert.equal(attemptTimeoutMs({ VEYA_ATTEMPT_TIMEOUT_MS: "3000" }), 3_000);
  assert.equal(attemptTimeoutMs({ VEYA_ATTEMPT_TIMEOUT_MS: "60000" }), 14_000);
  assert.equal(attemptTimeoutMs({ VEYA_ATTEMPT_TIMEOUT_MS: "60000", VEYA_INFER_TIMEOUT_MS: "5000" }), 4_000);
  assert.equal(attemptTimeoutMs({ VEYA_ATTEMPT_TIMEOUT_MS: "junk" }), 6_000);
  assert.equal(suspectMs({}), 10_000);
  assert.equal(suspectMs({ VEYA_SUSPECT_MS: "0" }), 0);
});

test("a hung replica is abandoned after the attempt timeout and the other replica answers", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env({ "instance-0": never }, { VEYA_ATTEMPT_TIMEOUT_MS: "1000", VEYA_INFER_TIMEOUT_MS: "5000" });
  const started = Date.now();
  const response = await handle(decision(), e, ctx());
  const elapsed = Date.now() - started;
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
  assert.equal(response.headers.get("x-veya-attempts"), "2");
  assert.ok(elapsed >= 900 && elapsed < 2_500, `elapsed ${elapsed}`);
});

test("a hung readiness probe (cold start) also fails over within the attempt timeout", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const zero = () => ok("instance-0");
  zero.readyz = never;
  const e = await env({ "instance-0": zero }, { VEYA_ATTEMPT_TIMEOUT_MS: "800" });
  const response = await handle(decision(), e, ctx());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
});

test("total time never exceeds the overall deadline when both replicas hang", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env({ "instance-0": never, "instance-1": never }, { VEYA_ATTEMPT_TIMEOUT_MS: "1000", VEYA_INFER_TIMEOUT_MS: "2000" });
  const started = Date.now();
  const response = await handle(decision(), e, ctx());
  const elapsed = Date.now() - started;
  assert.equal(response.status, 504);
  assert.equal((await response.json()).code, "upstream_timeout");
  assert.ok(elapsed < 2_600, `elapsed ${elapsed}`);
  assert.equal(inferCalls(e).length, 2, "exactly one retry");
});

test("a 503/504 from one replica is retried once on the other", async () => {
  for (const status of [500, 502, 503, 504]) {
    const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
    const e = await env({ "instance-0": () => Response.json({ code: "x" }, { status }) });
    const response = await handle(decision(), e, ctx());
    assert.equal(response.status, 200, `status ${status}`);
    assert.equal(response.headers.get("x-veya-replica"), "1");
    assert.equal(inferCalls(e).length, 2);
  }
});

test("4xx answers are returned as-is and never retried", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env({ "instance-0": () => Response.json({ code: "invalid_input" }, { status: 400 }) });
  const response = await handle(decision(), e, ctx());
  assert.equal(response.status, 400);
  assert.equal(inferCalls(e).length, 1);
});

test("load shed on one replica is absorbed by the other", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env({ "instance-0": busy });
  const response = await handle(decision(), e, ctx());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-veya-replica"), "1");
});

test("when every replica sheds load the caller gets a structured 503 model_busy with Retry-After", async () => {
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const e = await env({ "instance-0": busy, "instance-1": busy });
  const response = await handle(decision(), e, ctx());
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("retry-after"), String(BUSY_RETRY_AFTER_S));
  assert.ok(response.headers.get("strict-transport-security"));
  const body = await response.json();
  assert.equal(body.code, "model_busy");
  assert.equal(body.retry_after, BUSY_RETRY_AFTER_S);
  assert.ok(body.request_id);
});

test("a replica that timed out is tried second for the suspect window, then first again", async () => {
  let now = 1_000_000;
  const cache = createReadyCache(() => now);
  const handle = createHandler(BUILD, { readyCache: cache, random: () => 0 });
  let hang = true;
  const e = await env({ "instance-0": () => (hang ? never() : ok("instance-0")) }, { VEYA_ATTEMPT_TIMEOUT_MS: "500" });
  assert.equal((await handle(decision(), e, ctx())).headers.get("x-veya-replica"), "1");
  hang = false;
  const before = inferCalls(e).length;
  const second = await handle(decision(), e, ctx());
  assert.equal(second.headers.get("x-veya-replica"), "1");
  assert.equal(second.headers.get("x-veya-attempts"), "1");
  assert.equal(inferCalls(e).length - before, 1, "suspect replica not tried while the other answers");
  now += suspectMs({}) + 1;
  const third = await handle(decision(), e, ctx());
  assert.equal(third.headers.get("x-veya-replica"), "0");
});

test("replica ordering puts suspects last and is random otherwise", () => {
  const cache = createReadyCache(() => 0);
  assert.deepEqual(orderReplicas(2, cache, () => 0), [0, 1]);
  assert.deepEqual(orderReplicas(2, cache, () => 0.9), [1, 0]);
  cache.suspect(0, 5_000);
  assert.deepEqual(orderReplicas(2, cache, () => 0), [1, 0]);
  cache.mark(0, 1_000);
  assert.deepEqual(orderReplicas(2, cache, () => 0), [0, 1], "a fresh readiness 200 clears suspicion");
});

test("only side-effect-free decision calls are retried", async () => {
  assert.deepEqual([...IDEMPOTENT_INFERENCE], ["/v1/decisions"]);
  assert.equal(await isLoadShed(busy()), true);
  assert.equal(await isLoadShed(Response.json({ code: "not_ready" }, { status: 503 })), false);
  assert.equal(await isLoadShed(new Response("oops", { status: 503 })), false);
  assert.equal(await isLoadShed(Response.json({ code: "model_busy" }, { status: 500 })), false);
});
