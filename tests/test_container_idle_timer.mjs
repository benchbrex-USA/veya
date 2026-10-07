// The container idle timer (sleepAfter) and the in-flight bookkeeping it depends on.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { DEFAULT_INFLIGHT_STALE_MS, inflightStaleMs, sleepAfterSeconds } from "../worker/edge_policy.mjs";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "cloudflare:workers") {
      return { shortCircuit: true, url: new URL("./cloudflare-workers-do-stub.mjs", import.meta.url).href };
    }
    if ((specifier.startsWith("./") || specifier.startsWith("../")) && context.parentURL) {
      const exact = fileURLToPath(new URL(specifier, context.parentURL));
      if (!existsSync(exact) && existsSync(exact + ".js")) {
        return { shortCircuit: true, url: new URL(specifier + ".js", context.parentURL).href };
      }
    }
    return nextResolve(specifier, context);
  },
});

function durableObjectContext() {
  const kv = new Map();
  return {
    id: { toString: () => "test" },
    container: { running: false },
    blockConcurrencyWhile: (fn) => fn(),
    storage: {
      get: async (key) => kv.get(key),
      put: async (key, value) => { kv.set(key, value); },
      delete: async (key) => kv.delete(key),
      getAlarm: async () => null,
      setAlarm: async () => {},
      deleteAlarm: async () => {},
      sync: async () => {},
      kv: { get: () => undefined, put: () => {}, delete: () => {} },
      sql: { exec: () => Object.assign([], { toArray: () => [], one: () => null }) },
    },
  };
}

async function container(env = { VEYA_SLEEP_AFTER: "15m" }) {
  const { VeyaContainer } = await import("../worker/veya-container.mjs");
  const instance = new VeyaContainer(durableObjectContext(), env);
  await new Promise((resolve) => setTimeout(resolve, 0));
  return instance;
}

function captureLogs(fn) {
  const lines = [];
  const { log, warn } = console;
  console.log = (line) => lines.push(JSON.parse(line));
  console.warn = (line) => lines.push(JSON.parse(line));
  try {
    return { result: fn(), lines };
  } finally {
    console.log = log;
    console.warn = warn;
  }
}

const MIN = 60_000;

test("library contract: a request stays in flight until its body is piped, and in-flight requests block sleep", () => {
  // Pinned evidence for worker/body_hygiene.mjs. If the library changes, re-check the analysis.
  const pkg = JSON.parse(readFileSync(new URL("../node_modules/@cloudflare/containers/package.json", import.meta.url), "utf8"));
  assert.equal(pkg.version, "0.3.7");
  const source = readFileSync(new URL("../node_modules/@cloudflare/containers/dist/lib/container.js", import.meta.url), "utf8");
  assert.match(source, /this\.inflightRequests\+\+;/);
  assert.match(source, /new IdentityTransformStream\(\);\s*res\.body\?\.pipeTo\(writable\)\.finally\(\(\) => \{\s*this\.decrementInflight\(\);/);
  assert.match(source, /isActivityExpired\(\) \{\s*if \(this\.inflightRequests > 0\) \{\s*this\.renewActivityTimeout\(\);\s*return false;/);
  assert.match(source, /if \(this\.isActivityExpired\(\)\) \{\s*await this\.onActivityExpired\(\);/);
  assert.match(source, /decrementInflight\(\) \{/);
});

test("an idle container with nothing in flight expires after sleepAfter", async () => {
  const c = await container();
  c.inflightRequests = 0;
  c.lastActivityAt = Date.now() - 16 * MIN;
  c.sleepAfterMs = Date.now() - MIN;
  const { result, lines } = captureLogs(() => c.isActivityExpired());
  assert.equal(result, true);
  assert.equal(lines.at(-1).event, "container_activity");
  assert.equal(lines.at(-1).expired, true);
  assert.equal(lines.at(-1).sleep_after, "15m");
});

test("a genuinely running request keeps the container awake", async () => {
  const c = await container();
  c.inflightRequests = 1;
  c.lastActivityAt = Date.now() - 10_000;
  c.sleepAfterMs = Date.now() - MIN;
  const { result, lines } = captureLogs(() => c.isActivityExpired());
  assert.equal(result, false);
  assert.equal(c.inflightRequests, 1);
  assert.ok(!lines.some((line) => line.event === "container_inflight_reset"));
});

test("a leaked in-flight count is reset after VEYA_INFLIGHT_STALE_MS and idle time counts from the last activity", async () => {
  const c = await container();
  // Stuck count, 3 min since the last request started or finished; the library kept renewing the timer.
  c.inflightRequests = 1;
  c.lastActivityAt = Date.now() - 3 * MIN;
  c.sleepAfterMs = Date.now() + 15 * MIN;
  let out = captureLogs(() => c.isActivityExpired());
  assert.equal(out.result, false);
  assert.equal(c.inflightRequests, 0);
  assert.equal(out.lines[0].event, "container_inflight_reset");
  assert.equal(out.lines[0].leaked, 1);
  const remaining = c.sleepAfterMs - Date.now();
  assert.ok(remaining > 11 * MIN && remaining <= 12 * MIN, `expires in ${remaining} ms`);

  // Same leak but the last real activity was 16 min ago: sleeps now.
  c.inflightRequests = 2;
  c.lastActivityAt = Date.now() - 16 * MIN;
  c.sleepAfterMs = Date.now() + 15 * MIN;
  out = captureLogs(() => c.isActivityExpired());
  assert.equal(out.result, true);
  assert.equal(c.inflightRequests, 0);
});

test("requests starting and finishing refresh lastActivityAt", async () => {
  const c = await container();
  c.lastActivityAt = 0;
  c.containerFetch = async () => new Response("ok");
  await c.fetch(new Request("https://veya.internal/readyz"));
  assert.ok(Date.now() - c.lastActivityAt < 1_000);
  c.lastActivityAt = 0;
  c.inflightRequests = 1;
  c.decrementInflight();
  assert.equal(c.inflightRequests, 0);
  assert.ok(Date.now() - c.lastActivityAt < 1_000);
});

test("stale threshold and sleepAfter parsing are bounded", () => {
  assert.equal(inflightStaleMs({}), DEFAULT_INFLIGHT_STALE_MS);
  assert.equal(inflightStaleMs({ VEYA_INFLIGHT_STALE_MS: "1000" }), 30_000);
  assert.equal(inflightStaleMs({ VEYA_INFLIGHT_STALE_MS: "99999999" }), 3_600_000);
  assert.equal(inflightStaleMs({ VEYA_INFLIGHT_STALE_MS: "junk" }), DEFAULT_INFLIGHT_STALE_MS);
  assert.equal(sleepAfterSeconds("15m"), 900);
  assert.equal(sleepAfterSeconds("1h"), 3600);
  assert.ok(Number.isNaN(sleepAfterSeconds("soon")));
});

test("staging /api/infer releases the readiness probe body before forwarding", async () => {
  const { default: worker } = await import("../worker/veya-container.mjs");
  const settled = [];
  function body(label) {
    let sent = false;
    return new ReadableStream({
      pull(controller) {
        if (!sent) { sent = true; controller.enqueue(new TextEncoder().encode("{}")); }
        else { settled.push([label, "read"]); controller.close(); }
      },
      cancel() { settled.push([label, "cancelled"]); },
    }, { highWaterMark: 0 });
  }
  const stub = {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      return new Response(body(path), { status: 200, headers: { "content-type": "application/json" } });
    },
  };
  const env = {
    VEYA_DEMO_TOKEN: "d".repeat(32),
    VEYA_EDGE_TOKEN: "e".repeat(32),
    VEYA_CONTAINER: { idFromName: (name) => name, get: () => stub, getByName: () => stub },
  };
  const tasks = [];
  const response = await worker.fetch(new Request("https://staging.example/api/infer", {
    method: "POST",
    headers: { authorization: "Bearer " + env.VEYA_DEMO_TOKEN, "content-type": "application/json" },
    body: JSON.stringify({ primitive: "choice" }),
  }), env, { waitUntil: (promise) => tasks.push(promise) });
  assert.equal(response.status, 200);
  await response.text();
  await Promise.all(tasks);
  assert.deepEqual(settled, [["/readyz", "cancelled"], ["/api/infer", "read"]]);
});
