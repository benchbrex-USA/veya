import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { DEFAULT_SLEEP_AFTER, sleepAfterFor } from "../worker/edge_policy.mjs";

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

function jsonc(path) {
  const text = readFileSync(new URL(path, import.meta.url), "utf8");
  return JSON.parse(text.replace(/^\s*\/\/.*$/gm, ""));
}

test("sleepAfterFor accepts 1m..24h and falls back to 8h otherwise", () => {
  assert.equal(DEFAULT_SLEEP_AFTER, "8h");
  assert.equal(sleepAfterFor({ VEYA_SLEEP_AFTER: "15m" }), "15m");
  assert.equal(sleepAfterFor({ VEYA_SLEEP_AFTER: "1h" }), "1h");
  assert.equal(sleepAfterFor({ VEYA_SLEEP_AFTER: " 90s " }), "90s");
  for (const bad of [undefined, "", "30s", "25h", "0m", "1d", "15 m", "abc", "-5m"]) {
    assert.equal(sleepAfterFor({ VEYA_SLEEP_AFTER: bad }), "8h", String(bad));
  }
  assert.equal(sleepAfterFor(undefined), "8h");
});

test("the container class applies VEYA_SLEEP_AFTER to its idle timer", async () => {
  const { VeyaContainer } = await import("../worker/veya-container.mjs");
  for (const [value, seconds] of [["15m", 900], ["1h", 3600], [undefined, 8 * 3600]]) {
    const instance = new VeyaContainer(durableObjectContext(), value === undefined ? {} : { VEYA_SLEEP_AFTER: value });
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(instance.sleepAfter, value || "8h");
    const remaining = Math.round((instance.sleepAfterMs - Date.now()) / 1000);
    assert.ok(Math.abs(remaining - seconds) <= 2, `${value}: ${remaining}s`);
  }
});

test("staging sleeps after 15m and production after 1h", () => {
  assert.equal(jsonc("../wrangler.staging.jsonc").vars.VEYA_SLEEP_AFTER, "15m");
  assert.equal(jsonc("../wrangler.production.jsonc").vars.VEYA_SLEEP_AFTER, "1h");
  assert.equal(jsonc("../wrangler.jsonc").vars.VEYA_SLEEP_AFTER, undefined, "the plain proxy has no container");
});

test("VEYA_MAX_INFLIGHT passes through to the container env; release pins cannot be overridden", async () => {
  const { containerEnvOverrides } = await import("../worker/edge_policy.mjs");
  assert.deepEqual(containerEnvOverrides({ VEYA_MAX_INFLIGHT: "3" }), { VEYA_MAX_INFLIGHT: "3" });
  for (const bad of [undefined, "", "0", "9", "12", "two"]) {
    assert.deepEqual(containerEnvOverrides({ VEYA_MAX_INFLIGHT: bad }), {}, String(bad));
  }
  const { VeyaContainer } = await import("../worker/veya-container.mjs");
  const instance = new VeyaContainer(durableObjectContext(), {
    VEYA_MAX_INFLIGHT: "3",
    VEYA_MANIFEST_SHA256: "f".repeat(64),
  });
  assert.equal(instance.envVars.VEYA_MAX_INFLIGHT, "3");
  assert.notEqual(instance.envVars.VEYA_MANIFEST_SHA256, "f".repeat(64));
  assert.equal(instance.envVars.VEYA_EDGE_MANAGED, "1");
  const plain = new VeyaContainer(durableObjectContext(), {});
  assert.equal(plain.envVars.VEYA_MAX_INFLIGHT, undefined);
});
