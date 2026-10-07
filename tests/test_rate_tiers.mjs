import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { authenticate, sha256Hex } from "../worker/api_keys.mjs";
import { createHandler, createReadyCache } from "../worker/production_edge.mjs";
import { DEFAULT_TIER, RATE_TIERS, ratePlan, tierFor } from "../worker/rate_tiers.mjs";

const KEY = "vk_test_customer_key";
const BUILD = { release: "r", checkpointSha256: "c".repeat(64), manifestSha256: "m".repeat(64) };

function limiter(name, log, success = true) {
  return { async limit({ key }) { log.push({ name, key }); return { success }; } };
}

function bindings(log, overrides = {}) {
  return {
    VEYA_RATE_INFERENCE: limiter("VEYA_RATE_INFERENCE", log),
    VEYA_RATE_HEAVY: limiter("VEYA_RATE_HEAVY", log),
    VEYA_RATE_SITE: limiter("VEYA_RATE_SITE", log),
    VEYA_RATE_SITE_HEAVY: limiter("VEYA_RATE_SITE_HEAVY", log),
    ...overrides,
  };
}

async function db({ tier = null, legacy = false, fail = null } = {}) {
  const hash = await sha256Hex(KEY);
  const queries = [];
  return {
    queries,
    prepare(sql) {
      queries.push(sql);
      return {
        bind(...args) {
          return {
            async first() {
              if (fail) throw new Error(fail);
              if (legacy && sql.includes("rate_tier")) throw new Error("D1_ERROR: no such column: p.rate_tier: SQLITE_ERROR");
              if (args[0] !== hash) return null;
              const row = { id: "key_1", organization_id: "org_1", project_id: "proj_1", key_hash: hash,
                scopes: JSON.stringify(["decisions:invoke"]), expires_at: null, revoked_at: null };
              if (sql.includes("rate_tier")) row.rate_tier = tier;
              return row;
            },
            async run() { return { success: true }; },
          };
        },
      };
    },
  };
}

const container = {
  idFromName: (name) => name,
  get: () => ({ async fetch(request) {
    const path = new URL(request.url).pathname;
    return path === "/readyz" ? new Response("{}") : Response.json({ route: "packed" });
  } }),
};

function decision(primitive = "choice") {
  return new Request("https://api.example/v1/decisions", {
    method: "POST",
    headers: { authorization: "Bearer " + KEY, "content-type": "application/json" },
    body: JSON.stringify({ primitive, criteria: { a: "x", b: "y" } }),
  });
}

async function call(tier, primitive, extra = {}) {
  const log = [];
  const env = { VEYA_DB: await db({ tier }), VEYA_CONTAINER: container, ...bindings(log), ...extra };
  const handle = createHandler(BUILD, { readyCache: createReadyCache(), random: () => 0 });
  const response = await handle(decision(primitive), env, { waitUntil() {} });
  await response.text();
  return { response, log };
}

test("tiers: unknown or missing names fall back to default; site exists", () => {
  assert.equal(tierFor(null), DEFAULT_TIER);
  assert.equal(tierFor("platinum"), DEFAULT_TIER);
  assert.equal(tierFor("__proto__"), DEFAULT_TIER);
  assert.equal(tierFor("site"), "site");
  assert.deepEqual(Object.keys(RATE_TIERS).sort(), ["default", "site"]);
});

test("default tier is unchanged: per API key on VEYA_RATE_INFERENCE / VEYA_RATE_HEAVY", async () => {
  let { response, log } = await call(null, "choice");
  assert.equal(response.status, 200);
  assert.deepEqual(log, [{ name: "VEYA_RATE_INFERENCE", key: "key_1" }]);
  ({ log } = await call(null, "noul"));
  assert.deepEqual(log, [{ name: "VEYA_RATE_HEAVY", key: "key_1" }]);
});

test("site tier counts per project on the VEYA_RATE_SITE bindings", async () => {
  let { response, log } = await call("site", "choice");
  assert.equal(response.status, 200);
  assert.deepEqual(log, [{ name: "VEYA_RATE_SITE", key: "project:proj_1" }]);
  ({ log } = await call("site", "score"));
  assert.deepEqual(log, [{ name: "VEYA_RATE_SITE_HEAVY", key: "project:proj_1" }]);
});

test("a site-tier 429 is the usual rate_limited envelope, tagged with the tier", async () => {
  const log = [];
  const { response } = await call("site", "choice", { VEYA_RATE_SITE: limiter("VEYA_RATE_SITE", log, false) });
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("retry-after"), "60");
  assert.equal(response.headers.get("x-veya-rate-tier"), "site");
});

test("a tier whose bindings are missing falls back to the stricter default, per key", () => {
  const log = [];
  const env = { VEYA_RATE_INFERENCE: limiter("VEYA_RATE_INFERENCE", log), VEYA_RATE_HEAVY: limiter("VEYA_RATE_HEAVY", log) };
  const identity = { keyId: "key_9", projectId: "proj_9", rateTier: "site" };
  assert.deepEqual({ ...ratePlan(env, identity, false), binding: undefined }, { tier: "default", binding: undefined, key: "key_9" });
  assert.equal(ratePlan(env, identity, true).binding, env.VEYA_RATE_HEAVY);
  assert.equal(ratePlan({}, identity, false).binding, null);
});

test("authenticate reads the tier, and falls back to the key-only query before migration 0002", async () => {
  let identity = await authenticate(await db({ tier: "site" }), KEY);
  assert.equal(identity.rateTier, "site");
  identity = await authenticate(await db({ tier: null }), KEY);
  assert.equal(identity.rateTier, null);
  const legacy = await db({ legacy: true });
  identity = await authenticate(legacy, KEY);
  assert.equal(identity.projectId, "proj_1");
  assert.equal(identity.rateTier, null);
  assert.equal(legacy.queries.length, 2);
  // Other D1 failures still surface (the edge answers 503 auth_unavailable).
  await assert.rejects(authenticate(await db({ fail: "D1_ERROR: network connection lost" }), KEY));
});

test("production config binds the site tier with distinct namespaces sized to the pool", () => {
  const text = readFileSync(new URL("../wrangler.production.jsonc", import.meta.url), "utf8").replace(/^\s*\/\/.*$/gm, "");
  const limits = Object.fromEntries(JSON.parse(text).ratelimits.map((r) => [r.name, r]));
  assert.deepEqual(limits.VEYA_RATE_INFERENCE.simple, { limit: 20, period: 60 });
  assert.deepEqual(limits.VEYA_RATE_HEAVY.simple, { limit: 6, period: 60 });
  assert.deepEqual(limits.VEYA_RATE_SITE.simple, { limit: 150, period: 60 });
  assert.deepEqual(limits.VEYA_RATE_SITE_HEAVY.simple, { limit: 45, period: 60 });
  assert.equal(new Set(Object.values(limits).map((r) => r.namespace_id)).size, 4);
  // 150/min choice is under the ~156/min the two replicas serve before model_busy.
  assert.ok(limits.VEYA_RATE_SITE.simple.limit <= 156);
});
