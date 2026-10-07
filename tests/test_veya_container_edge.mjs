import assert from "node:assert/strict";
import test from "node:test";

import { approvedArtifactKey, handleArtifact } from "../worker/artifact_route.mjs";
import {
  DEMO_IDENTITY,
  INFER_MAX_BODY_BYTES,
  INFER_RATE_LIMIT,
  INSTANCE_NAME,
  adminVersion,
  edgeHealth,
  edgeRateLimit,
  forwardHeaders,
  guardRequest,
  notReady,
  requestWeight,
} from "../worker/edge_policy.mjs";

const DEMO = "veya-demo-test-token";
const EDGE = "veya-edge-test-token";

function request(path, { method = "GET", headers = {}, body } = {}) {
  return new Request("https://veya.example" + path, { method, headers, body });
}

test("edge health does not describe an origin", async () => {
  const response = guardRequest(request("/__edge_health"), {}, new Map());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { edge: "ok", backend: "cloudflare-container" });
  assert.equal(INSTANCE_NAME, "veya-core");
});

test("inference requires the demo credential, not the infrastructure token", async () => {
  const missing = guardRequest(request("/api/infer", { method: "POST", body: "{}" }), { VEYA_EDGE_TOKEN: EDGE }, new Map());
  assert.equal(missing.status, 503);
  const denied = guardRequest(
    request("/api/infer", { method: "POST", body: "{}", headers: { authorization: "Bearer " + EDGE } }),
    { VEYA_EDGE_TOKEN: EDGE, VEYA_DEMO_TOKEN: DEMO },
    new Map(),
  );
  assert.equal(denied.status, 401);
  const allowed = guardRequest(
    request("/api/infer", {
      method: "POST",
      body: "{}",
      headers: { authorization: "Bearer " + DEMO, "content-length": "2" },
    }),
    { VEYA_EDGE_TOKEN: EDGE, VEYA_DEMO_TOKEN: DEMO },
    new Map(),
  );
  assert.equal(allowed, null);
  const adminDenied = guardRequest(
    request("/__admin/version", { headers: { authorization: "Bearer " + DEMO } }),
    { VEYA_EDGE_TOKEN: EDGE, VEYA_DEMO_TOKEN: DEMO },
    new Map(),
  );
  assert.equal(adminDenied.status, 401);
  const adminAllowed = guardRequest(
    request("/__admin/version", { headers: { authorization: "Bearer " + EDGE } }),
    { VEYA_EDGE_TOKEN: EDGE, VEYA_DEMO_TOKEN: DEMO },
    new Map(),
  );
  assert.equal(adminAllowed, null);
});

test("oversized and disallowed requests stop before the container", async () => {
  const huge = guardRequest(
    request("/api/infer", {
      method: "POST",
      headers: { authorization: "Bearer " + DEMO, "content-length": String(INFER_MAX_BODY_BYTES + 1) },
    }),
    { VEYA_DEMO_TOKEN: DEMO },
    new Map(),
  );
  assert.equal(huge.status, 413);
  assert.equal(INFER_MAX_BODY_BYTES, 32768);
  const method = guardRequest(request("/api/infer", { method: "PUT" }), { VEYA_DEMO_TOKEN: DEMO }, new Map());
  assert.equal(method.status, 405);
  const waiting = notReady(503);
  assert.equal(waiting.status, 503);
  assert.equal((await waiting.json()).code, "not_ready");
  assert.equal(notReady(200), null);
});

test("forwarded headers drop the edge credential", () => {
  const headers = forwardHeaders(
    request("/api/infer", {
      method: "POST",
      headers: { authorization: "Bearer " + DEMO, connection: "close", host: "evil.example" },
    }),
    "req-12345678",
  );
  assert.equal(headers.get("authorization"), null);
  assert.equal(headers.get("connection"), null);
  assert.equal(headers.get("host"), null);
  assert.equal(headers.get("x-request-id"), "req-12345678");
});

test("health helper stays minimal", async () => {
  const body = await edgeHealth().json();
  assert.deepEqual(Object.keys(body).sort(), ["backend", "edge"]);
});

test("the local fallback limiter keys on the demo identity and sets retry-after", () => {
  const buckets = new Map();
  const env = { VEYA_DEMO_TOKEN: DEMO };
  let last = null;
  for (let index = 0; index < INFER_RATE_LIMIT + 1; index += 1) {
    last = guardRequest(
      request("/api/infer", {
        method: "POST",
        headers: {
          authorization: "Bearer " + DEMO,
          "content-length": "2",
          "cf-connecting-ip": "203.0.113." + (index % 50),
          "x-forwarded-for": "198.51.100." + index,
        },
      }),
      env,
      buckets,
    );
  }
  assert.equal(last.status, 429);
  assert.equal(last.headers.get("retry-after"), "60");
  assert.equal(buckets.has(DEMO_IDENTITY), true);
  const skipped = guardRequest(
    request("/api/infer", {
      method: "POST",
      headers: { authorization: "Bearer " + DEMO, "content-length": "2" },
    }),
    { VEYA_DEMO_TOKEN: DEMO, VEYA_RATE_INFERENCE: { async limit() { return { success: true }; } } },
    buckets,
  );
  assert.equal(skipped, null);
});

test("high-n choice and other primitives are heavy", async () => {
  const candidates = Object.fromEntries(
    Array.from({ length: 20 }, (_, index) => ["c" + index, "label"]),
  );
  const heavy = await requestWeight(
    request("/api/infer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ primitive: "choice", criteria: candidates }),
    }),
  );
  assert.equal(heavy, "heavy");
  const score = await requestWeight(
    request("/api/infer", {
      method: "POST",
      body: JSON.stringify({ primitive: "score" }),
    }),
  );
  assert.equal(score, "heavy");
  const small = await requestWeight(
    request("/api/infer", {
      method: "POST",
      body: JSON.stringify({ primitive: "choice", criteria: { a: "one", b: "two" } }),
    }),
  );
  assert.equal(small, "inference");
});

test("cloudflare rate limit keys on the demo identity", async () => {
  const seen = [];
  const denied = await edgeRateLimit(
    request("/api/infer", { method: "POST", body: "{}", headers: { "cf-connecting-ip": "203.0.113.10" } }),
    { VEYA_RATE_INFERENCE: { async limit({ key }) { seen.push(key); return { success: false }; } } },
  );
  assert.deepEqual(seen, [DEMO_IDENTITY]);
  assert.equal(denied.status, 429);
  assert.equal(denied.headers.get("retry-after"), "60");
  const allowed = await edgeRateLimit(request("/healthz"), {
    VEYA_RATE_INFERENCE: { async limit() { throw new Error("health must not be limited"); } },
  });
  assert.equal(allowed, null);
});

test("admin version exposes release identifiers only", async () => {
  const body = await adminVersion({
    CF_VERSION_METADATA: { id: "version-1" },
    VEYA_SOURCE_COMMIT: "abc123",
    VEYA_SOURCE_DIRTY: "true",
    VEYA_RELEASE_ID: "0.3.5-audit-baseline.1",
    VEYA_MANIFEST_SHA256: "m".repeat(64),
    VEYA_CHECKPOINT_SHA256: "c".repeat(64),
    VEYA_EDGE_TOKEN: EDGE,
  }).json();
  assert.equal(body.worker_version_id, "version-1");
  assert.equal(body.release, "0.3.5-audit-baseline.1");
  assert.equal(body.source_dirty, true);
  assert.equal(JSON.stringify(body).includes(EDGE), false);
  assert.equal(JSON.stringify(body).includes(DEMO), false);
});

test("the artifact handler allows only the configured release object", async () => {
  const release = "0.3.5-audit-baseline.1";
  const good = new URL(
    "http://veya-r2.internal/releases/" + release + "/model.safetensors",
  );
  assert.equal(
    approvedArtifactKey(good, release),
    "releases/" + release + "/model.safetensors",
  );
  assert.equal(approvedArtifactKey(new URL("http://veya-r2.internal/releases/other/model.safetensors"), release), null);
  assert.equal(approvedArtifactKey(new URL("http://veya-r2.internal/releases/" + release + "/../model.safetensors"), release), null);
  assert.equal(approvedArtifactKey(new URL("http://veya-r2.internal/releases/" + release + "/%2e%2e/model.safetensors"), release), null);
  const seen = [];
  const env = {
    VEYA_RELEASE_ID: release,
    VEYA_ARTIFACTS: {
      async get(key) {
        seen.push(key);
        return { size: 4, body: new ReadableStream() };
      },
    },
  };
  const ok = await handleArtifact(new Request(good), env);
  assert.equal(ok.status, 200);
  assert.deepEqual(seen, ["releases/" + release + "/model.safetensors"]);
  const post = await handleArtifact(new Request(good, { method: "POST" }), env);
  assert.equal(post.status, 405);
  const missing = await handleArtifact(new Request(good), {
    VEYA_RELEASE_ID: release,
    VEYA_ARTIFACTS: { async get() { return null; } },
  });
  assert.equal(missing.status, 404);
});
