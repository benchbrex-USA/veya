import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { buildInferenceRequest, choiceRoute } from "../src/contract.ts";
import { BENCHMARK, MAX_FUSION, RELEASE, statusIsGenerallyAvailable } from "../src/release.ts";
import {
  cacheControl,
  generateApiKey,
  maskApiKey,
  securityHeaders,
  sha256Hex,
  surfaceForHost,
  timingSafeEqual,
  toggleFaq,
  verifyTurnstile,
} from "../src/security.ts";

test("choice request matches the gateway object shape", () => {
  const built = buildInferenceRequest({
    primitive: "choice",
    state: "Please refund the duplicate March invoice.",
    instructions: "Which label best fits the text?",
    criteria: [
      { id: "billing", description: "Invoice and refund questions" },
      { id: "access", description: "Login and permission questions" },
    ],
  });
  assert.equal(built.ok, true);
  if (!built.ok) return;
  assert.equal(built.body.primitive, "choice");
  assert.equal("authorization" in built.body, false);
  assert.equal(JSON.stringify(built.body).includes("VEYA_EDGE_TOKEN"), false);
  assert.equal(choiceRoute(2), "packed");
  assert.equal(choiceRoute(20), "shortlist_k8");
});

test("choice rejects a single candidate and duplicate ids", () => {
  const one = buildInferenceRequest({
    primitive: "choice",
    state: "text",
    instructions: "Which label?",
    criteria: [{ id: "only", description: "One" }],
  });
  assert.equal(one.ok, false);
  const duplicate = buildInferenceRequest({
    primitive: "choice",
    state: "text",
    instructions: "Which label?",
    criteria: [
      { id: "Billing", description: "A" },
      { id: "billing", description: "B" },
    ],
  });
  assert.equal(duplicate.ok, false);
});

test("noul omits criteria and score checks labels", () => {
  const noul = buildInferenceRequest({
    primitive: "noul",
    state: "Will this cancel?",
    instructions: "How likely is cancellation?",
    criteria: [],
  });
  assert.equal(noul.ok, true);
  if (noul.ok) assert.equal("criteria" in noul.body, false);
  const score = buildInferenceRequest({
    primitive: "score",
    state: "The account is locked.",
    instructions: "How severe is this?",
    criteria: ["low", "high"],
  });
  assert.equal(score.ok, true);
});

test("release labels are not generally available", () => {
  assert.equal(statusIsGenerallyAvailable(RELEASE.choice.status), false);
  assert.equal(RELEASE.noul.status, "RESEARCH ONLY");
  assert.equal(RELEASE.score.status, "RESEARCH ONLY");
  assert.equal(RELEASE.automation, "DISABLED");
  assert.equal(MAX_FUSION.promoted, false);
  assert.equal(BENCHMARK.dataset, "CLINC150");
});

test("api keys are masked and hashed", async () => {
  const key = generateApiKey("test");
  assert.match(key, /^veya_test_[A-Za-z0-9_-]{43}$/);
  const masked = maskApiKey(key);
  assert.equal(masked.startsWith(key.slice(0, 14)), true);
  assert.equal(masked.endsWith(key.slice(-4)), true);
  assert.ok(masked.length < key.length);
  const hash = await sha256Hex(key);
  assert.notEqual(hash, key);
  assert.equal(timingSafeEqual(hash, hash), true);
  assert.equal(timingSafeEqual(hash, "0".repeat(hash.length)), false);
});

test("security headers do not allow inline scripts and HSTS stays off", () => {
  const headers = securityHeaders(false);
  assert.equal(headers["content-security-policy"].includes("unsafe-inline"), false);
  assert.equal(headers["content-security-policy"].includes("frame-ancestors 'none'"), true);
  assert.equal(headers["x-content-type-options"], "nosniff");
  assert.equal("strict-transport-security" in headers, false);
  assert.equal(cacheControl("/assets/app.js"), "public, max-age=31536000, immutable");
  assert.equal(cacheControl("/api/playground"), "no-store");
});

test("hosts select the product surface", () => {
  assert.equal(surfaceForHost("veyadynamics.com"), "marketing");
  assert.equal(surfaceForHost("app.veyadynamics.com"), "console");
  assert.equal(surfaceForHost("docs-staging.veyadynamics.com"), "docs");
  assert.equal(surfaceForHost("api.veyadynamics.com"), "api");
});

test("faq toggle and turnstile failure are closed", async () => {
  assert.equal(toggleFaq(null, "a"), "a");
  assert.equal(toggleFaq("a", "a"), null);
  const passed = await verifyTurnstile("secret", "token", "203.0.113.8", async () => new Response(JSON.stringify({ success: false }), { status: 200 }));
  assert.equal(passed, false);
  const empty = await verifyTurnstile("", "token", "203.0.113.8");
  assert.equal(empty, false);
});

test("the model image does not copy this site and the UI does not name the reference product", () => {
  const dockerfile = readFileSync(new URL("../../deploy/cloudflare/Dockerfile", import.meta.url), "utf8");
  assert.equal(dockerfile.includes("site/"), false);
  assert.equal(dockerfile.includes("vite"), false);
  const source = readFileSync(new URL("../src/marketing.tsx", import.meta.url), "utf8");
  for (const banned of ["Tinker", "Thinking Machines", "Iowan", "GT America"]) {
    assert.equal(source.includes(banned), false, banned);
  }
});
