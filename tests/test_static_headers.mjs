import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SECURITY_HEADERS } from "../worker/edge_policy.mjs";

// Workers Static Assets serve site/dist directly (no Worker in the path), so the
// security headers for HTML/CSS/JS come from site/public/_headers, which Vite
// copies into site/dist.
function rulesFor(pattern) {
  const lines = readFileSync(new URL("../site/public/_headers", import.meta.url), "utf8").split("\n");
  const start = lines.indexOf(pattern);
  assert.ok(start >= 0, `no ${pattern} block`);
  const headers = {};
  for (const line of lines.slice(start + 1)) {
    if (!/^\s+\S/.test(line)) break;
    const [name, ...rest] = line.trim().split(":");
    headers[name.toLowerCase()] = rest.join(":").trim();
  }
  return headers;
}

test("static assets carry the same HSTS policy as Worker responses", () => {
  const headers = rulesFor("/*");
  assert.equal(headers["strict-transport-security"], SECURITY_HEADERS["strict-transport-security"]);
  assert.equal(headers["strict-transport-security"], "max-age=63072000; includeSubDomains");
  assert.ok(!/preload/.test(headers["strict-transport-security"]), "no preload until every subdomain is HTTPS-only");
  assert.equal(headers["x-content-type-options"], "nosniff");
  assert.equal(headers["x-frame-options"], "DENY");
  assert.ok(headers["content-security-policy"].includes("frame-ancestors 'none'"));
});

test("staging serves the built site from site/dist", () => {
  const text = readFileSync(new URL("../wrangler.staging.jsonc", import.meta.url), "utf8").replace(/^\s*\/\/.*$/gm, "");
  assert.equal(JSON.parse(text).assets.directory, "./site/dist");
});
