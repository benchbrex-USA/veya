import assert from "node:assert/strict";
import test from "node:test";

import { authenticate, hashesEqual, sha256Hex } from "../worker/api_keys.mjs";

test("api key lookup accepts only the matching unrevoked hash", async () => {
  assert.equal(await sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert.equal(hashesEqual("abc", "abd"), false);
  const hash = await sha256Hex("veya_live_example");
  const rows = new Map([[hash, {
    id: "key-1",
    organization_id: "org-a",
    project_id: "project-a",
    key_hash: hash,
    scopes: "[\"decisions:invoke\"]",
    expires_at: "2026-12-01T00:00:00.000Z",
    revoked_at: null,
  }]]);
  const db = {
    prepare() {
      return { bind(value) { return { async first() { return rows.get(value) || null; } }; } };
    },
  };
  const now = new Date("2026-10-07T00:00:00.000Z");
  const identity = await authenticate(db, "veya_live_example", now, "decisions:invoke");
  assert.equal(identity.organizationId, "org-a");
  assert.equal(await authenticate(db, "wrong-key", now, "decisions:invoke"), null);
  rows.get(hash).revoked_at = "2026-10-07T00:00:00.000Z";
  assert.equal(await authenticate(db, "veya_live_example", now, "decisions:invoke"), null);
});
