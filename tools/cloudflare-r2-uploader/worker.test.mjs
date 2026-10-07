import assert from "node:assert/strict";
import test from "node:test";

import { OBJECT_KEY, authorized, partNumberFrom, partsFrom, uploadIdFrom } from "./worker.mjs";

const TOKEN = "a".repeat(40);

test("only the fixed checkpoint key is allowed", () => {
  assert.equal(OBJECT_KEY, "releases/0.3.5-audit-baseline.1/model.safetensors");
  assert.equal(OBJECT_KEY.includes(".."), false);
});

test("bearer comparison rejects missing and mismatched tokens", () => {
  const ok = new Request("https://uploader.example/mpu/create", {
    headers: { authorization: `Bearer ${TOKEN}` },
  });
  assert.equal(authorized(ok, TOKEN), true);
  assert.equal(authorized(ok, `${TOKEN}x`), false);
  assert.equal(authorized(new Request("https://uploader.example/"), TOKEN), false);
  assert.equal(authorized(ok, "short"), false);
});

test("part and upload identifiers stay inside the fixed upload", () => {
  assert.equal(partNumberFrom("1"), 1);
  assert.equal(partNumberFrom("0"), null);
  assert.equal(partNumberFrom("10001"), null);
  assert.equal(uploadIdFrom("../releases/other"), null);
  assert.equal(uploadIdFrom("abc/def+123="), "abc/def+123=");
  const parts = partsFrom([
    { partNumber: 2, etag: "b" },
    { partNumber: 1, etag: "a" },
  ]);
  assert.deepEqual(parts, [
    { partNumber: 1, etag: "a" },
    { partNumber: 2, etag: "b" },
  ]);
  assert.equal(partsFrom([{ partNumber: 1, etag: "a" }, { partNumber: 1, etag: "b" }]), null);
});
