import assert from "node:assert/strict";
import test from "node:test";

import { selectReplica } from "../worker/pool.mjs";

test("the production pool asks for two identical replicas", async () => {
  const seen = [];
  const binding = { name: "VEYA_CONTAINER" };
  const chosen = await selectReplica(binding, {
    count: 2,
    async pick(candidate, count) {
      seen.push([candidate, count]);
      return { id: "replica" };
    },
  });
  assert.equal(chosen.id, "replica");
  assert.deepEqual(seen, [[binding, 2]]);
  await assert.rejects(selectReplica(binding, { count: 1, pick() {} }), /two replicas/);
  await assert.rejects(selectReplica(null, { count: 2, pick() {} }), /binding is missing/);
});
