/**
 * Isolated production edge (`veya-production`, wrangler.production.jsonc).
 *
 * Two identical Veya Core replicas share one R2 checkpoint. Customer calls use
 * D1 API keys. Specialist models are not enabled. All request handling lives in
 * ./production_edge.mjs so it can be tested without the Containers runtime.
 */

import { CHECKPOINT_SHA256, MANIFEST_SHA256, RELEASE_ID, SOURCE_COMMIT, SOURCE_DIRTY } from "./build-meta.mjs";
import { createHandler } from "./production_edge.mjs";

export { ContainerProxy, VeyaContainer } from "./veya-container.mjs";

const BUILD = {
  sourceCommit: SOURCE_COMMIT,
  sourceDirty: SOURCE_DIRTY,
  release: RELEASE_ID,
  manifestSha256: MANIFEST_SHA256,
  checkpointSha256: CHECKPOINT_SHA256,
};

const handle = createHandler(BUILD);

export default {
  async fetch(request, env, ctx) {
    return handle(request, env, ctx);
  },
};
