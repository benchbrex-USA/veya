/**
 * Public edge for the Cloudflare Container.
 *
 * One named instance, veya-core. The container has no public internet and
 * reads the checkpoint through the private R2 handler below. That handler is
 * not mounted on the public fetch path.
 */

import { Container, ContainerProxy, getContainer } from "@cloudflare/containers";

export { ContainerProxy };

import { ARTIFACT_HOST, handleArtifact } from "./artifact_route.mjs";
import {
  CHECKPOINT_BYTES,
  CHECKPOINT_SHA256,
  MANIFEST_SHA256,
  RELEASE_ID,
  SOURCE_COMMIT,
  SOURCE_DIRTY,
} from "./build-meta.mjs";
import {
  INSTANCE_NAME,
  adminVersion,
  edgeRateLimit,
  failure,
  forwardHeaders,
  guardRequest,
  notReady,
  requestId,
  sleepAfterFor,
  containerEnvOverrides,
  stamp,
} from "./edge_policy.mjs";

const buckets = new Map();

export class VeyaContainer extends Container {
  defaultPort = 8000;
  requiredPorts = [8000];
  // The library requests http://${pingEndpoint} and treats any HTTP response
  // as "port open". /readyz is still the route. Inference below waits for 200.
  pingEndpoint = "127.0.0.1/readyz";
  enableInternet = false;

  constructor(ctx, env) {
    super(ctx, env);
    // Per-environment idle timeout (VEYA_SLEEP_AFTER). The base class reads
    // this.sleepAfter after a microtask, so assigning it here takes effect.
    this.sleepAfter = sleepAfterFor(env);
    this.envVars = { ...this.envVars, ...containerEnvOverrides(env) };
  }
  envVars = {
    VEYA_CHECKPOINT_URL: `http://${ARTIFACT_HOST}/releases/${RELEASE_ID}/model.safetensors`,
    VEYA_CHECKPOINT_SHA256: CHECKPOINT_SHA256,
    VEYA_CHECKPOINT_BYTES: CHECKPOINT_BYTES,
    VEYA_RELEASE_ID: RELEASE_ID,
    VEYA_MANIFEST_SHA256: MANIFEST_SHA256,
    // The gateway keys its limiter on X-Veya-Tenant (set by the production Worker,
    // stripped from client requests by forwardHeaders) instead of the shared
    // Worker->container source address.
    VEYA_EDGE_MANAGED: "1",
  };
}

VeyaContainer.outboundByHost = {
  [ARTIFACT_HOST]: (request, env) => handleArtifact(request, env),
};

const BUILD = {
  sourceCommit: SOURCE_COMMIT,
  sourceDirty: SOURCE_DIRTY,
  release: RELEASE_ID,
  manifestSha256: MANIFEST_SHA256,
  checkpointSha256: CHECKPOINT_SHA256,
};

export default {
  async fetch(request, env) {
    const id = requestId(request);
    const url = new URL(request.url);
    const blocked = guardRequest(request, env, buckets);
    if (blocked) return blocked;
    const limited = await edgeRateLimit(request, env);
    if (limited) return limited;
    if (url.pathname === "/__admin/version") return adminVersion(env, BUILD);
    if (!env || !env.VEYA_CONTAINER) {
      return failure(id, 503, "container_unavailable");
    }
    try {
      const container = getContainer(env.VEYA_CONTAINER, INSTANCE_NAME);
      if (url.pathname === "/api/infer") {
        const probe = await container.fetch(new Request(new URL("/readyz", url.origin), { method: "GET" }));
        const waiting = notReady(probe.status);
        if (waiting) return waiting;
      }
      const outbound = new Request(request, { headers: forwardHeaders(request, id) });
      return stamp(await container.fetch(outbound));
    } catch {
      return failure(id, 502, "container_unavailable");
    }
  },
};
