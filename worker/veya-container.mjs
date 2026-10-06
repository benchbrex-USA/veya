/**
 * Public edge for the Cloudflare Container.
 *
 * One named instance, veya-core. The container has no public internet and
 * reads the checkpoint through the private R2 handler below. That handler is
 * not mounted on the public fetch path.
 */

import { Container, getContainer } from "@cloudflare/containers";

import { ARTIFACT_HOST, handleArtifact } from "./artifact_route.mjs";
import {
  INSTANCE_NAME,
  adminVersion,
  edgeRateLimit,
  failure,
  forwardHeaders,
  guardRequest,
  requestId,
} from "./edge_policy.mjs";

const buckets = new Map();
const RELEASE = "0.3.5-audit-baseline.1";

export class VeyaContainer extends Container {
  defaultPort = 8000;
  requiredPorts = [8000];
  sleepAfter = "30m";
  enableInternet = false;
  envVars = {
    VEYA_CHECKPOINT_URL: `http://${ARTIFACT_HOST}/releases/${RELEASE}/model.safetensors`,
    VEYA_CHECKPOINT_SHA256: "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    VEYA_CHECKPOINT_BYTES: "842609210",
    VEYA_RELEASE_ID: RELEASE,
  };
}

VeyaContainer.outboundByHost = {
  [ARTIFACT_HOST]: (request, env) => handleArtifact(request, env),
};

export default {
  async fetch(request, env) {
    const id = requestId(request);
    const blocked = guardRequest(request, env, buckets);
    if (blocked) return blocked;
    const limited = await edgeRateLimit(request, env);
    if (limited) return limited;
    if (new URL(request.url).pathname === "/__admin/version") return adminVersion(env);
    if (!env || !env.VEYA_CONTAINER) {
      return failure(id, 503, "container_unavailable");
    }
    try {
      const container = getContainer(env.VEYA_CONTAINER, INSTANCE_NAME);
      const outbound = new Request(request, { headers: forwardHeaders(request, id) });
      return await container.fetch(outbound);
    } catch {
      return failure(id, 502, "container_unavailable");
    }
  },
};
