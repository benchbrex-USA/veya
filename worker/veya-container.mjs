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
import { discard } from "./body_hygiene.mjs";
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
  inflightStaleMs,
  requestId,
  sleepAfterFor,
  sleepAfterSeconds,
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
    this.lastActivityAt = Date.now();
  }

  // ---- idle timer bookkeeping -------------------------------------------
  // The base class only lets the container sleep when inflightRequests is 0
  // (isActivityExpired). The edge releases every response body it does not
  // forward (body_hygiene.mjs), but a client that disconnects mid-request can
  // still strand one inside the library. lastActivityAt records the last time
  // a request started or finished, so a count that has not moved for
  // VEYA_INFLIGHT_STALE_MS (default 2 min, far above the 15 s edge deadline)
  // is treated as leaked instead of keeping the container awake forever.

  async fetch(request) {
    this.lastActivityAt = Date.now();
    return super.fetch(request);
  }

  decrementInflight() {
    this.lastActivityAt = Date.now();
    super.decrementInflight();
  }

  isActivityExpired() {
    const now = Date.now();
    const idleMs = now - this.lastActivityAt;
    if (this.inflightRequests > 0 && idleMs > inflightStaleMs(this.env)) {
      console.warn(JSON.stringify({ event: "container_inflight_reset", leaked: this.inflightRequests, idle_ms: idleMs }));
      this.inflightRequests = 0;
      // Idle time counts from the last real activity, not from the renewals the
      // stuck count caused.
      const windowMs = sleepAfterSeconds(this.sleepAfter) * 1000;
      if (Number.isFinite(windowMs)) this.sleepAfterMs = Math.min(this.sleepAfterMs, this.lastActivityAt + windowMs);
    }
    const expired = super.isActivityExpired();
    // One line per alarm (at most every 3 min): makes the idle timer observable.
    console.log(JSON.stringify({
      event: "container_activity",
      inflight: this.inflightRequests,
      idle_ms: idleMs,
      sleep_after: this.sleepAfter,
      expires_in_ms: this.sleepAfterMs - now,
      expired,
    }));
    return expired;
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
  async fetch(request, env, ctx) {
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
        // Only the status matters. An unread body would keep the request in
        // flight inside the container class and block sleepAfter.
        const released = discard(probe);
        if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(released);
        await released;
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
