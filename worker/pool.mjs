/**
 * Fixed pool of identical Veya Core replicas.
 * getRandom is injected in tests. The default import is the Cloudflare helper.
 */

import { discard } from "./body_hygiene.mjs";

export const REPLICA_COUNT = 2;

export function replicaName(index) {
  return `instance-${index}`;
}

export function replicaOrder(count, random = Math.random) {
  const first = Math.floor(random() * count);
  return Array.from({ length: count }, (_, offset) => (first + offset) % count);
}

export function stubFor(binding, index) {
  return binding.get(binding.idFromName(replicaName(index)));
}

export async function fetchFromPool(binding, requestFactory, options = {}) {
  const count = options.count ?? REPLICA_COUNT;
  if (!binding || typeof binding.idFromName !== "function" || typeof binding.get !== "function") {
    throw new Error("container binding is missing");
  }
  if (!Number.isInteger(count) || count < 2) {
    throw new Error("production pool requires two replicas");
  }
  const attempts = [];
  for (const index of replicaOrder(count, options.random)) {
    const stub = stubFor(binding, index);
    try {
      const response = await requestFactory(stub, index);
      const accept = options.accept || ((candidate) => candidate && candidate.ok);
      if (response && accept(response)) return { index, response, attempts };
      // A rejected response is never handed to the client: release it so the
      // container's in-flight count drops (see body_hygiene.mjs).
      await discard(response);
      attempts.push({ index, status: response ? response.status : 0 });
    } catch {
      attempts.push({ index, status: 0 });
    }
  }
  const error = new Error("no healthy replica");
  error.attempts = attempts;
  throw error;
}

export async function selectReplica(binding, options = {}) {
  const count = options.count ?? 2;
  if (!binding) throw new Error("container binding is missing");
  if (!Number.isInteger(count) || count < 2) {
    throw new Error("production pool requires two replicas");
  }
  const pick = options.pick || (await import("@cloudflare/containers")).getRandom;
  return pick(binding, count);
}
