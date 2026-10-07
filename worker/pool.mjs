/**
 * Fixed pool of identical Veya Core replicas.
 * getRandom is injected in tests. The default import is the Cloudflare helper.
 */

export async function selectReplica(binding, options = {}) {
  const count = options.count ?? 2;
  if (!binding) throw new Error("container binding is missing");
  if (!Number.isInteger(count) || count < 2) {
    throw new Error("production pool requires two replicas");
  }
  const pick = options.pick || (await import("@cloudflare/containers")).getRandom;
  return pick(binding, count);
}
