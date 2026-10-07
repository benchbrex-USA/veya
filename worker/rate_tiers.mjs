/**
 * Per-project rate-limit tiers for veya-production.
 *
 * A project's tier is projects.rate_tier in D1 (migration 0002; NULL = default).
 * Each tier maps to a pair of Workers rate-limit bindings (wrangler.production.jsonc)
 * and a counting key:
 *
 *   default  VEYA_RATE_INFERENCE 20/min, VEYA_RATE_HEAVY 6/min, counted per API key (unchanged)
 *   site     VEYA_RATE_SITE 150/min, VEYA_RATE_SITE_HEAVY 45/min, counted per project
 *
 * "site" is sized to replica capacity: 2 x standard-2 replicas serve ~2.6 rps
 * (~156/min) before the gateway sheds with 503 model_busy, so the tier lets a
 * public front end use the whole pool while per-visitor limits stay at that
 * front end (the `veya` edge keeps its 12 units/min per IP). Workers rate limits
 * are per Cloudflare location and approximate; model_busy remains the hard cap.
 *
 * Unknown tiers fall back to default. If a tier's binding is missing, the default
 * binding is used, keyed per API key, so a misconfiguration is never more lenient.
 */

export const DEFAULT_TIER = "default";

export const RATE_TIERS = Object.freeze({
  default: Object.freeze({ inference: "VEYA_RATE_INFERENCE", heavy: "VEYA_RATE_HEAVY", keyBy: "key" }),
  site: Object.freeze({ inference: "VEYA_RATE_SITE", heavy: "VEYA_RATE_SITE_HEAVY", keyBy: "project" }),
});

export function tierFor(name) {
  return typeof name === "string" && Object.hasOwn(RATE_TIERS, name) ? name : DEFAULT_TIER;
}

function usable(binding) {
  return Boolean(binding) && typeof binding.limit === "function";
}

/** Which binding and counting key apply to this identity and request weight. */
export function ratePlan(env, identity, heavy) {
  const tier = tierFor(identity && identity.rateTier);
  const spec = RATE_TIERS[tier];
  const binding = env && env[heavy ? spec.heavy : spec.inference];
  if (usable(binding)) {
    const key = spec.keyBy === "project" ? `project:${identity.projectId}` : identity.keyId;
    return { tier, binding, key };
  }
  const fallback = env && env[heavy ? RATE_TIERS.default.heavy : RATE_TIERS.default.inference];
  return { tier: DEFAULT_TIER, binding: usable(fallback) ? fallback : null, key: identity && identity.keyId };
}
