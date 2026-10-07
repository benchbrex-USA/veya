// k6 load test for POST /v1/decisions on veya-production.
//
//   VEYA_BASE_URL=https://api.veyadynamics.com VEYA_API_KEY=... k6 run loadtest/k6-decisions.js
//
// Environment:
//   VEYA_BASE_URL   required, e.g. https://veya-production.<subdomain>.workers.dev
//   VEYA_API_KEY    required, a D1 project key with decisions:invoke (never commit it)
//   VEYA_RATE       target requests/second for the constant-arrival scenario (default 0.25)
//   VEYA_DURATION   scenario duration (default 5m)
//   VEYA_VUS        pre-allocated VUs (default 10)
//   VEYA_P95_MS     p95 latency threshold in ms for 2xx decisions (default 1500)
//   VEYA_P99_MS     p99 latency threshold in ms (default 4000)
//   VEYA_MAX_ERROR  max non-2xx/429 error rate (default 0.01)
//   VEYA_HEAVY      "1" to mix in 20% non-choice ("score") requests, which use the
//                   heavy limiter (6/min per key)
//
// The edge enforces 20 inference requests/min per API key (6/min heavy). One key
// at the default 0.25 rps (15/min) stays under it; to measure capacity rather
// than the limiter, run several k6 instances with different keys. 429s are
// counted separately and do not fail the error-rate threshold.

import http from "k6/http";
import { check } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = (__ENV.VEYA_BASE_URL || "").replace(/\/+$/, "");
const API_KEY = __ENV.VEYA_API_KEY || "";
if (!BASE_URL || !API_KEY) {
  throw new Error("VEYA_BASE_URL and VEYA_API_KEY must be set");
}

const RATE = Number(__ENV.VEYA_RATE || 0.25);
const DURATION = __ENV.VEYA_DURATION || "5m";
const VUS = Number(__ENV.VEYA_VUS || 10);
const P95 = Number(__ENV.VEYA_P95_MS || 1500);
const P99 = Number(__ENV.VEYA_P99_MS || 4000);
const MAX_ERROR = Number(__ENV.VEYA_MAX_ERROR || 0.01);
const HEAVY = __ENV.VEYA_HEAVY === "1";

const decisionErrors = new Rate("veya_decision_errors");
const rateLimited = new Counter("veya_rate_limited");
const timeouts = new Counter("veya_upstream_timeouts");
const decisionLatency = new Trend("veya_decision_latency", true);

// 429 is the documented per-key limiter, not a failure of the service.
http.setResponseCallback(http.expectedStatuses({ min: 200, max: 299 }, 429));

// k6 only accepts integer rates; express fractional rps as N per timeUnit.
const perMinute = Math.max(1, Math.round(RATE * 60));

export const options = {
  scenarios: {
    decisions: {
      executor: "constant-arrival-rate",
      rate: perMinute,
      timeUnit: "1m",
      duration: DURATION,
      preAllocatedVUs: VUS,
      maxVUs: VUS * 4,
    },
  },
  thresholds: {
    "http_req_duration{expected_response:true}": [`p(95)<${P95}`],
    http_req_failed: [`rate<${MAX_ERROR}`],
    veya_decision_latency: [`p(95)<${P95}`, `p(99)<${P99}`],
    veya_decision_errors: [`rate<${MAX_ERROR}`],
    checks: ["rate>0.99"],
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

// Same contract as web/contracts.py validate(): primitive, state, instructions, criteria.
const SMALL = {
  primitive: "choice",
  state: "I was charged twice for my subscription this month.",
  instructions: "Which team should handle this ticket?",
  criteria: {
    billing: "Payments, invoices, refunds and charges",
    technical: "Bugs, outages and errors in the product",
    account: "Login, password and profile changes",
    sales: "Pricing, plans and upgrades",
  },
};

const HEAVY_PAYLOAD = JSON.stringify({
  primitive: "score",
  state: "The checkout page has been down for two hours and customers cannot pay.",
  instructions: "How urgent is this incident?",
  criteria: ["low", "medium", "high", "critical"],
});
const SMALL_PAYLOAD = JSON.stringify(SMALL);

export default function () {
  const heavy = HEAVY && Math.random() < 0.2;
  const response = http.post(`${BASE_URL}/v1/decisions`, heavy ? HEAVY_PAYLOAD : SMALL_PAYLOAD, {
    headers: {
      authorization: `Bearer ${API_KEY}`,
      "content-type": "application/json",
    },
    timeout: "30s",
    tags: { kind: heavy ? "heavy" : "small" },
  });
  if (response.status === 429) {
    rateLimited.add(1);
    return;
  }
  if (response.status === 504) timeouts.add(1);
  const ok = response.status >= 200 && response.status < 300;
  decisionErrors.add(!ok);
  if (ok) decisionLatency.add(response.timings.duration);
  check(response, {
    "status is 2xx or 429": (r) => (r.status >= 200 && r.status < 300) || r.status === 429,
    "hsts present": (r) => Boolean(r.headers["Strict-Transport-Security"]),
  });
}
