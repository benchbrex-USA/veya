/**
 * Console and site Worker. The browser never receives VEYA_EDGE_TOKEN.
 * Customer API keys are hashed before storage. Model weights are not served.
 */

import { buildInferenceRequest } from "../src/contract.ts";
import {
  cacheControl,
  generateApiKey,
  keyPrefix,
  latencyBucket,
  securityHeaders,
  sha256Hex,
  surfaceForHost,
  verifyTurnstile,
} from "../src/security.ts";

interface Sql {
  prepare(query: string): {
    bind(...values: unknown[]): {
      run(): Promise<unknown>;
      all<T>(): Promise<{ results: T[] }>;
    };
  };
}

interface UsagePoint {
  indexes?: string[];
  blobs?: string[];
  doubles?: number[];
}

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> };
  DB?: Sql;
  VEYA_API?: { fetch(input: Request): Promise<Response> };
  USAGE?: { writeDataPoint(point: UsagePoint): void };
  VEYA_CONSOLE_RATE?: { limit(input: { key: string }): Promise<{ success: boolean }> };
  VEYA_EDGE_TOKEN?: string;
  SESSION_SECRET?: string;
  TURNSTILE_SECRET?: string;
  VEYA_ENVIRONMENT?: string;
  VEYA_HSTS?: string;
}

function json(body: unknown, status = 200, extra?: HeadersInit): Response {
  const headers = new Headers(extra);
  headers.set("content-type", "application/json; charset=utf-8");
  headers.set("cache-control", "no-store");
  return new Response(JSON.stringify(body), { status, headers });
}

async function sessionOk(request: Request, env: Env): Promise<boolean> {
  const cookie = request.headers.get("cookie") || "";
  const match = cookie.match(/(?:^|;\s*)veya_session=([^;]+)/);
  if (!match || !env.SESSION_SECRET) return false;
  const [body, mac] = decodeURIComponent(match[1]).split(".");
  if (!body || !mac) return false;
  const expected = await sha256Hex(`${env.SESSION_SECRET}.${body}`);
  return expected === mac;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const headers = securityHeaders(env.VEYA_HSTS === "true");
    if (url.pathname.startsWith("/api/")) return api(request, env, url);
    if (surfaceForHost(url.host) === "api") {
      return json({ code: "not_found", detail: "This host does not serve the model." }, 404);
    }
    const asset = await env.ASSETS.fetch(request);
    const response = new Response(asset.body, asset);
    for (const [name, value] of Object.entries(headers)) response.headers.set(name, value);
    response.headers.set("cache-control", cacheControl(url.pathname));
    return response;
  },
};

async function api(request: Request, env: Env, url: URL): Promise<Response> {
  if (env.VEYA_CONSOLE_RATE) {
    const key = request.headers.get("cf-connecting-ip") || "local";
    const limited = await env.VEYA_CONSOLE_RATE.limit({ key });
    if (!limited.success) {
      return json({ code: "rate_limited", detail: "Too many requests." }, 429, { "retry-after": "60" });
    }
  }
  if (url.pathname === "/api/session" && request.method === "POST") return openSession(request, env);
  if (!(await sessionOk(request, env))) {
    return json({ code: "unauthorized", detail: "Authentication is required." }, 401);
  }
  if (url.pathname === "/api/playground" && request.method === "POST") return playground(request, env);
  if (url.pathname === "/api/keys" && request.method === "POST") return createKey(request, env);
  if (url.pathname === "/api/keys" && request.method === "GET") return listKeys(env);
  if (url.pathname === "/api/projects" && request.method === "GET") return listProjects(env);
  if (url.pathname === "/api/projects" && request.method === "POST") return createProject(request, env);
  if (url.pathname === "/api/usage" && request.method === "GET") {
    return json({ requests: [], note: "No usage has been recorded." });
  }
  return json({ code: "not_found", detail: "Not found." }, 404);
}

async function openSession(request: Request, env: Env): Promise<Response> {
  const payload = (await request.json().catch(() => null)) as { token?: string } | null;
  const ip = request.headers.get("cf-connecting-ip") || "";
  const secret = env.TURNSTILE_SECRET || "";
  const passed = await verifyTurnstile(secret, payload?.token || "", ip);
  if (!passed || !env.SESSION_SECRET) {
    return json({ code: "unauthorized", detail: "Identity check failed." }, 401);
  }
  const body = JSON.stringify({ sub: "local", exp: Date.now() + 12 * 60 * 60 * 1000 });
  const mac = await sha256Hex(`${env.SESSION_SECRET}.${body}`);
  const headers = new Headers({ "cache-control": "no-store" });
  headers.append(
    "set-cookie",
    `veya_session=${encodeURIComponent(`${body}.${mac}`)}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=43200`,
  );
  return json({ ok: true }, 200, headers);
}

async function playground(request: Request, env: Env): Promise<Response> {
  const raw = (await request.json().catch(() => null)) as {
    primitive?: "choice" | "noul" | "score";
    state?: string;
    instructions?: string;
    criteria?: { id: string; description: string }[] | string[];
  } | null;
  if (!raw?.primitive || !raw.state || !raw.instructions) {
    return json({ code: "invalid_input", detail: "Incomplete request." }, 400);
  }
  const criteria = raw.criteria ?? [];
  const built = buildInferenceRequest({
    primitive: raw.primitive,
    state: raw.state,
    instructions: raw.instructions,
    criteria,
  });
  if (!built.ok) return json({ code: "invalid_input", detail: built.error }, 400);
  if (!env.VEYA_API) {
    return json(
      { code: "api_unavailable", detail: "The Veya API service binding is not configured." },
      503,
    );
  }
  const headers = new Headers({ "content-type": "application/json" });
  if (env.VEYA_EDGE_TOKEN) headers.set("authorization", `Bearer ${env.VEYA_EDGE_TOKEN}`);
  const started = Date.now();
  const upstream = await env.VEYA_API.fetch(
    new Request("https://api.veyadynamics.com/api/infer", {
      method: "POST",
      headers,
      body: JSON.stringify(built.body),
    }),
  );
  const latency = Date.now() - started;
  const text = await upstream.text();
  env.USAGE?.writeDataPoint({
    indexes: ["console"],
    blobs: [built.body.primitive, String(upstream.status), latencyBucket(latency), "0.3.5-audit-baseline.1"],
    doubles: [latency],
  });
  return new Response(text, {
    status: upstream.status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function createProject(request: Request, env: Env): Promise<Response> {
  if (!env.DB) return json({ code: "unavailable", detail: "D1 is not configured." }, 503);
  const body = (await request.json().catch(() => null)) as { name?: string; organization?: string } | null;
  const name = body?.name?.trim() || "";
  const organization = body?.organization?.trim() || "";
  if (!name || !organization || name.length > 80 || organization.length > 80) {
    return json({ code: "invalid_input", detail: "Organization and project names are required." }, 400);
  }
  const now = new Date().toISOString();
  const orgId = crypto.randomUUID();
  const projectId = crypto.randomUUID();
  await env.DB.prepare("INSERT INTO organizations (id, name, created_at) VALUES (?, ?, ?)").bind(orgId, organization, now).run();
  await env.DB.prepare("INSERT INTO projects (id, organization_id, name, created_at) VALUES (?, ?, ?, ?)").bind(projectId, orgId, name, now).run();
  await env.DB.prepare("INSERT INTO audit_events (id, organization_id, action, created_at) VALUES (?, ?, ?, ?)").bind(crypto.randomUUID(), orgId, "project.create", now).run();
  return json({ id: projectId, organization_id: orgId, name }, 201);
}

async function listProjects(env: Env): Promise<Response> {
  if (!env.DB) return json({ projects: [] });
  const rows = await env.DB.prepare(
    "SELECT projects.id, projects.name, organizations.name AS organization FROM projects JOIN organizations ON organizations.id = projects.organization_id ORDER BY projects.created_at DESC LIMIT 50",
  ).bind().all<{ id: string; name: string; organization: string }>();
  return json({ projects: rows.results });
}

async function createKey(request: Request, env: Env): Promise<Response> {
  if (!env.DB) return json({ code: "unavailable", detail: "D1 is not configured." }, 503);
  const body = (await request.json().catch(() => null)) as { project_id?: string } | null;
  if (!body?.project_id) return json({ code: "invalid_input", detail: "A project is required." }, 400);
  const mode = env.VEYA_ENVIRONMENT === "production" ? "live" : "test";
  const token = generateApiKey(mode);
  const now = new Date().toISOString();
  const id = crypto.randomUUID();
  await env.DB.prepare(
    "INSERT INTO api_keys (id, organization_id, project_id, prefix, key_hash, created_at) SELECT ?, organization_id, id, ?, ?, ? FROM projects WHERE id = ?",
  ).bind(id, keyPrefix(token), await sha256Hex(token), now, body.project_id).run();
  return json({ id, prefix: keyPrefix(token), token, shown_once: true }, 201);
}

async function listKeys(env: Env): Promise<Response> {
  if (!env.DB) return json({ keys: [] });
  const rows = await env.DB.prepare(
    "SELECT id, prefix, created_at, last_used_at, revoked_at FROM api_keys ORDER BY created_at DESC LIMIT 50",
  ).bind().all();
  return json({ keys: rows.results });
}
