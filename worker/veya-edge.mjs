/**
 * Veya edge proxy (Cloudflare Workers).
 *
 * The audited decision model is a PyTorch checkpoint served by
 * web/model_server.py on 127.0.0.1:8080, fronted by the stdlib gateway in
 * web/server.py. Workers cannot execute PyTorch, so this Worker's only job is
 * to give the stack a stable public hostname and forward every request to the
 * live gateway origin (VEYA_ORIGIN, set in wrangler.jsonc / dashboard vars).
 *
 * Security notes:
 *  - The browser's Origin header is forwarded untouched, so the gateway's
 *    same-origin policy (VEYA_ALLOWED_ORIGINS) still governs model requests.
 *  - The model service itself remains loopback-only; only the gateway is
 *    ever reachable from the edge.
 */

const EDGE_HEALTH_PATH = "/__edge_health";

function originOf(env) {
  const raw = (env && env.VEYA_ORIGIN) || "";
  return raw.replace(/\/+$/, "");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = originOf(env);

    if (url.pathname === EDGE_HEALTH_PATH) {
      return new Response(
        JSON.stringify({ edge: "ok", origin_configured: Boolean(origin) }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      );
    }

    if (!origin) {
      return new Response(
        JSON.stringify({
          code: "origin_not_configured",
          detail: "VEYA_ORIGIN is not set for this Worker.",
        }),
        { status: 503, headers: { "content-type": "application/json" } },
      );
    }

    const target = new URL(url.pathname + url.search, origin);
    const headers = new Headers(request.headers);
    headers.set("Host", target.host);
    // The gateway speaks plain identity; avoid double-encoding through the edge.
    headers.set("Accept-Encoding", "identity");
    // Hop-by-hop headers must not be forwarded.
    headers.delete("Connection");
    headers.delete("Proxy-Authenticate");
    headers.delete("Proxy-Authorization");
    headers.delete("TE");
    headers.delete("Trailer");
    headers.delete("Transfer-Encoding");
    headers.delete("Upgrade");

    const hasBody = request.method !== "GET" && request.method !== "HEAD";

    try {
      const upstream = await fetch(target.toString(), {
        method: request.method,
        headers,
        body: hasBody ? request.body : undefined,
        redirect: "manual",
      });

      const responseHeaders = new Headers(upstream.headers);
      // The edge re-chunks the body; a stale Content-Length would corrupt it.
      responseHeaders.delete("Content-Length");
      responseHeaders.delete("Transfer-Encoding");
      responseHeaders.delete("Connection");

      return new Response(upstream.body, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers: responseHeaders,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({
          code: "origin_unreachable",
          detail: "The Veya gateway origin did not respond.",
          request_id: request.headers.get("x-request-id") || null,
        }),
        { status: 502, headers: { "content-type": "application/json" } },
      );
    }
  },
};
