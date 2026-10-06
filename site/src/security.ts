/** Browser-safe helpers. No infrastructure credential is accepted or returned. */

const KEY_RE = /^veya_(test|live)_[A-Za-z0-9_-]{43}$/;

export function securityHeaders(hsts: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    "content-security-policy":
      "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "x-frame-options": "DENY",
  };
  if (hsts) headers["strict-transport-security"] = "max-age=31536000; includeSubDomains";
  return headers;
}

export function surfaceForHost(host: string): "marketing" | "console" | "docs" | "api" {
  const name = host.split(":")[0].toLowerCase();
  if (name.startsWith("app.") || name.startsWith("app-staging.")) return "console";
  if (name.startsWith("docs.") || name.startsWith("docs-staging.")) return "docs";
  if (name.startsWith("api.") || name.startsWith("api-staging.")) return "api";
  return "marketing";
}

export function cacheControl(pathname: string): string {
  if (pathname.startsWith("/assets/")) return "public, max-age=31536000, immutable";
  if (pathname.startsWith("/api/")) return "no-store";
  return "no-cache";
}

export async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function timingSafeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

export function generateApiKey(mode: "test" | "live"): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  const body = btoa(String.fromCharCode(...bytes)).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
  return `veya_${mode}_${body}`;
}

export function maskApiKey(key: string): string {
  if (!KEY_RE.test(key) && key.length < 18) return "hidden";
  return `${key.slice(0, 14)}…${key.slice(-4)}`;
}

export function keyPrefix(key: string): string {
  return key.slice(0, 16);
}

export function toggleFaq(open: string | null, id: string): string | null {
  return open === id ? null : id;
}

export async function verifyTurnstile(
  secret: string,
  token: string,
  ip: string,
  fetcher: typeof fetch = fetch,
): Promise<boolean> {
  if (!secret || !token) return false;
  const body = new URLSearchParams({ secret, response: token, remoteip: ip });
  const response = await fetcher("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body,
  });
  if (!response.ok) return false;
  const payload = (await response.json()) as { success?: boolean };
  return payload.success === true;
}

export function latencyBucket(ms: number): string {
  if (ms < 50) return "lt50";
  if (ms < 100) return "lt100";
  if (ms < 250) return "lt250";
  if (ms < 500) return "lt500";
  return "gte500";
}
