/**
 * Private R2 bridge for the container. This is not a public Worker route.
 * The container may GET or HEAD one immutable release object. The Worker
 * streams the R2 body and does not hold an R2 access key.
 */

export const ARTIFACT_HOST = "veya-r2.internal";

const FILES = new Set([
  "model.safetensors",
  "manifest.json",
  "model.safetensors.sha256",
]);

const RELEASE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

export function approvedArtifactKey(url, releaseId) {
  if (typeof releaseId !== "string" || !RELEASE_ID.test(releaseId)) return null;
  if (url.hostname !== ARTIFACT_HOST) return null;
  const raw = url.pathname;
  if (raw.includes("%") || raw.includes("\\") || raw.includes("\0")) return null;
  const parts = raw.split("/");
  if (parts[0] !== "") return null;
  const segments = parts.slice(1);
  if (segments.some((part) => part === "" || part === "." || part === "..")) return null;
  if (segments.length !== 3) return null;
  if (segments[0] !== "releases" || segments[1] !== releaseId || !FILES.has(segments[2])) {
    return null;
  }
  return segments.join("/");
}

export async function handleArtifact(request, env) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return artifactError(405, "method_not_allowed");
  }
  let url;
  try {
    url = new URL(request.url);
  } catch {
    return artifactError(403, "forbidden");
  }
  const key = approvedArtifactKey(url, env && env.VEYA_RELEASE_ID);
  if (!key) return artifactError(403, "forbidden");
  const bucket = env && env.VEYA_ARTIFACTS;
  if (!bucket || typeof bucket.get !== "function") {
    return artifactError(503, "artifact_unavailable");
  }
  let object;
  try {
    object = await bucket.get(key);
  } catch {
    return artifactError(502, "artifact_unavailable");
  }
  if (!object) return artifactError(404, "not_found");
  const headers = new Headers();
  headers.set("content-type", "application/octet-stream");
  headers.set("cache-control", "private, no-store");
  if (object.size != null) headers.set("content-length", String(object.size));
  if (request.method === "HEAD") return new Response(null, { status: 200, headers });
  return new Response(object.body, { status: 200, headers });
}

function artifactError(status, code) {
  return new Response(JSON.stringify({ code }), {
    status,
    headers: { "content-type": "application/json", "cache-control": "private, no-store" },
  });
}
