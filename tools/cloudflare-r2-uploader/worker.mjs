/**
 * Temporary authenticated uploader for one immutable checkpoint key.
 * It is not a general R2 proxy and it does not read an object key from the request.
 */

export const OBJECT_KEY = "releases/0.3.5-audit-baseline.1/model.safetensors";

const MAX_PART = 10000;

export function authorized(request, token) {
  if (typeof token !== "string" || token.length < 32) return false;
  const header = request.headers.get("authorization") || "";
  const prefix = "Bearer ";
  if (!header.startsWith(prefix)) return false;
  const presented = header.slice(prefix.length);
  if (presented.length !== token.length) return false;
  let diff = 0;
  for (let index = 0; index < presented.length; index += 1) {
    diff |= presented.charCodeAt(index) ^ token.charCodeAt(index);
  }
  return diff === 0;
}

export function partNumberFrom(value) {
  if (!/^[1-9]\d{0,4}$/.test(value || "")) return null;
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1 || number > MAX_PART) return null;
  return number;
}

export function uploadIdFrom(value) {
  if (typeof value !== "string" || value.length < 8 || value.length > 2048) return null;
  if (value.includes("..") || value.includes("%") || value.includes("\\") || value.includes("\0")) return null;
  if (/[\u0000-\u001f\u007f]/.test(value)) return null;
  return value;
}

export function partsFrom(body) {
  if (!Array.isArray(body) || body.length < 1 || body.length > MAX_PART) return null;
  const seen = new Set();
  const parts = [];
  for (const item of body) {
    if (!item || typeof item !== "object") return null;
    const partNumber = partNumberFrom(String(item.partNumber));
    const etag = item.etag;
    if (!partNumber || typeof etag !== "string" || etag.length < 1 || etag.length > 256) return null;
    if (etag.includes("\n") || etag.includes("\r")) return null;
    if (seen.has(partNumber)) return null;
    seen.add(partNumber);
    parts.push({ partNumber, etag });
  }
  parts.sort((left, right) => left.partNumber - right.partNumber);
  return parts;
}

function json(body, status, extra) {
  const headers = new Headers(extra);
  headers.set("content-type", "application/json; charset=utf-8");
  headers.set("cache-control", "no-store");
  return new Response(JSON.stringify(body), { status, headers });
}

function denied() {
  return json({ code: "unauthorized" }, 401);
}

export default {
  async fetch(request, env) {
    if (!authorized(request, env && env.UPLOAD_TOKEN)) return denied();
    const url = new URL(request.url);
    if (url.pathname === "/mpu/create" && request.method === "POST") return create(env);
    if (url.pathname === "/mpu/part" && request.method === "PUT") return part(request, env, url);
    if (url.pathname === "/mpu/complete" && request.method === "POST") return complete(request, env, url);
    if (url.pathname === "/mpu/abort" && request.method === "DELETE") return abort(env, url);
    if (url.pathname === "/verify-object" && request.method === "GET") return verify(env);
    return json({ code: "not_found" }, 404);
  },
};

async function create(env) {
  let upload;
  try {
    upload = await env.VEYA_ARTIFACTS.createMultipartUpload(OBJECT_KEY, {
      customMetadata: {
        release: "0.3.5-audit-baseline.1",
        sha256: "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        "artifact-type": "model-checkpoint",
        "source-commit": "c96d5f78dd0c2cb5f6661827c8e1f890459b451d",
      },
    });
  } catch {
    upload = await env.VEYA_ARTIFACTS.createMultipartUpload(OBJECT_KEY);
  }
  return json({ uploadId: upload.uploadId }, 201);
}

async function part(request, env, url) {
  const uploadId = uploadIdFrom(url.searchParams.get("uploadId"));
  const partNumber = partNumberFrom(url.searchParams.get("partNumber"));
  if (!uploadId || !partNumber) return json({ code: "invalid_input" }, 400);
  const length = request.headers.get("content-length");
  if (length && Number(length) > 20 * 1024 * 1024) return json({ code: "request_too_large" }, 413);
  const upload = env.VEYA_ARTIFACTS.resumeMultipartUpload(OBJECT_KEY, uploadId);
  const uploaded = await upload.uploadPart(partNumber, request.body);
  return json({ partNumber: uploaded.partNumber, etag: uploaded.etag }, 200);
}

async function complete(request, env, url) {
  const uploadId = uploadIdFrom(url.searchParams.get("uploadId"));
  const parts = partsFrom(await request.json().catch(() => null));
  if (!uploadId || !parts) return json({ code: "invalid_input" }, 400);
  const upload = env.VEYA_ARTIFACTS.resumeMultipartUpload(OBJECT_KEY, uploadId);
  const object = await upload.complete(parts);
  return json({ key: OBJECT_KEY, size: object.size ?? null }, 200);
}

async function abort(env, url) {
  const uploadId = uploadIdFrom(url.searchParams.get("uploadId"));
  if (!uploadId) return json({ code: "invalid_input" }, 400);
  const upload = env.VEYA_ARTIFACTS.resumeMultipartUpload(OBJECT_KEY, uploadId);
  await upload.abort();
  return json({ aborted: true }, 200);
}

async function verify(env) {
  const object = await env.VEYA_ARTIFACTS.get(OBJECT_KEY);
  if (!object) return json({ code: "not_found" }, 404);
  const headers = new Headers({ "cache-control": "no-store" });
  if (object.size != null) {
    headers.set("content-length", String(object.size));
    headers.set("x-r2-size", String(object.size));
  }
  if (object.httpEtag) headers.set("etag", object.httpEtag);
  return new Response(object.body, { status: 200, headers });
}
