/** D1 API-key checks. The raw key is never stored or logged. */

export async function sha256Hex(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function hashesEqual(left, right) {
  if (typeof left !== "string" || typeof right !== "string" || left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

export function bearer(header) {
  if (typeof header !== "string" || !header.startsWith("Bearer ")) return "";
  return header.slice("Bearer ".length).trim();
}

export async function authenticate(db, presented, now = new Date(), scope = "decisions:invoke") {
  const candidate = await sha256Hex(typeof presented === "string" ? presented : "");
  const row = db
    ? await db.prepare(
      "SELECT id, organization_id, project_id, key_hash, scopes, expires_at, revoked_at FROM api_keys WHERE key_hash = ?",
    ).bind(candidate).first()
    : null;
  const stored = row && typeof row.key_hash === "string" ? row.key_hash : candidate.replace(/./g, "0");
  let accepted = hashesEqual(candidate, stored) && Boolean(row);
  if (!row || row.revoked_at) accepted = false;
  if (row && row.expires_at && Date.parse(row.expires_at) <= now.getTime()) accepted = false;
  let scopes = [];
  try {
    scopes = row && row.scopes ? JSON.parse(row.scopes) : [];
  } catch {
    scopes = [];
  }
  if (!Array.isArray(scopes) || !scopes.includes(scope)) accepted = false;
  if (!accepted) return null;
  return {
    keyId: row.id,
    organizationId: row.organization_id,
    projectId: row.project_id,
    scopes,
  };
}
