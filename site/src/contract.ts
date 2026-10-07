/** Mirrors the public gateway bounds in web/contracts.py. Does not call the model. */

export type ServedPrimitive = "choice" | "noul" | "score";

export interface InferenceBody {
  primitive: ServedPrimitive;
  state: string;
  instructions: string;
  criteria?: Record<string, string> | string[];
}

export type Built =
  | { ok: true; body: InferenceBody }
  | { ok: false; error: string };

const MAX_CHOICE = 100;
const MAX_STATE = 12_000;
const MAX_INSTRUCTIONS = 500;

function normal(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

export function buildInferenceRequest(input: {
  primitive: ServedPrimitive;
  state: string;
  instructions: string;
  criteria: { id: string; description: string }[] | string[];
}): Built {
  const state = input.state.trim();
  if (!state) return { ok: false, error: "State must be nonempty text." };
  if (new TextEncoder().encode(state).length > MAX_STATE) {
    return { ok: false, error: "State exceeds 12 KB." };
  }
  const instructions = input.instructions.trim();
  if (!instructions || instructions.length > MAX_INSTRUCTIONS) {
    return { ok: false, error: "Question must be nonempty text of at most 500 characters." };
  }
  if (input.primitive === "noul") {
    return { ok: true, body: { primitive: "noul", state, instructions } };
  }
  if (input.primitive === "score") {
    const labels = (input.criteria as string[]).map((label) => label.trim()).filter(Boolean);
    if (labels.length < 2 || labels.length > 10) {
      return { ok: false, error: "Score requires 2–10 ordered labels." };
    }
    if (new Set(labels.map(normal)).size !== labels.length) {
      return { ok: false, error: "Score labels must be unique." };
    }
    return { ok: true, body: { primitive: "score", state, instructions, criteria: labels } };
  }
  const rows = input.criteria as { id: string; description: string }[];
  const criteria: Record<string, string> = {};
  const seen = new Set<string>();
  for (const row of rows) {
    const id = row.id.trim();
    const description = row.description.trim();
    if (!id || id.length > 80) return { ok: false, error: "Candidate ID must be 1–80 characters." };
    if (!description || description.length > 240) {
      return { ok: false, error: "Candidate descriptions must be text of at most 240 characters." };
    }
    const key = normal(id);
    if (seen.has(key)) return { ok: false, error: "Candidate IDs must be unique." };
    seen.add(key);
    criteria[id] = description;
  }
  const count = Object.keys(criteria).length;
  if (count < 2 || count > MAX_CHOICE) {
    return { ok: false, error: `Choice requires 2–${MAX_CHOICE} candidates.` };
  }
  return { ok: true, body: { primitive: "choice", state, instructions, criteria } };
}

export function choiceRoute(candidateCount: number): "packed" | "shortlist_k8" {
  return candidateCount >= 20 ? "shortlist_k8" : "packed";
}
