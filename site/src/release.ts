/** Facts copied from the audited manifest and release verdict. Not a live model load. */

export const RELEASE = {
  id: "0.3.5-audit-baseline.1",
  architecture: "veya-shortlist-k8-packed-rerank-v1",
  scorer: "base_decision_head",
  languages: ["en"],
  automation: "DISABLED",
  automationReason:
    "The validation-selected nominal 5% threshold observed 6.99% risk on development-exposed reporting data. No production risk guarantee is supported.",
  choice: {
    status: "RESEARCH RELEASE",
    packedMax: 19,
    shortlistAt: 20,
    k: 8,
    note: "English Choice loads and serves. Blind end-to-end accuracy is not release-candidate grade.",
  },
  noul: {
    status: "RESEARCH ONLY",
    note: "The endpoint exists. This release has no independent audited Noul calibration artifact.",
  },
  score: {
    status: "RESEARCH ONLY",
    note: "Categorical Score is served as research. Ordinal Score is not served.",
  },
  notServed: ["systemone", "ordinal score", "multilingual primitives", "phase 9"],
  verdict: "Veya is not currently a release candidate. English Choice is a research release.",
} as const;

export const BENCHMARK = {
  dataset: "CLINC150",
  split: "val",
  examples: 3000,
  candidates: 150,
  status: "SEALED_EVALUATED",
  release: RELEASE.id,
  recallAt8: 0.356,
  conditionalAccuracy: 0.9803370786516854,
  endToEnd: 0.349,
  ece: 0.26731340633441336,
  caveat:
    "One sealed validation pass. High-cardinality retrieval, not reranking, is the blind bottleneck. These figures are not a production certification.",
} as const;

export const MAX_FUSION = {
  status: "RESEARCH",
  promoted: false,
  recommendation: "RESEARCH",
} as const;

/** Same origin as this Worker. A separate api hostname is not deployed. */
export const API_ORIGIN = "";

export function statusIsGenerallyAvailable(status: string): boolean {
  return status === "GENERALLY AVAILABLE";
}
