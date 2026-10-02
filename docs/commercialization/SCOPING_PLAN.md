# Veya Commercial Infrastructure Scoping Plan

> **Status**: DRAFT — derived from `docs/commercialization/final_report.md` gaps and the workload contract in `docs/commercialization/workload_contract.md`.

---

## 1. Product Scope (What the API Must Do)

| Capability | Current State | Target |
|---|---|---|
| **Typed decision API** | Research SDK (`veya.load`, `router.predict`) | Versioned REST + gRPC endpoints for `choice` / `score` / `noul` |
| **Model routing** | Built-in `Router` class | API-level `/v1/route` with explainable routing metadata |
| **Selective deferral** | Research `SelectiveExtra` | Policy-as-code: per-org confidence thresholds, audit log |
| **Customer model pinning** | None | `POST /v1/models/pin` — freeze a checkpoint hash per project |
| **Idempotency** | None | `Idempotency-Key` header on all mutating endpoints |
| **Usage metering** | None | Per-decision, per-project, per-model aggregation |

---

## 2. SDKs

| Language | Priority | Deliverables |
|---|---|---|
| **Python** | P0 | `pip install veya-cloud` — async/httpx client, Pydantic models, retry/backoff, context-manager sessions |
| **TypeScript/JS** | P1 | `npm i @veyadynamics/sdk` — typed fetch wrapper, zod schemas, React hooks for streaming |
| **Go** | P2 | Minimal client for backend services |
| **Rust** | P3 | If demand |

**Common contract**: OpenAPI 3.1 spec → codegen for all three.

---

## 3. Metering & Billing

| Component | Design |
|---|---|
| **Event stream** | Kafka topic `veyas.decisions` — schema: `project_id, model, primitive, options, latency_ms, tokens, success, timestamp` |
| **Aggregation** | Flink/Spark streaming → hourly rollups in Postgres / ClickHouse |
| **Pricing model** | Tiered: free tier (10k decisions/mo), pay-as-you-go ($0.00015/decision), enterprise (committed volume discount) |
| **Invoicing** | Monthly PDF + CSV; Stripe for self-serve, manual for enterprise |
| **Spend controls** | Per-project hard caps, alerting at 80%/100% |

---

## 4. Org / Project Primitives

```
Organization
  └── Project (billing scope, model pins, rate limits)
       ├── API Key (scoped to project, rotatable, prefix `vya_`)
       ├── Webhook endpoints (decision callbacks, quota alerts)
       └── Members (roles: Owner, Admin, Developer, Viewer)
```

API surface:
- `POST /v1/orgs` — create org
- `POST /v1/projects` — create project under org
- `GET /v1/projects/{id}/usage` — current period usage
- `POST /v1/projects/{id}/pins` — pin model checkpoint
- `POST /v1/apikeys` — issue/rotate keys

---

## 5. Auth & Access

| Layer | Mechanism |
|---|---|
| **API keys** | HMAC-SHA256 signed, prefix `vya_live_` / `vya_test_` |
| **SSO** | OIDC (Okta, Entra ID, Google Workspace) — SAML 2.0 for enterprise |
| **RBAC** | Org-level roles; project-level ACLs; resource-scoped tokens |
| **Audit log** | Immutable append-only (CloudTrail-style) — decisions, key rotations, policy changes, model pin events |
| **Secrets** | Vault-backed; rotation API for customer-managed keys |

---

## 6. Tenancy & Isolation

| Requirement | Approach |
|---|---|
| **Data isolation** | Per-project DB schemas; encryption-at-rest per project key |
| **Network isolation** | PrivateLink / VPC peering for enterprise; default: TLS over public internet |
| **Model isolation** | Customer-pinned checkpoints never evicted; dedicated GPU pool option |
| **Compliance** | SOC 2 Type II target; GDPR DPA; HIPAA BAA (future) |

---

## 7. Observability (Customer-Facing)

- **Latency percentiles** (p50/p95/p99) per model per project
- **Error rates** by primitive, by model
- **Calibration drift** alerts (ECE shift > 5%)
- **Quota utilization** real-time
- **Dashboards**: Grafana embed / native React components

---

## 8. Implementation Phases

| Phase | Duration | Deliverable |
|---|---|---|
| **0: Spec freeze** | 1 wk | OpenAPI 3.1, DB schema, event contracts signed off |
| **1: Core API** | 3 wks | FastAPI service (Python), Postgres, Redis, `/v1/choice`, `/v1/score`, `/v1/noul`, `/v1/route`, API key auth, idempotency |
| **2: Metering pipeline** | 2 wks | Kafka → Flink → ClickHouse, hourly rollups, usage endpoint |
| **3: Org/Project mgmt** | 2 wks | CRUD APIs, RBAC, API key rotation, webhooks |
| **4: Python SDK** | 1 wk | Generated from OpenAPI, tests, PyPI publish |
| **5: TypeScript SDK** | 1 wk | Generated, npm publish, docs |
| **6: SSO + Audit** | 2 wks | OIDC integration, immutable audit log, admin console |
| **7: Private networking** | 2 wks | PrivateLink, VPC peering, dedicated GPU pool provisioning |
| **8: Load test + GA** | 2 wks | 10k RPS sustained, chaos tests, runbooks, status page |

**Total estimate**: ~16 weeks to GA with 2 engineers.

---

## 9. Infrastructure Choices

| Layer | Recommendation | Rationale |
|---|---|---|
| **API runtime** | Fly.io / Railway / Cloud Run | Fast deploys, GPU support, scale-to-zero |
| **DB** | Postgres (managed) + ClickHouse (analytics) | ACID for billing, OLAP for usage |
| **Message bus** | Kafka (managed: Confluent Cloud / Redpanda) | Replay, ordering, exactly-once |
| **Secrets** | HashiCorp Vault (managed) or cloud KMS | Rotation, audit, dynamic credentials |
| **Observability** | Grafana Cloud + Tempo + Loki | Unified logs/metrics/traces |
| **CI/CD** | GitHub Actions → ArgoCD / Flux | GitOps, progressive delivery |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Model quality insufficient for paid tier** | High | Blocks monetization | Gate GA behind 98% scorer; keep free tier as research sandbox |
| **GPU cost > revenue at low volume** | Medium | Margin compression | Tiered pricing; reserved instances for enterprise; CPU fallback for `noul` |
| **Licensing (SGD CC-BY-SA) blocks commercial retriever** | High | Retriever cannot ship commercially | Train replacement retriever on permissive data; ship base retriever only |
| **Competitor benchmark unavailable** | Medium | No credible positioning | Run internal Jev-equivalent benchmark; publish methodology |
| **Customer demand unvalidated** | High | Building unused features | Phase 1: 2 design partners, shadow mode, revenue letters of intent |

---

## 11. Decision Points (Need Stakeholder Input)

1. **Pricing**: Per-decision vs. per-token vs. seat-based? (Current assumption: per-decision)
2. **Self-serve vs. sales-led**: Launch with self-serve Stripe checkout or require sales contact?
3. **Data residency**: EU / US regions from day 1 or phased?
4. **SLA target**: 99.9% availability (3 nines) or 99.95%? Impacts infra cost.

---

## 12. Next Actions

1. [ ] Review and approve this scoping plan
2. [ ] Freeze OpenAPI spec (derive from `veya/agent.py` + `router.py` endpoints)
3. [ ] Provision dev infra (Postgres, Kafka, Redis) in staging
4. [ ] Implement Phase 1 core API skeleton
5. [ ] Recruit 2 design partners for shadow evaluation

---

*Generated from the commercialization gate report gaps. See `docs/commercialization/final_report.md` for the evidence behind each gap.*