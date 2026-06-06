# Comparative Analysis: agent-utilities vs. Palantir AIP

> Robust capability-by-capability comparison of **agent-utilities** against
> **Palantir's Artificial Intelligence Platform (AIP)** "End-to-End Agentic
> Architecture" (12 capabilities) and the general Agentic AI Reference
> Architecture (9 layers). Grounded in a code audit of the live codebase, June 2026.
>
> **Bottom line:** agent-utilities already implements **~10/12** Palantir
> capabilities in full and **architecturally exceeds** AIP on four axes that fall
> out *for free* from its unique substrate (formal OWL+SHACL ontology,
> graph-native Rust compute, closed-loop self-evolution, reward-weighted routing).
> Two genuine, well-scoped gaps were closed in this cycle — the **Ontology Action
> System** (`CONCEPT:KG-2.25`) and the **declarative Resilience Policy**
> (`CONCEPT:ORCH-1.36`); the remainder are a documented backlog.

---

## 1. Why compare to Palantir at all?

Palantir's pitch is that the **Ontology** — a governed, executable model unifying
*data + logic + actions + security* — is the moat that stops enterprise agents
from hallucinating and lets them act on real operational state under the same
governance as humans. That is precisely the thesis agent-utilities is built on
(the Epistemic Knowledge Graph, Pillar 2). So the comparison is apples-to-apples,
and the interesting questions are: *where do we already match it, where do we
genuinely lag, and what do we get that a closed commercial platform structurally
cannot?*

## 2. Palantir AIP 12-Capability Mapping

| # | Palantir AIP capability | agent-utilities implementation | Status |
|---|---|---|---|
| 1 | Secure LLM Integration & Access | `core/model_factory.py:create_model` (multi-provider, BYOM); `security/permissions_kernel.py` (HMAC identity, roles); `security/guardrails.py:PolicyEngine` (moderation) | **FULL** |
| 2 | End-to-End Observability | `observability/token_tracker.py` (4-bucket token accounting), `observability/audit_logger.py` (append-only audit), OpenTelemetry tracer in `orchestration/engine.py`, Langfuse widget | **FULL** |
| 3 | Context Engineering | `knowledge_graph/core/context_builder.py`; CDC via `kafka_graph_sync.py` / `nats_queue_backend.py`; 15-phase enrichment pipeline | **FULL** |
| 4 | **The Ontology System** | `knowledge_graph/core/owl_bridge.py` (promote→reason→downfeed), `ontology.ttl` + 19 domain modules, `core/shacl_validator.py` (governance gate), `core/ogm.py`. **Nouns** unified; **verbs** were property-only → now first-class via `CONCEPT:KG-2.25` | **FULL+ (gap closed)** |
| 5 | Vector, Compute, Tool Services | epistemic-graph Rust engine (`core/graph_compute.py`), HNSW `retrieval/capability_index.py`, vendor-neutral `EmbeddingFactory`, ontology-driven tool swap (`providesCapability`/`swappableWith`) | **FULL** |
| 6 | Security & Governance | `security/permissions_kernel.py` (role/capability), `tool_guard.py`, SHACL gate, `isolation` zero-trust, Eunomia policies, marking via `ActorContext.tenant_id` | **FULL** |
| 7 | Agent Lifecycle (build→orchestrate→evaluate→iterate) | `harness/` (component registry, evaluation_engine, agentic_evolution_engine + GEPA), `orchestration/durable_execution.py`, golden-loop | **FULL** |
| 8 | Operational Automation | KG/NATS/Kafka event-sourced automations, `research/golden_loop.py`, durable workflows | **FULL** |
| 9 | Development Environments | `mcp/kg_server.py` (MCP first-class), CLI, Python SDK, notebook live-artifacts | **FULL** |
| 10 | Human + AI Applications | `core/company_brain.py` (shared KG, per-actor filtering), topological analytics, multi-tenant `ActorContext` | **FULL** |
| 11 | Package, Release, Deploy | pip package, CI guardrails, Docker compose templates — **no formal release-channel system** | **PARTIAL** |
| 12 | Enterprise Automation | agents under identity governance build pipelines/logic; ServiceNow/ERPNext/etc. extractors; `blast_radius.py` impact analysis | **FULL** |

## 3. General 9-Layer Reference Architecture (condensed)

| Layer | Status | Anchor |
|---|---|---|
| User/Client | FULL | MCP / HTTP / CLI / notebooks |
| Orchestration / Control Plane | FULL | `orchestration/engine.py`, `graph/routing/`, `graph/planning/` |
| Agent Layer (roles) | FULL | `core/registry/`, `team_composer`, capability protocols |
| Tools & Integrations | FULL | MCP, ontology-driven tool swap, enterprise extractors |
| Memory & Knowledge | FULL | short-term `memory_engine`, vector HNSW, KB, episodic, profile/RLM |
| Monitoring & Observability | FULL | token tracker, audit, OTel |
| **Reliability & Failure Mgmt** | **PARTIAL → FULL** | circuit breaker + durable checkpoints existed; **declarative retry/backoff/fallback added** (`CONCEPT:ORCH-1.36`) |
| Governance & Security | FULL | permissions kernel, SHACL gate, zero-trust |
| Foundation / Infra | FULL | model factory, pluggable backends, Kafka/NATS, secrets |

## 4. Hidden value-adds — what we get *for free* that AIP structurally cannot

These are not "catch-up" items; they are emergent advantages of the substrate.

1. **Formal OWL2 + SHACL ontology, not best-practice typing.** Palantir's
   "ontology" is a governed schema; ours is an *executable logic*. Every fact is
   either declared or a derived OWL consequence (`owl_bridge.py` HermiT/owlready2),
   and the SHACL gate (`governance.shapes.ttl`) *quarantines* invalid nodes before
   they persist. **For free:** the new Action System's permission rules are
   *reasoned* — `Action requiresCapability X`, `Agent providesCapability X` ⇒
   `Agent mayInvoke Action` is inferred, not hand-maintained.
2. **Graph-native Rust compute.** PageRank, spectral, VF2, and the entire quant
   kernel suite run in the `epistemic-graph` Tokio engine over MessagePack/UDS —
   no Python overhead, no DB round-trip. AIP leans on backend DB compute.
3. **Closed-loop self-evolution.** The golden loop (research→synthesize→variant
   pool→evals→promote) and GEPA prompt optimization mean the system *improves its
   own skills and prompts from eval failures*. AIP has no architectural
   self-improvement primitive.
4. **Reward-weighted routing.** `capability_index.record_outcome()` feeds an EMA
   back into `designate()`, so routing learns from outcomes with no retraining.
5. **Vendor-neutral, MCP-first, multi-tenant cohabitation.** Humans and agents
   share one governed KG; any MCP client (Claude Code, Devin, …) can query/mutate
   under the same policy fabric. AIP is a closed platform.

## 5. Genuine gaps (and disposition)

| Gap | Severity | Disposition |
|---|---|---|
| **Ontology Actions as first-class verbs** (governed, parameterized, audited) | HIGH | **CLOSED — `CONCEPT:KG-2.25` Ontology Action System** |
| **Declarative retry/backoff/fallback** (L7 reliability) | MEDIUM | **CLOSED — `CONCEPT:ORCH-1.36` Resilience Policy** |
| Formal HITL escalation matrix | MEDIUM | Backlog — `approval_manager.py` exists; needs an escalation/cost-benefit policy |
| Release channels (beta/stable/edge, canary) | MEDIUM | Backlog — packaging/CI exists; release-track system is infra, deferred |
| Langfuse deep core integration (auto span export) | LOW | Backlog — OTel present; wire a Langfuse exporter |
| Auto-merge of golden-loop skill proposals | MEDIUM | Backlog (intentional safety choice — propose-only today) |

## 6. Implementation plan

**Delivered this cycle (fully implemented, no stubs, tested, concept-governed):**

- **`CONCEPT:KG-2.25` — Ontology Action System.** First-class `OntologyAction`
  verbs operating on ontology objects: `ActionRegistry`, permission-gated +
  audited + KG-persisted `ActionExecutor`, OWL `:Action` class with
  `actsOn` / `requiresCapability` / `hasParameter` / `producesEffect`, a SHACL
  shape validating action definitions. Directly realizes Palantir's
  data+logic+**actions**+security unification — and, because of our substrate,
  invocation eligibility is *reasoned* and every invocation is a queryable KG node.
- **`CONCEPT:ORCH-1.36` — Resilience Policy.** Declarative retry (max attempts,
  exponential backoff + jitter, retry-on exception predicate), fallback chain, and
  timeout, wired into the live specialist-execution path in `orchestration/`
  (wire-first, not a sidecar).
- **Investor-persona debate voices** (`CONCEPT:KG-2.6`) — Bull/Bear `DebateEngine`
  now loads persona prompt bodies (Buffett vs Burry, any stem); archetype stamped
  on each argument for the audit trail.

**Backlog (scoped, not stubbed):** HITL escalation matrix, release channels,
Langfuse exporter, golden-loop auto-merge. Each is tracked as a future concept;
none are half-implemented in the tree.

---

*This document is the source of truth for the AIP comparison. Capability claims
are anchored to concrete modules above; re-audit when pillars change.*
