# KG‑2.8 — Knowledge‑Graph Enrichment & Cross‑Category Interlinking Strategy

> Status: DRAFT strategy (2026‑06‑03). Scope: turn ingestion from a *structural
> index* into a *deeply enriched, cross‑linked epistemic graph* across every
> ingestion category, optimized for **local‑first speed**. Seeds the pytest
> quality audit (its first vertical slice) and the research→implementation
> evolution loop. Related: KG‑2.4 (autonomous analysis), KG‑2.7 (assimilation),
> the backend single‑interface abstraction, and `docs/concepts.yaml`.

## 1. Problem & goal

Today codebase ingestion extracts only `Code`/`Symbol` nodes with `IMPLEMENTS`/
`calls` edges + per‑symbol embeddings. There is **no semantic understanding, no
rich per‑category ontology, and no cross‑category linking**. So the graph can
answer "what calls X" but not "how is this feature implemented", "what DB
patterns exist", "which research concept does this code realize", "which chat
decision led to this spec", or "what pytests need work".

**Goal:** every ingested artifact, in every category, gets five enrichment
layers, and artifacts are linked **within and across** categories — fast,
incremental, and local.

## 2. The five enrichment layers (applied per artifact)

| Layer | What | Engine | Cost lever |
|---|---|---|---|
| L0 Structural | parse → typed entities + structural edges | epistemic‑graph (Rust AST) | fast; offload to Rust |
| L1 Vector | embeddings for every entity | vllm‑embed (bge‑m3) | **batch** (193ms→6.6ms, 29×) |
| L2 Semantic | NL "capability cards" (summary, responsibilities, patterns) | LLM (vllm) map‑reduce | **cache by `content_hash`/`ast_hash`**; hierarchical |
| L3 Relationships | intra‑ + cross‑category typed edges w/ evidence | LLM + embedding + VF2 | candidate‑gen by embedding, LLM only confirms |
| L4 Inference | derived facts (transitive, subclass, rules) | OWL/datalog (Rust) | run once at end, incremental |

Principle: **structure is cheap (Rust), meaning is expensive (LLM)** — so spend
LLM only where embeddings/structure can't decide, and never recompute an
unchanged `content_hash`.

## 3. The `__bus__` rethink — separate the planes (top‑down)

`__bus__` was designed as a *shared agent‑communication channel*. That is the
right idea for **coordination**, but it became overloaded as ingest scratch +
durable knowledge + comms on one tenant — which saturates the single daemon. We
split into explicit planes (one daemon, multiple tenants):

- **Coordination plane — `__bus__`**: agent‑to‑agent comms, events, the task
  queue, live shared signals. Keep as designed. Low‑volume, latency‑sensitive.
- **Knowledge plane — `kg` tenant (+ pggraph L3 durable)**: the enriched
  entity/relationship graph. The durable product. (Recommend migrating durable
  knowledge OFF `__bus__` into a dedicated `kg` tenant so comms churn never bloats
  the knowledge graph and vice‑versa.)
- **Scratch plane — ephemeral `stage_<job>` tenants**: per‑ingest extraction,
  merged into the knowledge plane then dropped. **(Implemented — KG‑2.8 item C.)**

This keeps `__bus__` as the clean shared channel you intended, while ingestion
and the knowledge store get their own isolated, scalable space.

## 4. Per‑category enrichment (deep relationships within each type)

Each category gets an `EnrichmentExtractor` producing typed entities + intra‑
category edges + a capability card.

- **Code** → `File/Module/Class/Function/Method/Test/Endpoint/DBTable/Query/Config`;
  edges `CALLS/IMPORTS/IMPLEMENTS/TESTS/COVERS/PERSISTS_TO/EXPOSES/CONFIGURES/RAISES`.
  Cards per file→module→repo (hierarchical). Detectors: ORM/SQL→DBTable/Query,
  route decorators→Endpoint, `test_*`+asserts/mocks/fixtures→Test.
- **Skills** → `Skill/Trigger/WorkflowStep`; `USES_TOOL/COMPOSES/TRIGGERED_BY`.
- **MCP servers** → `Server/Tool/Resource` + I/O schemas; `EXPOSES/REQUIRES`.
- **Prompts** → `Prompt/Role/Instruction/Capability`; `TARGETS_MODEL/GRANTS`.
- **Papers/Documents** → `Paper/Concept/Method/Claim/Result`; `CITES/USES_METHOD/SUPPORTS`.
- **Chats/Conversations** → `Conversation/Decision/Insight/ProblemSolved` (LCM episodic);
  `DECIDED/RESOLVED/REFERENCES`.
- **Specs (`.specify`)** → `Spec/Requirement/Plan/Task`; `REQUIRES/PLANS`.

## 5. Cross‑category interlinking (the high‑value layer)

The bridges that make the graph "intrinsic understanding" rather than seven silos.
Unifying substrate: a **shared `Concept`/`Capability`/`Feature` upper ontology** +
a **common embedding space** (every entity embedded with bge‑m3 → similarity is the
universal connector).

| From → To | Edge | How derived |
|---|---|---|
| Skill → MCP Tool | `USES_TOOL` | capability index + schema match |
| Skill/Prompt → Code | `IMPLEMENTED_BY` | embedding + LLM confirm |
| Paper Concept → Code | `REALIZES` / `INSPIRES` | embedding bridge + LLM (this *is* the evolution loop) |
| Chat Decision → Spec/Code | `LED_TO` / `DECIDED` | provenance: why code exists |
| Spec Requirement → Code/Test | `SATISFIED_BY` / `VERIFIED_BY` | trace links |
| MCP Server → Infra node | `DEPLOYED_ON` | ties to infra topology KG |
| Code Feature ⇄ Code Feature (cross‑repo) | `SIMILAR_TO` / `SHARED_PATTERN` / `DEPENDS_ON` | embedding + **VF2** + LLM (microservice/dedup/condensation) |

Mechanisms (cheap→expensive): (1) canonical shared `Concept` nodes (from
`concepts.yaml` + embedding clustering); (2) embedding‑similarity candidate edges;
(3) VF2 structural isomorphism for recurring patterns; (4) LLM relationship
extraction (typed + evidence) only on candidates; (5) OWL/datalog inference for
transitive/derived facts.

## 6. Speed & efficiency (local‑first)

1. **Ephemeral scratch tenants** (done) — ingest never contends with comms/knowledge.
2. **Incremental by hash** — skip any entity whose `content_hash`/`ast_hash` is
   unchanged; enrichment is a diff, not a rebuild.
3. **Batch everything LLM** — embeddings batched (29×); capability cards generated
   concurrently, hierarchically (leaf→root map‑reduce).
4. **Offload structure to Rust** — community/centrality/PageRank/VF2/datalog run in
   epistemic‑graph, not Python (verified primitives exist).
5. **Decouple enrichment from ingestion** — L0/L1 inline (fast); L2–L4 as an async
   enrichment queue so ingestion stays quick and enrichment fills in.
6. **Bulk‑ingest profile + daemon gating** (done — `KG_INGEST_PROFILE=structural`,
   `KG_BULK_INGEST`).

## 7. Ontology expansion

- `ontology_software.ttl` (code/test/architecture), `ontology_research.ttl`
  (papers/concepts/methods), `ontology_agentic.ttl` (skills/prompts/mcp/agents),
  plus an **upper ontology** linking them via shared `Concept/Capability/Feature`.
- Expand the `owl_bridge` promotion whitelist + add SWRL‑style rules:
  `MockHeavyTest`, `UntestedCode`, `DormantTest`, `RealizesConcept`,
  `CrossRepoDuplicate`, transitive `depends_on`, etc. Downfeed inferred edges so
  the agent/LLM can query them.

## 8. Architecture: one framework, many extractors

A single `EnrichmentPipeline` with: per‑category `EnrichmentExtractor` plugins
(L0–L2), a shared `CrossLinker` (L3), and an `InferencePass` (L4). Backend‑agnostic
(rides the `GraphBackend` single interface + conformance suite). Same code path for
local single‑repo, the agent‑utilities monorepo, and enterprise GitLab/GitHub
(scale = more workers + more scratch tenants + incremental hashing).

## 8b. Implementation status (2026-06-03)

- **Phase 0 — DONE:** backend-contract fixes, scratch tenants (item C), conformance suite.
- **Phase 1 — DONE:** Code/Test slice. Rust `ParseFile` emits native test metrics;
  `enrichment/` maps→classifies→writes; "which pytests need work" is a graph query.
- **Phase 2 — DONE:** full code understanding. Rust emits class facts
  (bases/methods/decorators/abstract); `patterns.py` detects design patterns;
  `features.py` clusters the call graph via the **engine's community detection**;
  `cards.py` generates LLM capability cards ("how is it implemented", thinking
  disabled, cached by ast_hash); queries `how_implemented`/`code_by_pattern`/
  `list_features`; `ontology_software.ttl` + `owl_bridge` promotable types. CLI:
  `python -m agent_utilities.knowledge_graph.enrichment <path> [--features --cards
  --pattern X --how NAME]`.
- **Phase 3 — NEXT:** other categories (skills/mcp/prompts/papers/chats/specs) +
  `CrossLinker` (REALIZES/IMPLEMENTED_BY/SIMILAR_TO via embeddings+VF2) + OWL
  cross-category inference; batched/concurrent card generation for scale.

## 8c. Phase 3 design — cross-ingestion discovery & codebase evolution

The payoff layer: find relationships across ingestion points by topic/goal, and
turn ingested research/documents into concrete codebase enhancements.

**Document/concept extraction (all non-code categories).** A `DocumentExtractor`
produces, per artifact: a `Document` node with **metadata** (type-aware: paper →
title/authors/abstract/methods/claims; email → from/to/subject/date; BRD →
requirements; SOW → deliverables/terms; book → chapters/topics), chunk +
**vector embeddings**, and LLM-extracted **`Concept`** nodes (key ideas/
techniques/claims) with `MENTIONS` edges. Concepts are the universal bridge.

**Shared concept space + semantic cross-linking.** Every entity (code symbol,
feature, concept, document) is embedded in one space (bge-m3). The `CrossLinker`
uses the **engine's semantic search (HNSW/cosine — compute layer)** to propose
cross-category edges, optionally LLM-confirmed:
`RELATES_TO`, `REALIZES` (paper concept → code that implements it), `MENTIONS`,
`SIMILAR_TO`. This is how "find everything related to topic/goal X" works:
embed the topic → nearest entities across ALL categories → return the subgraph.

**Goal/topic-driven discovery.** `find_related(topic_or_goal)` → ranked
cross-ingestion matches + relationship subgraph. Topics/goals can be standing
(stored) so discovery runs continuously as new content is ingested.

**Research → codebase evolution loop (the headline).** For a target codebase
(e.g. `agent-packages/agents/<x>`):
1. **Relevance + gap**: rank ingested Concepts (from papers/docs) by value to the
   codebase = relevance (embedding sim to the codebase's features/cards) ×
   novelty (NOT already `REALIZES`-linked) × impact (concept centrality / source
   quality).
2. **Feature distillation**: LLM turns top concepts + the codebase's current
   capability cards into ranked **enhancement proposals** (`distill_enhancements`).
3. **Spec distillation**: top enhancements → **SDD-format specs** written into the
   codebase's `.specify/` (feeds the SDD skill). `what_specs_could_we_build(codebase)`
   returns value-ranked spec proposals with their KG evidence (which papers/
   concepts/code drove each).
4. **Action plan**: spec → task breakdown → hand to the implementer.

So: feature extraction → spec distillation → action plan → implement/evolve,
all value-ranked from the KG's codebases + papers + documents. Same framework,
LLM injectable, embeddings/search on the engine, cached by content hash.

## 9. Phased rollout

- **Phase 0 (done):** backend‑contract correctness, item‑C scratch tenants,
  conformance suite, bulk‑ingest speed.
- **Phase 1:** `EnrichmentPipeline` framework + **Code/Test vertical slice** +
  capability cards + `ontology_software.ttl` + needs‑work axioms → **answers the
  pytest‑audit questions natively** (the audit becomes a KG query).
- **Phase 2:** per‑category extractors + ontologies for skills/mcp/prompts/papers/
  chats/specs.
- **Phase 3:** `CrossLinker` + upper ontology + OWL inference (cross‑category).
- **Phase 4:** query/agent layer — NL question → `designate()` → reasoned, cited
  answer over capability cards + inferred facts.

## 10. Success criteria

Ask the graph (not a script): "what pytests need work", "how is feature X
implemented", "what DB patterns are used", "which code realizes paper concept Y",
"which services share this feature" — and get reasoned, cited answers. Re‑ingest of
unchanged code is ~free (hash‑incremental). Enterprise‑scale is a bounded batch,
incremental thereafter.
