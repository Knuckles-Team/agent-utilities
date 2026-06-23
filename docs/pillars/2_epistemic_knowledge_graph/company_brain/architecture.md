# Company Brain Architecture

> Deep-dive into the architectural foundations of the Company Brain — the operational state infrastructure that makes the Knowledge Graph safe for multi-writer, multi-reader, multi-tenant organizational intelligence.

---

## The Problem: From Developer Brain to Company Brain

Traditional knowledge graphs and RAG systems are designed for a single pattern: **one agent writes, one agent reads, context is reconstructed at query time**. This works for a developer's personal AI assistant. It does not work for an organization.

An organization has:

- **Multiple writers** — 50 AI agents, 200 humans, 30 automated services, all updating state simultaneously
- **Multiple readers** — Every app, dashboard, workflow, and decision surface needs the same state
- **Multiple tenants** — Engineering can't see HR's compensation data; the trading desk can't see compliance's investigation notes
- **Conflicting interpretations** — Agent A says "this customer is low risk" while Agent B says "this customer is high risk"
- **Regulatory constraints** — PII must be access-controlled; financial data must have audit trails
- **Temporal dynamics** — Information goes stale; decisions made yesterday may be wrong today

The Company Brain solves this by treating the Knowledge Graph not as a **storage system** but as **operational state infrastructure** — the same way a database treats data not as files but as transactional, consistent, isolated state.

---

## Architectural Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Company Brain Facade                         │
│                 (CompanyBrain class)                            │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│Concurrency│ Tenancy │ Conflict │Provenance│  Events  │Permiss- │
│ Manager  │ Manager │ Resolver │ Tracker  │Ingester  │  ions   │
├──────────┴──────────┴──────────┴──────────┴──────────┴─────────┤
│              IntelligenceGraphEngine (KG-2.0)                  │
│   ┌─────────┬─────────┬──────────┬──────────┬────────────┐    │
│   │ Query   │ Memory  │Ingestion │   AHE    │ Federation │    │
│   │ Mixin   │ Mixin   │  Mixin   │  Mixin   │   Mixin    │    │
│   └─────────┴─────────┴──────────┴──────────┴────────────┘    │
├───────────────────────────────────────────────────────────────┤
│                   Graph Backends                               │
│   ┌──────────────┬──────────┬──────────┬─────────────────┐   │
│   │epistemic-graph│ Postgres │   OWL    │ contrib (neo4j, │   │
│   │ (authority)  │ (mirror) │ (reason) │ falkordb, …)    │   │
│   └──────────────┴──────────┴──────────┴─────────────────┘   │
├───────────────────────────────────────────────────────────────┤
│                  OWL Ontology (~26KB)                          │
│          BFO / PROV-O / SKOS / FIBO aligned                   │
└───────────────────────────────────────────────────────────────┘
```

### Layer 1: OWL Ontology (Bottom)

The foundation is the **OWL ontology** (`ontology.ttl`, ~26KB) aligned with:
- **BFO** (Basic Formal Ontology) — Upper ontology for foundational categories
- **PROV-O** — W3C Provenance Ontology for attribution and derivation
- **SKOS** — Simple Knowledge Organization System for taxonomic hierarchies
- **FIBO** — Financial Industry Business Ontology for financial domain concepts

This ontology defines the company-specific perspective that transforms raw data into organizational knowledge. It is not a static schema — the `OWLBridge` runs promote→reason→downfeed cycles that discover new facts through transitive closure, symmetric property inference, and RDFS+ reasoning.

### Layer 2: Graph Backends

The `GraphBackend` abstraction supports multiple storage backends:

| Backend | Use Case | ACID | Vector Search | Scale |
|:--------|:---------|:-----|:--------------|:------|
| **epistemic-graph** | The one authority / system of record (Rust, out-of-process UDS client) — compute + cache + semantic + durable persistence | ACID | Via embeddings | Single-node |
| **PostgreSQL (pg-age)** | Optional mirror (eventually consistent, write-only fan-out from the authority) | Full | pgvector | Cluster |
| **OWL/RDFLib** | Reasoning-only | N/A | N/A | In-memory |
| **contrib** (Neo4j, FalkorDB, LadybugDB) | Optional mirrors under `backends/contrib/` | Varies | Varies | Varies |

### Layer 3: IntelligenceGraphEngine

The core engine uses **mixin composition** to assemble capabilities:

- **QueryMixin** — Cypher execution, semantic search, hybrid retrieval
- **MemoryMixin** — Memory CRUD, embedding generation, node linking
- **IngestionMixin** — Episode, MCP, A2A, skill, and batch ingestion
- **AHEMixin** — Self-improvement, experience tracking, reward signals
- **RegistryMixin** — Agent/tool/skill discovery and registration
- **FederationMixin** — External ontology federation, SPARQL endpoints

### Layer 4: Company Brain Infrastructure

Six composable classes that sit **above** the engine and **below** the application:

1. **GraphConcurrencyManager** — Version vectors, CAS, graph-level locks
2. **TenancyManager** — Tenant isolation, hierarchies, scoped queries
3. **ConflictResolver** — Contradiction detection, configurable merge strategies
4. **ProvenanceTracker** — Trust hierarchies, read audits, mandatory attribution
5. **EventStreamIngester** — Webhook adapters, async ingestion, CDC
6. **DataLevelPermissions** — Node ACLs, classification labels, query filtering

### Layer 5: CompanyBrain Facade

A single entry point that composes all six primitives:

```python
brain = CompanyBrain(
    default_lock_mode=LockMode.OPTIMISTIC,
    default_merge_strategy=MergeStrategy.HIGHEST_CONFIDENCE_WINS,
    enforce_provenance=True,
)
```

---

## Design Principles

### 1. Actor-Agnostic

The Company Brain does not distinguish between human and AI intelligence in its infrastructure. Both are `actors` with:
- An `actor_id` (unique identifier)
- An `ActorType` (human, ai_agent, automated_service, hybrid_team, system)
- Identical permission evaluation
- Identical provenance tracking
- Identical conflict resolution

This means a human analyst and an AI agent writing to the same node are subject to the **exact same** concurrency control, conflict detection, and permission checks.

### 2. Ontology-First

Most systems start with storage and bolt ontology on later. We start with OWL and bolt storage underneath. This means:
- Company-specific perspective is **native** — extend `ontology.ttl` without changing the engine
- Reasoning is **built-in** — `OWLBridge.run_cycle()` discovers new facts automatically
- Semantic subsumption enables **lensing** — the same state viewed through different perspectives

### 3. Composable Infrastructure

Each of the six primitives is a standalone class that can be used independently:

```python
# Use only concurrency control
from agent_utilities.knowledge_graph.core.company_brain import GraphConcurrencyManager
gcm = GraphConcurrencyManager()

# Use only permissions
from agent_utilities.knowledge_graph.core.company_brain import DataLevelPermissions
dlp = DataLevelPermissions()
```

Or composed together via the `CompanyBrain` facade:

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
brain = CompanyBrain()
brain.concurrency.track_node("customer:001")
brain.permissions.classify_node("customer:001", DataClassification.CONFIDENTIAL)
```

### 4. Provenance-Native

Every mutation to the Company Brain carries mandatory provenance metadata:
- **Who** wrote it (`actor_id`, `actor_type`)
- **What** assertion type (`raw_data`, `agent_inference`, `human_judgment`)
- **Why** it was written (`rationale`)
- **From where** it was derived (`derived_from`, `source_system`)
- **How confident** the writer is (`confidence` score 0.0–1.0)

### 5. Self-Maintaining

The Company Brain is not judged by the first answer — it is judged by what happens after six months. Infrastructure primitives include:
- **Temporal decay** — Importance scores decay over time (Ebbinghaus-style)
- **Concept merging** — Similar concepts are automatically merged
- **Memory synthesis** — Episodes distill into Preferences and Principles
- **Staleness detection** — Fingerprint-based change classification
- **OWL reasoning** — Autonomous promote→reason→downfeed cycles

---

## 5-Pillar Integration

The Company Brain is not a standalone product — it is the **substrate** that the 5-Pillar ecosystem sits on:

| Pillar | How It Integrates with Company Brain |
|:-------|:-------------------------------------|
| **ORCH-1.x** (Orchestration) | Routes work to agents based on Company Brain state; uses tenant-scoped routing |
| **KG-2.x** (Knowledge Graph) | **IS** the Company Brain substrate; all 6 primitives extend KG-2.0 |
| **AHE-3.x** (Harness Engineering) | Self-improvement feeds back into the brain; reward signals update provenance |
| **ECO-4.x** (Ecosystem) | 40-repo ecosystem provides peripheral sensors; event streaming ingests ecosystem state |
| **OS-5.x** (Agent OS) | Permissions, security, observability wrap the brain in governance |

---

## Data Flow

```
External Events                    Actors (Human + AI)
    │                                      │
    ▼                                      ▼
EventStreamIngester              ProvenanceTracker
    │                                      │
    ▼                                      ▼
┌──────────────────────────────────────────────┐
│           GraphConcurrencyManager            │
│     (Version Vectors, CAS, Locks)            │
├──────────────────────────────────────────────┤
│              ConflictResolver                │
│   (Detect → Strategy → Resolve/Escalate)     │
├──────────────────────────────────────────────┤
│              TenancyManager                  │
│    (Tenant Scoping, Hierarchy, Isolation)     │
├──────────────────────────────────────────────┤
│           DataLevelPermissions               │
│     (Node ACLs, Classification, Filtering)    │
├──────────────────────────────────────────────┤
│        IntelligenceGraphEngine               │
│  (epistemic-graph authority + opt. mirrors)  │
└──────────────────────────────────────────────┘
                    │
                    ▼
            OWL Ontology (~26KB)
         BFO/PROV-O/SKOS/FIBO
```

1. **External events** arrive via `EventStreamIngester` (webhooks, Kafka, CDC)
2. **Actors** (human or AI) submit mutations via the engine
3. **Concurrency control** checks version vectors and acquires locks
4. **Conflict resolution** detects contradictions and applies merge strategies
5. **Tenant scoping** ensures mutations land in the correct namespace
6. **Permission checks** verify the actor can write to the target node
7. **The engine** applies the mutation with full provenance metadata
8. **OWL reasoning** triggers if the mutation crosses significance thresholds

---

## Relationship to Existing Infrastructure

The Company Brain does **not** replace existing infrastructure. It **extends** it:

| Existing | Company Brain Extension |
|:---------|:----------------------|
| `KGVersionEngine` (KG-2.0) | + Version vectors, CAS, multi-writer safety |
| `PermissionsKernel` (OS-5.1) | + Data-level ACLs, classification labels |
| `AuditLogger` (OS-5.6) | + Read audit trails, provenance chains |
| `SynthesisEngine` (KG-2.0) | + Conflict-aware synthesis |
| `AsyncioConcurrencyManager` (OS-5.3) | + Graph-level locking (not just session-level) |
| `OWLBridge` (KG-2.0) | + Continuous reasoning triggers |
| `ingest_external_batch` | + Real-time event streaming |
