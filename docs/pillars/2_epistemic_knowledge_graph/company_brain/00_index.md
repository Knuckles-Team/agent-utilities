# Company Brain Documentation

> **Own the ontology. Own the policy. Own the judgment.**
> The Company Brain is not another app — it is the infrastructure layer that every app sits on.

Welcome to the Company Brain documentation for `agent-utilities`. This directory contains comprehensive, verbose documentation covering every aspect of the Company Brain architecture — the operational state infrastructure that transforms a single-agent knowledge graph into a multi-writer, multi-reader, multi-tenant organizational brain.

---

## What Is the Company Brain?

The Company Brain is the infrastructure layer that lets every app, agent, workflow, and human decision surface act from the **same company state**. It is not a chatbot. It is not a knowledge base. It is not tool access. It is **operational state infrastructure** with:

- **Provenance** — Who wrote this, why, from what source, and with what confidence
- **Permissions** — Who can see what, inherited across tenant hierarchies
- **Ontology** — Company-specific perspective applied to raw data via OWL reasoning
- **Action Traces** — What action followed a decision, with append-only audit trails
- **Concurrency Control** — Multiple agents and humans writing the same state safely
- **Conflict Resolution** — What happens when two actors disagree
- **Evals** — Did agents have the right context before they acted
- **Multi-Tenancy** — Multiple teams with isolated data and inherited permissions

### Actor-Agnostic Design

The Company Brain treats **humans, AI agents, automated services, and hybrid human+AI teams** as equal first-class participants. There is no hierarchy between organic and synthetic intelligence — every primitive applies the same rules regardless of who (or what) is interacting with the brain. An `ActorType` enum distinguishes them for provenance and audit purposes, but the infrastructure itself imposes no capability differences.

---

## Documentation Index

| Document | Description |
|:---------|:------------|
| [Architecture](architecture.md) | Full architectural deep-dive: state graph, mixin composition, backend abstraction, 5-Pillar integration |
| [Concurrency Control](concurrency.md) | Version vectors, Compare-And-Swap, graph-level locking, multi-writer safety |
| [Multi-Tenancy](multi_tenancy.md) | Tenant isolation, hierarchies, scoped queries, membership management |
| [Conflict Resolution](conflict_resolution.md) | Contradiction detection, merge strategies, source arbitration |
| [Provenance](provenance.md) | PROV-O integration, trust hierarchies, read audits, mandatory attribution |
| [Event Streaming](event_streaming.md) | Real-time ingestion, webhook adapters, CDC patterns |
| [Permissions](permissions.md) | Data-level permissions, node ACLs, classification labels |
| [Ontology](ontology.md) | OWL ontology design, BFO/PROV-O/SKOS alignment, reasoning cycles |
| [Gap Analysis](gap_analysis.md) | Maturity scorecard across 12 Company Brain dimensions |
| [Roadmap](roadmap.md) | Strategic implementation roadmap and phased plan |

---

## Quick Start

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import (
    ActorType, DataClassification, MergeStrategy
)

# Initialize the Company Brain
brain = CompanyBrain(
    default_merge_strategy=MergeStrategy.HIGHEST_CONFIDENCE_WINS,
    enforce_provenance=True,
)

# Create a tenant for a team (human-created)
engineering = brain.tenancy.create_tenant(
    "Engineering",
    created_by="director:sarah",
    created_by_type=ActorType.HUMAN,
)

# Add both human and AI members
brain.tenancy.add_member(
    "analyst:jane", ActorType.HUMAN,
    engineering.tenant_id, role="admin"
)
brain.tenancy.add_member(
    "agent:code-reviewer", ActorType.AI_AGENT,
    engineering.tenant_id, role="member"
)

# Track a node for concurrency control
brain.concurrency.track_node("service:api-gateway")

# Set data-level permissions
brain.permissions.classify_node(
    "service:api-gateway",
    DataClassification.CONFIDENTIAL,
    data_owner="analyst:jane"
)

# Record provenance on a write
brain.provenance.record_write(
    node_id="service:api-gateway",
    actor_id="agent:code-reviewer",
    actor_type=ActorType.AI_AGENT,
    action="update",
    rationale="Detected configuration drift in health check endpoint",
    confidence=0.92,
)

# Check status of all subsystems
print(brain.status())
```

---

## Key Source Files

| File | Purpose |
|:-----|:--------|
| `agent_utilities/knowledge_graph/core/company_brain.py` | Infrastructure module: 6 composable classes + CompanyBrain facade |
| `agent_utilities/models/company_brain.py` | Pydantic models: enums, data structures, node/edge types |
| `agent_utilities/knowledge_graph/core/engine.py` | IntelligenceGraphEngine — the state graph substrate |
| `agent_utilities/knowledge_graph/core/kg_versioning.py` | KGVersionEngine — git-like transactional mutations |
| `agent_utilities/knowledge_graph/core/owl_bridge.py` | OWLBridge — promote→reason→downfeed reasoning cycles |
| `agent_utilities/knowledge_graph/core/maintainer.py` | GraphMaintainer — 13 autonomous maintenance operations |
| `agent_utilities/knowledge_graph/memory/optimization_engine.py` | SynthesisEngine — homeostatic memory distillation |
| `agent_utilities/security/permissions_kernel.py` | PermissionsKernel — RBAC tool-level access control |
| `agent_utilities/observability/audit_logger.py` | AuditLogger — append-only compliance logging |
| `agent_utilities/knowledge_graph/ontology.ttl` | OWL ontology (~26KB) with BFO/PROV-O/SKOS/FIBO alignment |

---

## Concept Registration

The Company Brain infrastructure is registered under **CONCEPT:KG-2.6** in the 5-Pillar ecosystem:

- **Pillar**: KG-2 (Knowledge Graph & Retrieval)
- **Concept ID**: KG-2.6
- **Name**: Company Brain Infrastructure
- **Module**: `agent_utilities.knowledge_graph.core.company_brain`
- **Models**: `agent_utilities.models.company_brain`
- **Status**: Production-ready

### Related Concepts

| Concept | Name | Relationship |
|:--------|:-----|:-------------|
| KG-2.0 | IntelligenceGraphEngine | Foundation substrate |
| KG-2.3 | Structural Fingerprint Engine | Staleness detection |
| KG-2.7 | Cross-Pillar Synergy Engine | Ecosystem integration |
| OS-5.1 | PermissionsKernel | Tool-level permissions (extended by data-level) |
| OS-5.3 | Session Concurrency | Session-level locking (extended by graph-level) |
| OS-5.6 | AuditLogger | Append-only audit trails (extended by read audits) |
| ORCH-1.2 | Squeeze-Evolve Routing | Context routing integration |
| AHE-3.3 | TeamConfig | Reward tracking for hybrid teams |
