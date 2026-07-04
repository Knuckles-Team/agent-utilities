# Agent Utilities Documentation

> The complete technical documentation for the `agent-utilities` ecosystem — the infrastructure substrate for multi-agent organizational intelligence.

---

> [!NOTE]
> **Experience the Platform in Action**
> Read our comprehensive **[Technical Novel: The Narrative Journey](journey.md)** to see all 5 pillars and 70 concepts trace the high-stakes execution of a quantitative portfolio rebalancing mandate.

---

## How This Documentation Is Organized

All documentation is organized under the **5-Pillar Architecture**. Each pillar has:
- A **summary page** (`{pillar_name}.md`) explaining the rationale, implementation, and benefits
- A **directory** containing detailed concept references, guides, and deep-dives
- **Concept IDs** (`CONCEPT:AU-ORCH.planning.orchestration-overview`, `AU-KG.compute.kg-x`, etc.) linking docs to source code 1:1

```
docs/
├── index.md                          ← You are here
├── journey.md                        ← The Technical Novel: Narrative Journey
├── overview.md                       ← Concept Galaxy (70 canonical concepts, Mermaid map)
│
└── pillars/
    ├── 1_graph_orchestration.md       ← Pillar 1 summary
    ├── 1_graph_orchestration/         ← Concept references & guides
    │   ├── architecture.md
    │   ├── agents.md
    │   ├── ORCH-1.3-Execution_&_State_Safety.md
    │   └── ...
    │
    ├── 2_epistemic_knowledge_graph.md ← Pillar 2 summary
    ├── 2_epistemic_knowledge_graph/   ← Concept references & guides
    │   ├── knowledge-graph.md
    │   ├── KG-2.5-Topological_Mincut_Partitioning.md
    │   ├── enterprise_ingestion.md
    │   ├── company_brain/             ← Company Brain deep-dive (11 pages)
    │   │   ├── 00_index.md
    │   │   ├── architecture.md
    │   │   ├── concurrency.md
    │   │   ├── multi_tenancy.md
    │   │   ├── conflict_resolution.md
    │   │   ├── provenance.md
    │   │   ├── event_streaming.md
    │   │   ├── permissions.md
    │   │   ├── ontology.md
    │   │   ├── gap_analysis.md
    │   │   └── roadmap.md
    │   └── ...
    │
    ├── 3_agentic_harness_engineering.md
    ├── 3_agentic_harness_engineering/
    │   ├── AHE_ARCHITECTURE.md
    │   ├── AHE-3.7-Heavy_Thinking_Orchestration.md
    │   └── ...
    │
    ├── 4_ecosystem_peripherals.md
    ├── 4_ecosystem_peripherals/
    │   └── ...
    │
    ├── 5_agent_os_infrastructure.md
    └── 5_agent_os_infrastructure/
        ├── permissions-kernel.md
        ├── OS-5.3-Session_Concurrency_Management.md
        ├── OS-5.9-Gateway_Service_Dashboard.md
        └── ...
```

---

## Quick Navigation

### Getting Started

| Document | Description |
|:---------|:------------|
| [Technical Novel: The Narrative Journey](journey.md) | Experience `agent-utilities` live via a high-stakes quantitative rebalancing story |
| [Concept Galaxy (overview.md)](overview.md) | High-level map of all 70 canonical concepts across 5 pillars |

### The 5 Pillars

| # | Pillar | Summary | Key Capability |
|:--|:-------|:--------|:---------------|
| 1 | [Graph Orchestration](pillars/1_graph_orchestration.md) | Routing, planning, execution, state management | Routes work to the right agent/model |
| 2 | [Epistemic Knowledge Graph](pillars/2_epistemic_knowledge_graph.md) | Memory, ontology, retrieval, reasoning, **Company Brain** | Maintains organizational state with provenance |
| 3 | [Agentic Harness Engineering](pillars/3_agentic_harness_engineering.md) | Evaluation, self-improvement, experience tracking | Makes the system smarter over time |
| 4 | [Ecosystem & Peripherals](pillars/4_ecosystem_peripherals.md) | Tooling, connectors, integrations, governance, 40-repo ecosystem | Connects to external systems and enforces governance |
| 5 | [Agent OS Infrastructure](pillars/5_agent_os_infrastructure.md) | Permissions, security, observability, governance | Wraps everything in policy and compliance |

### Company Brain (Pillar 2 Deep-Dive)

The Company Brain is the operational state layer that transforms the Knowledge Graph into a multi-writer, multi-reader, multi-tenant organizational brain. Actor-agnostic: humans, AIs, and hybrid teams are all first-class participants.

| Document | Description |
|:---------|:------------|
| [Company Brain Index](pillars/2_epistemic_knowledge_graph/company_brain/00_index.md) | Master overview, quick start, concept registration |
| [Architecture](pillars/2_epistemic_knowledge_graph/company_brain/architecture.md) | 5-layer architecture, data flow, design principles |
| [Concurrency](pillars/2_epistemic_knowledge_graph/company_brain/concurrency.md) | Version vectors, CAS, graph-level locking |
| [Multi-Tenancy](pillars/2_epistemic_knowledge_graph/company_brain/multi_tenancy.md) | Tenant hierarchies, scoped queries, membership |
| [Conflict Resolution](pillars/2_epistemic_knowledge_graph/company_brain/conflict_resolution.md) | Contradiction detection, 5 merge strategies |
| [Provenance](pillars/2_epistemic_knowledge_graph/company_brain/provenance.md) | PROV-O, trust hierarchies, read audits |
| [Event Streaming](pillars/2_epistemic_knowledge_graph/company_brain/event_streaming.md) | Real-time ingestion, webhook adapters |
| [Permissions](pillars/2_epistemic_knowledge_graph/company_brain/permissions.md) | Node-level ACLs, data classification labels |
| [Ontology](pillars/2_epistemic_knowledge_graph/company_brain/ontology.md) | OWL reasoning cycles, BFO/PROV-O/SKOS alignment |
| [Gap Analysis](pillars/2_epistemic_knowledge_graph/company_brain/gap_analysis.md) | 12-dimension maturity scorecard |
| [Roadmap](pillars/2_epistemic_knowledge_graph/company_brain/roadmap.md) | 5-phase strategic plan |

### Key Reference Guides

| Guide | Pillar | Path |
|:------|:-------|:-----|
| Architecture Deep-Dive | P1 | [pillars/1_graph_orchestration/architecture.md](pillars/1_graph_orchestration/architecture.md) |
| Knowledge Graph | P2 | [pillars/2_epistemic_knowledge_graph/knowledge-graph.md](pillars/2_epistemic_knowledge_graph/knowledge-graph.md) |
| Enterprise Ingestion | P2 | [pillars/2_epistemic_knowledge_graph/enterprise_ingestion.md](pillars/2_epistemic_knowledge_graph/enterprise_ingestion.md) |
| AHE Architecture | P3 | [pillars/3_agentic_harness_engineering/AHE_ARCHITECTURE.md](pillars/3_agentic_harness_engineering/AHE_ARCHITECTURE.md) |
| Agent OS Architecture | P5 | [pillars/5_agent_os_infrastructure/agent-os-architecture.md](pillars/5_agent_os_infrastructure/agent-os-architecture.md) |
| Permissions Kernel | P5 | [pillars/5_agent_os_infrastructure/permissions-kernel.md](pillars/5_agent_os_infrastructure/permissions-kernel.md) |
| Configuration | P5 | [pillars/5_agent_os_infrastructure/configuration.md](pillars/5_agent_os_infrastructure/configuration.md) |
| **Enterprise Entities** | P2 | [pillars/2_epistemic_knowledge_graph/enterprise_entities.md](pillars/2_epistemic_knowledge_graph/enterprise_entities.md) |
| **Enterprise Agent Governance** | P4 | [pillars/4_ecosystem_peripherals.md#enterprise-agent-governance](pillars/4_ecosystem_peripherals.md#-enterprise-agent-governance-eco-416--eco-422) |
| **Graph DB Deployment** | P2 | [guides/graph-db-deployment.md](guides/graph-db-deployment.md) |
| **OWL Ontology & OS Synergies** | P2/P5 | [owl_kg_synergies.md](owl_kg_synergies.md) |
| **Gateway Service Dashboard** | GW | [pillars/5_agent_os_infrastructure/OS-5.9-Gateway_Service_Dashboard.md](pillars/5_agent_os_infrastructure/OS-5.9-Gateway_Service_Dashboard.md) |

---

## Documentation Standards

### Naming Conventions

| Type | Pattern | Example |
|:-----|:--------|:--------|
| Pillar summary | `{N}_{pillar_name}.md` | `2_epistemic_knowledge_graph.md` |
| Concept reference | `{ID}-{Name}.md` | `KG-2.5-Topological_Mincut_Partitioning.md` |
| Guide / narrative | `{topic}.md` (lowercase, hyphens) | `knowledge-graph.md` |
| Sub-section index | `00_index.md` | `company_brain/00_index.md` |

### Structure of a Concept Reference

Every `CONCEPT:ID` document follows:

```markdown
# {Concept Name} (CONCEPT:{ID})

## Overview
Brief description of what this concept does.

## Why (Rationale)
Why this capability exists.

## How (Implementation)
Key classes, modules, and data flows.

## API Reference
Public API surface with examples.

## Related Concepts
Cross-references to other CONCEPT:IDs.
```

### Structure of a Pillar Summary

```markdown
# Pillar {N}: {Name}

## Overview
## Why We Built This (Rationale)
## How It Works (Implementation)
## Benefits Introduced
## Key Concepts Leveraged
## Enterprise Ontology Alignments (if applicable)
```

---

## Concept ID Registry

All concepts are uniquely identified and traceable to source code:

| Prefix | Pillar | Range |
|:-------|:-------|:------|
| `AU-ORCH.planning.orchestration-overview` | Graph Orchestration Engine | 1.0 – 1.29 |
| `AU-KG.compute.kg-x` | Epistemic Knowledge Graph | 2.0 – 2.20 |
| `AU-AHE.optimization.telemetry-optimization` | Agentic Harness Engineering | 3.0 – 3.16 |
| `ECO-4.x` | Ecosystem & Peripherals | 4.0 – 4.23 |
| `OS-5.x` | Agent OS Infrastructure | 5.0 – 5.10 |
| `GW-1.x` | Gateway Service Dashboard | 1.0 |
| `GBOT-6.x` | GeniusBot Desktop Cockpit | 6.0 – 6.6 |
