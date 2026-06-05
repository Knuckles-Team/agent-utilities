# Agent Utilities Ecosystem

> [!NOTE]
> Welcome to the official documentation for the `agent-utilities` ecosystem — the infrastructural substrate for multi-agent organizational intelligence.

This repository provides the core primitives, memory engines, orchestration layers, and execution sandboxes required to build, deploy, and scale autonomous AI agents within an enterprise context.

---

## 🌟 Start Here: The Narrative Journey

> [!TIP]
> **Experience the Platform in Action**
> Read our comprehensive **[Technical Novel: The Narrative Journey](journey.md)** to see all 5 pillars and the unified concept architecture trace the high-stakes execution of a quantitative portfolio rebalancing mandate — and how the same brain now runs a **full enterprise** (ITSM, ERP, BPM, EA) across whichever vendor tools a client deploys, made **trusted and self-correcting** by source-authority survivorship, data permissions, and a human-correction → rule loop. It's the best way to understand how the pieces fit together.

---

## 🏛️ The 5-Pillar Architecture

The entire ecosystem is organized into five foundational pillars, each handling a distinct layer of organizational intelligence.

| # | Pillar | Summary | Key Capability |
|:-:|:-------|:--------|:---------------|
| **1** | **[Graph Orchestration](pillars/1_graph_orchestration.md)** | Routing, planning, execution, and state management via directed acyclic graphs. | Routes work to the right agent/model |
| **2** | **[Epistemic Knowledge Graph](pillars/2_epistemic_knowledge_graph.md)** | The Single Company Brain: Memory, ontology, retrieval, and structural reasoning. | Maintains organizational state with provenance |
| **3** | **[Agentic Harness](pillars/3_agentic_harness_engineering.md)** | Continuous evaluation, interpretability, and self-improvement loops. | Makes the system smarter over time |
| **4** | **[Ecosystem & Peripherals](pillars/4_ecosystem_peripherals.md)** | Dynamic capability discovery, MCP servers, connectors, and governance policy. | Connects to external systems securely |
| **5** | **[Agent OS Infrastructure](pillars/5_agent_os_infrastructure.md)** | Kernel, permissions, vllm.arpa context constraints, observability, and safety sandboxes. | Wraps everything in policy and compliance |

---

## 🧠 The Single Company Brain (Pillar 2 Deep-Dive)

The **Single Company Brain** (`CONCEPT:KG-2.7`) is the operational state layer that transforms the Epistemic Knowledge Graph into a multi-writer, multi-reader, multi-tenant organizational memory. It is strictly governed by Ontology Alignment Bridges and Entailment-Aware Permission Scopers.

| Document | Description |
|:---------|:------------|
| **[Company Brain Index](pillars/2_epistemic_knowledge_graph/company_brain/00_index.md)** | Master overview, quick start, and architectural registration |
| **[Architecture](pillars/2_epistemic_knowledge_graph/company_brain/architecture.md)** | 5-layer architecture, data flow, design principles |
| **[Ontology & Permissions](pillars/2_epistemic_knowledge_graph/company_brain/ontology.md)** | OWL reasoning cycles, BFO/PROV-O alignment, and SHACL validation |

---

## 📚 Key Reference Guides

> [!IMPORTANT]
> Dive deep into the specific subsystems that power the `agent-utilities` ecosystem.

- **[Concept Galaxy (overview.md)](overview.md)**: High-level map of all unified canonical concepts across the 5 pillars.
- **[Canonical Concept Map](concept_map.md)**: The 1:1 traceability matrix mapping concepts to code modules.
- **[Architecture Deep-Dive](guides/architecture.md)**: Pillar 1 detailed structural overview.
- **[Agent OS Architecture](guides/agent-os-architecture.md)**: Pillar 5 kernel and execution boundaries.
- **[Gateway Service Dashboard](pillars/5_agent_os_infrastructure/OS-5.9-Gateway_Service_Dashboard.md)**: The real-time observability cockpit.

---

## 🛠️ Documentation Standards

We employ a strict **Concept ID Registry** to ensure 1:1:1 traceability between **Code** (Docstrings), **Tests**, and **Documentation**.

If you are contributing documentation, please adhere to the standard file naming conventions:
- Pillar summary: `{N}_{pillar_name}.md`
- Concept reference: `{ID}-{Name}.md` (e.g. `KG-2.5-Topological_Analysis.md`)

> [!NOTE]
> All new concept proposals must go through the DSTDD design phase. See `.specify/design/_template.md` for the required KG analysis.
