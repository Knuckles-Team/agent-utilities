# agent-utilities Canonical Concept Registry

> **Single source of truth** for all CONCEPT: tags in the ecosystem.
> Consolidated to **34 canonical concepts** with gap-free numbering.
> Components marked 🔬 have research-backed enhancements.
>
> **Rule**: All new concept proposals must go through the DSTDD design phase.
> See `.specify/design/_template.md` for the required KG analysis.

---

## Traceability Matrix

Every concept has 1:1:1 traceability across:
- **Code**: `CONCEPT:X.Y` tag in module docstrings
- **Tests**: `CONCEPT:X.Y` tag in test file docstrings
- **Docs**: Dedicated page in `docs/pillars/<pillar>/`

---

## Pillar 1: Graph Orchestration Engine (ORCH)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `ORCH-1.0` | Intelligence Graph Core | 14 | 16 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.1` | HTN Planning Pipeline | 25 | 11 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.2` | Specialist Routing & Discovery | 28 | 14 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.3` | Execution Safety & State | 11 | 2 | [ORCH-1.3](pillars/1_graph_orchestration/ORCH-1.3-Execution_Safety_And_State.md) |
| `ORCH-1.4` | Capability Wiring Engine | 11 | 2 | [ORCH-1.4](pillars/1_graph_orchestration/ORCH-1.4-Capability_Wiring_Engine.md) |
| `ORCH-1.5` | Agent Orchestrator 🔬 | 3 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.6` | DSTDD Pipeline | 3 | 3 | [ORCH-1.6](pillars/1_graph_orchestration/ORCH-1.6-DSTDD_Pipeline.md) |

Key modules: `graph/builder.py`, `graph/nodes.py`, `graph/planner.py`, `graph/routing.py`, `graph/executor.py`, `graph/hsm.py`, `graph/lifecycle.py`, `core/default_catalog.py`, `capabilities/checkpointing.py`, `graph/dynamic_graph_orchestrator.py`, `graph/agent_orchestrator.py`, `sdd/orchestrator.py`, 🔬 `graph/coordination.py`

---

## Pillar 2: Epistemic Knowledge Graph (KG)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `KG-2.0` | Active Knowledge Graph | 42 | 24 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.1` | Tiered Memory & Context 🔬 | 17 | 8 | [KG-2.1](pillars/2_epistemic_knowledge_graph/KG-2.1-Tiered_Memory_And_Context.md) |
| `KG-2.2` | Ontology & Epistemics | 26 | 6 | [KG-2.2](pillars/2_epistemic_knowledge_graph/KG-2.2-Ontology_And_Epistemics.md) |
| `KG-2.3` | Graph Integrity & Retrieval 🔬 | 14 | 4 | [KG-2.3](pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md) |
| `KG-2.4` | Inductive Knowledge & Hypergraphs | 12 | 5 | [KG-2.4](pillars/2_epistemic_knowledge_graph/KG-2.4-Inductive_Knowledge_And_Hypergraphs.md) |
| `KG-2.5` | Topological Analysis | 11 | 4 | [KG-2.5](pillars/2_epistemic_knowledge_graph/KG-2.5-Topological_Analysis.md) |
| `KG-2.6` | Domain: Finance | 46 | 31 | [KG-2.6](pillars/2_epistemic_knowledge_graph/KG-2.6-Domain_Finance.md) |
| `KG-2.7` | Research Intelligence | 2 | 0 | [KG-2.7](pillars/2_epistemic_knowledge_graph/KG-2.7-Research_Intelligence.md) |
| `KG-2.8` | Domain: Enterprise | 13 | 2 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.9` | External Graph Federation | 1 | 0 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |

Key modules: `knowledge_graph/core/engine.py`, `knowledge_graph/core/engine_memory.py`, `knowledge_graph/core/engine_tasks.py`, `knowledge_graph/core/topological_analysis_engine.py`, `knowledge_graph/research/research_intelligence_engine.py`, `knowledge_graph/memory/consolidation.py`, `knowledge_graph/ontology.ttl`, `knowledge_graph/retrieval/retrieval_quality.py`, `knowledge_graph/pipeline/document_deletion.py`, `knowledge_graph/pipeline/document_update.py`, `domains/finance/`, `knowledge_graph/orchestration/engine_enterprise.py`, `knowledge_graph/pipeline/phases/external_graphs.py`, `scripts/install_git_hooks.py`, `scripts/submit_diff.py`

---

## Pillar 3: Agentic Harness Engineering (AHE)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `AHE-3.0` | Agentic Harness Core | 16 | 3 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.1` | Continuous Evaluation Engine | 16 | 7 | [AHE-3.1](pillars/3_agentic_harness_engineering/AHE-3.1-Continuous_Evaluation_Engine.md) |
| `AHE-3.2` | Agentic Evolution Engine | 16 | 5 | [AHE-3.2](pillars/3_agentic_harness_engineering/AHE-3.2-Agentic_Evolution_Engine.md) |
| `AHE-3.3` | Team & Synergy Optimization | 14 | 5 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.4` | Distributed Agentic Evolution | 11 | 1 | [AHE-3.4](pillars/3_agentic_harness_engineering/AHE-3.4-Distributed_Agentic_Evolution.md) |
| `AHE-3.5` | Heavy Thinking & Background Intelligence | 11 | 1 | [AHE-3.5](pillars/3_agentic_harness_engineering/AHE-3.5-Heavy_Thinking_And_Background_Intelligence.md) |
| `AHE-3.6` | Backtest & Curriculum | 10 | 2 | [AHE-3.6](pillars/3_agentic_harness_engineering/AHE-3.6-Backtest_And_Curriculum.md) |

Key modules: `harness/evaluation_engine.py`, `harness/agentic_evolution_engine.py`, `graph/team_composer.py`, `agentic_evolution/forge.py`, `knowledge_graph/orchestration/engine_ahe.py`

---

## Pillar 4: Ecosystem & Peripherals (ECO)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `ECO-4.0` | Tool Interface & MCP Factory | 26 | 18 | [ECO-4.0](pillars/4_ecosystem_peripherals/ECO-4.0-Tool_Interface_And_MCP_Factory.md) |
| `ECO-4.1` | A2A Network & Consensus 🔬 | 7 | 3 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.2` | Community Telemetry & Ecosystem Map | 5 | 1 | [ECO-4.2](pillars/4_ecosystem_peripherals/ECO-4.2-Community_Telemetry_And_Ecosystem_Map.md) |
| `ECO-4.3` | Market Data Connectors | 9 | 3 | [ECO-4.3](pillars/4_ecosystem_peripherals/ECO-4.3-Market_Data_Connectors.md) |
| `ECO-4.4` | KG MCP Server & Execution | 2 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.5` | Terminal Agent Launcher | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |

Key modules: `mcp/server_factory.py`, `mcp/kg_server.py` (incl. `kg_launch_terminal_agent`), `ecosystem/bridge.py`, `protocols/a2a_graph_skill.py`, `protocols/data_connector.py`, `tools/tool_filtering.py`

---

## Pillar 5: Agent OS Infrastructure (OS)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `OS-5.0` | Agent OS Kernel & XDG Paths | 17 | 6 | [Pillar Summary](pillars/5_agent_os_infrastructure.md) |
| `OS-5.1` | Security & Auth | 19 | 8 | [OS-5.1](pillars/5_agent_os_infrastructure/OS-5.1-Security_And_Auth.md) |
| `OS-5.2` | Resource Scheduling 🔬 | 18 | 4 | [Pillar Summary](pillars/5_agent_os_infrastructure.md) |
| `OS-5.3` | Guardrails & Safety | 7 | 3 | [OS-5.3](pillars/5_agent_os_infrastructure/OS-5.3-Guardrails_And_Safety.md) |
| `OS-5.4` | Telemetry & Observability | 7 | 3 | [OS-5.4](pillars/5_agent_os_infrastructure/OS-5.4-Telemetry_And_Observability.md) |

Key modules: `core/paths.py`, `security/guardrails.py`, `security/tool_guard.py`, `core/cognitive_scheduler.py`, `observability/token_tracker.py`, `observability/audit_logger.py`

---

## Concept Lifecycle

```
New Feature Request
       │
       ▼
  ┌────────────────────┐
  │ KG Analogy Search  │  ← Does a similar concept already exist?
  │ (similarity ≥ 0.7) │
  └────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
  EXTEND      PROPOSE
    │             │
    ▼             ▼
  Augment    NewConceptProposal
  existing   (requires C4 diagram,
  concept     pillar assignment,
              pipeline phase)
    │             │
    └──────┬──────┘
           │
           ▼
  .specify/design/<feature>/design.md
           │
           ▼
  SDDManager.validate_design()
           │
           ▼
  .specify/specs/<feature>/spec.md
```
