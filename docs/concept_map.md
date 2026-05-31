# agent-utilities Canonical Concept Registry

> **Single Source of Truth** for all CONCEPT: tags in the ecosystem.
> Synthesized to **64 canonical concepts** with gap-free numbering and logically derived naming.
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
| `ORCH-1.0` | Orchestration Engine | 14 | 16 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.1` | HTN Planning Pipeline | 25 | 11 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.2` | Specialist Routing & Discovery | 28 | 14 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.3` | Execution Safety & State | 11 | 2 | [ORCH-1.3](pillars/1_graph_orchestration/ORCH-1.3-Execution_Safety_And_State.md) |
| `ORCH-1.4` | Capability Wiring Engine | 11 | 2 | [ORCH-1.4](pillars/1_graph_orchestration/ORCH-1.4-Capability_Wiring_Engine.md) |
| `ORCH-1.5` | Agent Orchestrator 🔬 | 3 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.6` | DSTDD Pipeline | 3 | 3 | [ORCH-1.6](pillars/1_graph_orchestration/ORCH-1.6-DSTDD_Pipeline.md) |
| `ORCH-1.7` | Prediction Linkage Layer 🔬 | 1 | 0 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.8` | RecursiveMAS Latent Orchestrator 🔬 | 0 | 0 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.25` | Parallel Engine | 5 | 0 | [ORCH-1.25](pillars/1_graph_orchestration/ORCH-1.25-Parallel_Engine.md) |
| `ORCH-1.26` | RLM-Native Hierarchical Synthesis | 2 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.27` | Autonomous Department Orchestration | 3 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.28` | Reactive Event Sourcing | 3 | 1 | [ORCH-1.28](pillars/1_graph_orchestration/ORCH-1.28-Reactive_Event_Sourcing.md) |
| `ORCH-1.29` | WASM Micro-Agent Execution | 1 | 1 | [OS-5.6](pillars/5_agent_os_infrastructure/OS-5.6-Massive_Scale_Architecture.md) |
| `ORCH-1.30` | Structured Predict-RLM Runtime | 2 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |
| `ORCH-1.31` | GEPA Reflective Prompt Optimizer | 2 | 1 | [Pillar Summary](pillars/1_graph_orchestration.md) |

Key modules: `graph/builder.py`, `graph/nodes.py`, `graph/planner.py`, `graph/routing.py`, `graph/executor.py`, `graph/hsm.py`, `graph/lifecycle.py`, `core/default_catalog.py`, `capabilities/checkpointing.py`, `sdd/orchestrator.py`, `graph/kg_graph_factory.py`, `orchestration/agent_runner.py`, `graph/parallel_engine.py`, `graph/manifest_generators.py`, `models/execution_manifest.py`, `graph/reactive/ledger.py`, `graph/reactive/dispatcher.py`, `core/wasm_runner.py`, `rlm/predict_rlm.py`, `rlm/gepa.py`, 🔬 `graph/coordination.py`, 🔬 `orchestration/prediction_linkage.py`

---

## Pillar 2: Epistemic Knowledge Graph (KG)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `KG-2.0` | Active Knowledge Graph | 42 | 24 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.1` | Tiered Memory & Context 🔬 | 17 | 8 | [KG-2.1](pillars/2_epistemic_knowledge_graph/KG-2.1-Tiered_Memory_And_Context.md) |
| `KG-2.2` | Ontology, Epistemics & DSPy Integration | 27 | 6 | [KG-2.2](pillars/2_epistemic_knowledge_graph/KG-2.2-Ontology_And_Epistemics.md) |
| `KG-2.3` | Graph Integrity & Retrieval 🔬 | 14 | 4 | [KG-2.3](pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md) |
| `KG-2.4` | Inductive Knowledge | 12 | 5 | [KG-2.4](pillars/2_epistemic_knowledge_graph/KG-2.4-Inductive_Knowledge_And_Hypergraphs.md) |
| `KG-2.5` | Topological Analysis | 11 | 4 | [KG-2.5](pillars/2_epistemic_knowledge_graph/KG-2.5-Topological_Analysis.md) |
| `KG-2.6` | Domain: Finance | 46 | 31 | [KG-2.6](pillars/2_epistemic_knowledge_graph/KG-2.6-Domain_Finance.md) |
| `KG-2.7` | Research Intelligence | 2 | 0 | [KG-2.7](pillars/2_epistemic_knowledge_graph/KG-2.7-Research_Intelligence.md) |
| `KG-2.8` | Memory Stability | 13 | 2 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.9` | Multi-Domain Architecture | 6 | 2 | [KG-2.9](pillars/2_epistemic_knowledge_graph/KG-2.9-Quant_Orchestration.md) |
| `KG-2.10` | Domain: Enterprise | 4 | 1 | [KG-2.10](pillars/2_epistemic_knowledge_graph/KG-2.10-Observational_Memory_Bridge.md) |
| `KG-2.11` | Vectorized Retrieval | 1 | 0 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.12` | Company Operations Domain | 1 | 1 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.13` | Company Intelligence Graph | 1 | 1 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.14` | Skill-Graph ↔ KG Bidirectional Sync | 2 | 1 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.15` | Centralized Epistemic Gateway & Transaction Proxy | 3 | 3 | [KG-2.15](centralized_kg_coordination.md) |
| `KG-2.16` | Compiled Rust (epistemic-graph) & Rustworkx Compute Engine 🔬 | 1 | 1 | [OS-5.6](pillars/5_agent_os_infrastructure/OS-5.6-Massive_Scale_Architecture.md) |
| `KG-2.17` | Rust-Compiled Epistemic Reasoning Backend 🔬 | 1 | 1 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.18` | High-Performance Quant Epistemic-Graph Engine 🔬 | 1 | 1 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |
| `KG-2.19` | Speculative Graph Brancher 🔬 | 1 | 1 | [KG-2.19](pillars/2_epistemic_knowledge_graph/KG-2.19-Speculative_Graph_Brancher.md) |
| `KG-2.20` | Semantic Compactor & Refactorer 🔬 | 1 | 1 | [KG-2.20](pillars/2_epistemic_knowledge_graph/KG-2.20-Semantic_Compactor_And_Refactorer.md) |
| `KG-3.0` | Ingestion Engine | 1 | 24 | [Pillar Summary](pillars/2_epistemic_knowledge_graph.md) |

Key modules: `knowledge_graph/core/engine.py`, `knowledge_graph/core/engine_memory.py`, `knowledge_graph/core/engine_tasks.py`, `knowledge_graph/core/graph_compute.py`, `knowledge_graph/core/topological_analysis_engine.py`, `knowledge_graph/research/research_intelligence_engine.py`, `knowledge_graph/memory/synthesis.py`, `knowledge_graph/memory/memory_materializer.py`, `knowledge_graph/memory/observer.py`, `knowledge_graph/memory/reflector.py`, `knowledge_graph/memory/startup_context.py`, `knowledge_graph/ontology.ttl`, `knowledge_graph/retrieval/retrieval_quality.py`, `knowledge_graph/pipeline/document_deletion.py`, `knowledge_graph/pipeline/document_update.py`, `domains/finance/`, `knowledge_graph/orchestration/engine_enterprise.py`, `knowledge_graph/pipeline/phases/external_graphs.py`, `knowledge_graph/ingestion/engine.py`, `scripts/install_git_hooks.py`, `scripts/submit_diff.py`, `mcp/kg_server.py`, `mcp/kg_coordinator.py`, `knowledge_graph/backends/ladybug_backend.py`, 🔬 `knowledge_graph/core/ar_graph.py`, 🔬 `knowledge_graph/core/time_series_graph.py`

---

## Pillar 3: Agentic Harness Engineering (AHE)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `AHE-3.0` | Agentic Harness Core | 16 | 3 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.1` | Continuous Evaluation & DSPy Math Optimization | 17 | 7 | [AHE-3.1](pillars/3_agentic_harness_engineering/AHE-3.1-Continuous_Evaluation_Engine.md) |
| `AHE-3.2` | Agentic Evolution Engine | 16 | 5 | [AHE-3.2](pillars/3_agentic_harness_engineering/AHE-3.2-Agentic_Evolution_Engine.md) |
| `AHE-3.3` | Team & Synergy Optimization | 14 | 5 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.4` | Distributed Agentic Evolution | 11 | 1 | [AHE-3.4](pillars/3_agentic_harness_engineering/AHE-3.4-Distributed_Agentic_Evolution.md) |
| `AHE-3.5` | Heavy Thinking & Background Intelligence | 11 | 1 | [AHE-3.5](pillars/3_agentic_harness_engineering/AHE-3.5-Heavy_Thinking_And_Background_Intelligence.md) |
| `AHE-3.6` | Backtest & Curriculum | 10 | 2 | [AHE-3.6](pillars/3_agentic_harness_engineering/AHE-3.6-Backtest_And_Curriculum.md) |
| `AHE-3.7` | KG-Native Task Detection 🔬 | 1 | 0 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.15` | Agent-Interpretable Model Evolver | 2 | 1 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-3.16` | LLM-Graded Interpretability Tests | 2 | 1 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-4.0` | Physical Knowledge Distillation Engine 🔬 | 1 | 1 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-4.1` | Multi-Optimizer Prompt Selection Strategy 🔬 | 1 | 1 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |
| `AHE-4.2` | GitOps Commit & Evolution Boundary Traceability 🔬 | 1 | 1 | [Pillar Summary](pillars/3_agentic_harness_engineering.md) |

Key modules: `harness/evaluation_engine.py`, `harness/agentic_evolution_engine.py`, `graph/team_composer.py`, `agentic_evolution/forge.py`, `knowledge_graph/orchestration/engine_ahe.py`, `knowledge_graph/distillation/physical_distiller.py`, `harness/evolve_agent.py`, 🔬 `harness/distributed_state_manager.py`

---

## Pillar 4: Ecosystem & Peripherals (ECO)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `ECO-4.0` | Tool Interface & MCP Factory | 26 | 18 | [ECO-4.0](pillars/4_ecosystem_peripherals/ECO-4.0-Tool_Interface_And_MCP_Factory.md) |
| `ECO-4.1` | A2A Network & Consensus 🔬 | 7 | 3 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.2` | Community Telemetry & Ecosystem Map | 5 | 1 | [ECO-4.2](pillars/4_ecosystem_peripherals/ECO-4.2-Community_Telemetry_And_Ecosystem_Map.md) |
| `ECO-4.3` | Market Data Connectors | 9 | 3 | [ECO-4.3](pillars/4_ecosystem_peripherals/ECO-4.3-Market_Data_Connectors.md) |
| `ECO-4.4` | KG MCP Server & Execution | 2 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.5` | Native Messaging Backend Abstraction | 21 | 17 | [ECO-4.5](pillars/4_ecosystem_peripherals/ECO-4.5-Native_Messaging_Backend.md) |
| `ECO-4.10` | Agent Toolkit Ingestor | 1 | 1 | [ECO-4.10](pillars/4_ecosystem_peripherals/ECO-4.10-Agent_Toolkit_Ingestor.md) |
| `ECO-4.11` | MCP Live Discovery | 2 | 2 | [ECO-4.11](pillars/4_ecosystem_peripherals/ECO-4.11-MCP_Live_Discovery.md) |
| `ECO-4.12` | Self-Documenting Skill-Graph | 2 | 1 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.13` | Company Infrastructure Orchestration | 3 | 2 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.14` | Infrastructure Blueprint Library | 1 | 1 | [Pillar Summary](pillars/4_ecosystem_peripherals.md) |
| `ECO-4.15` | Pluggable Event Queue Backend | 3 | 2 | [OS-5.6](pillars/5_agent_os_infrastructure/OS-5.6-Massive_Scale_Architecture.md) |
| `ECO-4.16` | Hierarchical AGENTS.md & Team Context | 2 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-hierarchical-agentsmd--team-context-eco-416) |
| `ECO-4.17` | Self-Improving AGENTS.md Reflector | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-self-improving-agentsmd-reflector-eco-417) |
| `ECO-4.18` | Deterministic Lint Enforcement Hook | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-deterministic-lint-enforcement-hook-eco-418) |
| `ECO-4.19` | Plugin Bundle Distribution System | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-plugin-bundle-distribution-system-eco-419) |
| `ECO-4.20` | Permission Policy Engine | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-permission-policy-engine-eco-420) |
| `ECO-4.21` | Configuration Staleness Auditor | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-configuration-staleness-auditor-eco-421) |
| `ECO-4.22` | Governance Workflow Pipeline | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-governance-workflow-pipeline-eco-422) |
| `ECO-4.23` | Codebase Map Generator | 1 | 0 | [Pillar Summary](pillars/4_ecosystem_peripherals.md#-codebase-map-generator-eco-423) |

Key modules: `mcp/server_factory.py`, `mcp/kg_server.py` (incl. `kg_launch_terminal_agent`), `ecosystem/bridge.py`, `ecosystem/hook_installer.py`, `ecosystem/agents_md_reflector.py`, `ecosystem/lint_enforcement_hook.py`, `ecosystem/plugin_bundle.py`, `ecosystem/permission_policy.py`, `ecosystem/config_staleness_auditor.py`, `ecosystem/governance_workflow.py`, `ecosystem/agent_manager_dashboard.py`, `tools/codebase_map_tools.py`, `knowledge_graph/core/agents_md.py`, `knowledge_graph/memory/startup_context.py`, `graph/subagent_patterns.py`, `protocols/a2a_graph_skill.py`, `protocols/data_connector.py`, `tools/tool_filtering.py`, `tools/dynamic_tool_orchestrator.py`, `knowledge_graph/core/engine_ingestion.py`, `knowledge_graph/core/engine_mcp_discovery.py`, `knowledge_graph/core/queue_backend.py`, `knowledge_graph/core/nats_queue_backend.py`, `knowledge_graph/core/kafka_queue_backend.py`

---

## Pillar 5: Agent OS Infrastructure (OS)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|---------------|:---:|:---:|---|
| `OS-5.0` | Agent OS Kernel & XDG Paths | 17 | 6 | [Pillar Summary](pillars/5_agent_os_infrastructure.md) |
| `OS-5.1` | Security & Auth | 19 | 8 | [OS-5.1](pillars/5_agent_os_infrastructure.md#secrets--authentication-conceptos-51) |
| `OS-5.2` | Resource Scheduling 🔬 | 18 | 4 | [Pillar Summary](pillars/5_agent_os_infrastructure.md) |
| `OS-5.3` | Guardrails & Safety | 7 | 3 | [OS-5.3](pillars/5_agent_os_infrastructure.md#-declarative-sensory-guardrails--safety-contracts-conceptos-53) |
| `OS-5.4` | Telemetry & Observability | 7 | 3 | [OS-5.4](pillars/5_agent_os_infrastructure.md#-telemetry-observability--token-usage-conceptos-54) |
| `OS-5.5` | Reactive Budget Guardrails | 2 | 1 | [OS-5.5](pillars/5_agent_os_infrastructure/OS-5.5-Reactive_Budget_Guardrails.md) |
| `OS-5.6` | Massive Scale Architecture & Sandbox | 2 | 2 | [OS-5.6](pillars/5_agent_os_infrastructure/OS-5.6-Massive_Scale_Architecture.md) |
| `OS-5.7` | Distributed Replay & Compliance Engine | 1 | 1 | [OS-5.7](pillars/5_agent_os_infrastructure/OS-5.7-Distributed_Replay_And_Coordination.md) |
| `OS-5.8` | OS-Level Hardened Tool Sandbox Executor | 1 | 1 | [OS-5.8](pillars/5_agent_os_infrastructure/OS-5.8-Hardened_WASM_Executor.md) |
| `OS-5.9` | Epistemic Resource Scheduler 🔬 | 1 | 1 | [OS-5.9](pillars/5_agent_os_infrastructure/OS-5.9-Epistemic_Resource_Scheduler.md) |
| `OS-5.10` | Ontological Guardrail Engine 🔬 | 1 | 1 | [OS-5.10](pillars/5_agent_os_infrastructure/OS-5.10-Ontological_Guardrail_Engine.md) |

Key modules: `core/paths.py`, `security/guardrails.py`, `security/tool_guard.py`, `core/cognitive_scheduler.py`, `observability/token_tracker.py`, `observability/audit_logger.py`, `graph/reactive/budget.py`, `core/wasm_runner.py`, `gateway/aggregator.py`, `gateway/registry.py`, `gateway/config.py`, `gateway/api.py`, `gateway/ws.py`

---

## Gateway Service Dashboard (GW)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|----------------|:------------:|:-----:|----------|
| `GW-1.0` | Gateway Service Dashboard | 58 | 7 | [GW-1.0](pillars/5_agent_os_infrastructure/GW-1.0-Gateway_Service_Dashboard.md) |

Key modules: `gateway/__init__.py`, `gateway/models.py`, `gateway/registry.py`, `gateway/config.py`, `gateway/aggregator.py`, `gateway/api.py`, `gateway/ws.py`, `gateway/widgets/base.py`, `gateway/widgets/*.py` (50 widget modules)

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

## Company Operations Concepts (Phase 9 — AI-First Autonomous Company)

| Concept ID | Name | Pillar | Key Module |
|------------|------|--------|------------|
| `KG-2.12` | Company Operations Domain | KG | `ontology_company.ttl`, `models/company.py` |
| `KG-2.13` | Company Intelligence Graph | KG | `models/company.py` |
| `KG-2.14` | Skill-Graph ↔ KG Bidirectional Sync | KG | `skill-graph-builder` |
| `ORCH-1.26` | RLM-Native Hierarchical Synthesis | ORCH | `graph/parallel_engine.py` |
| `ORCH-1.27` | Autonomous Department Orchestration | ORCH | `graph/manifest_generators.py` |
| `ECO-4.12` | Self-Documenting Skill-Graph | ECO | `skill_graphs/agent-utilities/` |
| `ECO-4.13` | Company Infrastructure Orchestration | ECO | `ontology_company_infra.ttl` |
| `ECO-4.14` | Infrastructure Blueprint Library | ECO | `skill_graphs/infrastructure-blueprints/` |

---

## Pillar 6: GeniusBot Desktop Cockpit (GBOT)

| ID | Canonical Name | Code Modules | Tests | Doc Page |
|----|----------------|:------------:|:-----:|----------|
| `GBOT-6.0` | Desktop Cockpit Orchestrator | `geniusbot/geniusbot.py` | `tests/test_geniusbot.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.1` | Ecosystem Dynamic Tab Matrix | `geniusbot/plugins/` | `tests/test_plugins.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.2` | Embedded Terminal Sandbox | `geniusbot/qt/terminal_widget.py` | `tests/test_terminal_widget.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.3` | Universal Tool Approval Gate | `geniusbot/qt/tool_guard.py` | `tests/test_tool_guard.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.4` | Topological Cockpit Memory | `geniusbot/utils/agent_bridge.py` | `tests/test_agent_bridge.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.5` | Multi-Tenant Daemon & Tray | `geniusbot/utils/daemon.py` | `tests/test_daemon.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
| `GBOT-6.6` | High-Performance Visual Finance Cockpit | `geniusbot/qt/finance_cockpit.py` | `tests/test_finance_cockpit.py` | [GeniusBot Cockpit](pillars/6_geniusbot_cockpit.md) |
