# Pillar Reference

The 5-pillar architecture (see the table on [the homepage](../index.md)) each has a summary document plus a folder of numbered concept docs (`ORCH-1.x`, `KG-2.x`, `AHE-3.x`, `ECO-4.x`, `OS-5.x`) — one file per concept, cross-referenced to code via `CONCEPT:` markers and to [`docs/concepts.yaml`](../concepts.yaml) (the single source of truth for the concept registry; see [Concept Registry](../concept_map.md)). Start with a pillar's summary page, then drop into its concept docs for implementation detail — parallel in structure to how [architecture/index.md](../architecture/index.md) organizes the cross-cutting subsystem docs.

## Cross-pillar overviews

Whole-system views that cut across all five pillars.

- [C4 Architecture](architecture_c4.md)
- [Ecosystem Integration & Concept Wiring](master_integration.md)
- [Memory Architecture](memory_architecture.md)

## Pillar 1 — Graph Orchestration

Routing, planning, execution, and state management via directed acyclic graphs. Concept docs cover the RLM runtime, GEPA program optimization, workflow lifecycle, and swarm/agent-runner execution.

- [Graph Orchestration Engine](1_graph_orchestration.md)
- [Execution & State Safety](1_graph_orchestration/ORCH-1.3-Execution_Safety_And_State.md)
- [Swarm Preset Template Engine](1_graph_orchestration/ORCH-1.4-Capability_Wiring_Engine.md)
- [DSTDD Pipeline: Design-Spec-Test Driven Development](1_graph_orchestration/ORCH-1.5-DSTDD_Pipeline.md)
- [AU-ORCH.execution.service-registry-initialization: KG-Driven Graph Factory](1_graph_orchestration/ORCH-1.7-KG_Graph_Factory.md)
- [ORCH-1.21: Agent Runner — KG-to-LLM Execution Bridge](1_graph_orchestration/ORCH-1.8-Agent_Runner.md)
- [Parallel Engine](1_graph_orchestration/ORCH-1.8-Parallel_Engine.md)
- [Workflow Distillation & Skill-as-Workflow](1_graph_orchestration/ORCH-1.8-Workflow_Distillation.md)
- [ORCH-1.24: Workflow Lifecycle Management](1_graph_orchestration/ORCH-1.9-Workflow_Lifecycle.md)
- [Reactive Event Sourcing](1_graph_orchestration/ORCH-1.10-Reactive_Event_Sourcing.md)
- [Structured Predict-RLM Runtime + Subagent Contracts](1_graph_orchestration/ORCH-1.12-Structured_RLM_Outputs.md)
- [Native GEPA program optimization](1_graph_orchestration/ORCH-1.13-GEPA_Optimization.md)
- [Role-Specialized Model Routing](1_graph_orchestration/ORCH-1.27-Role_Specialized_Model_Routing.md)
- [Composable Skills + Generic Environment Adapter](1_graph_orchestration/ORCH-1.28-Composable_Skills_And_Generic_Adapter.md)
- [RLM Resilience + Structured Telemetry](1_graph_orchestration/ORCH-1.29-RLM_Resilience_And_Telemetry.md)
- [Held-out generalization for native program optimization](1_graph_orchestration/ORCH-1.30-Generalizing_GEPA.md)
- [Graph-native program optimization state](1_graph_orchestration/ORCH-1.31-Graph_Native_Optimization_State.md)
- [ORCH-1.32 — KG-Governed Agent Swarm](1_graph_orchestration/ORCH-1.32-KG_Governed_Agent_Swarm.md)
- [Tiered RLM Code Sandbox + Capability Router](1_graph_orchestration/ORCH-1.38-Tiered_RLM_Sandbox.md)

## Pillar 2 — Epistemic Knowledge Graph

The Single Company Brain: memory, ontology, retrieval, and structural reasoning. The largest pillar (39 concept docs) — includes the dedicated Company Brain deep-dive subfolder below.

- [Epistemic Knowledge Graph](2_epistemic_knowledge_graph.md)
- [Token-Aware Context Compaction](2_epistemic_knowledge_graph/KG-2.1-Tiered_Memory_And_Context.md)
- [OWL-Driven Semantic Subsumption](2_epistemic_knowledge_graph/KG-2.2-Ontology_And_Epistemics.md)
- [Retrieval Quality Gate](2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md)
- [Cross-Pillar Synergy Engine](2_epistemic_knowledge_graph/KG-2.4-Inductive_Knowledge_And_Hypergraphs.md)
- [Topological Mincut Partitioning](2_epistemic_knowledge_graph/KG-2.5-Topological_Analysis.md)
- [Financial Trading Pipeline](2_epistemic_knowledge_graph/KG-2.6-Domain_Finance.md)
- [Research Intelligence Pipeline](2_epistemic_knowledge_graph/KG-2.7-Research_Intelligence.md)
- [Semantic Compactor & Refactorer](2_epistemic_knowledge_graph/KG-2.7-Semantic_Compactor_And_Refactorer.md)
- [Speculative Graph Brancher](2_epistemic_knowledge_graph/KG-2.7-Speculative_Graph_Brancher.md)
- [KG-2.6: Observational Memory Bridge](2_epistemic_knowledge_graph/KG-2.8-Observational_Memory_Bridge.md)
- [Quant Orchestration](2_epistemic_knowledge_graph/KG-2.8-Quant_Orchestration.md)
- [Bi-Temporal Memory Layers](2_epistemic_knowledge_graph/KG-2.11-Bi_Temporal_Memory_Layers.md)
- [Memory-First Retrieval](2_epistemic_knowledge_graph/KG-2.12-Memory_First_Retrieval.md)
- [Background Learning Engine](2_epistemic_knowledge_graph/KG-2.13-Background_Learning_Engine.md)
- [Ground-Truth Context Authority](2_epistemic_knowledge_graph/KG-2.14-Ground_Truth_Authority.md)
- [Resilient Retrieval](2_epistemic_knowledge_graph/KG-2.15-Resilient_Retrieval.md)
- [Memory Hygiene](2_epistemic_knowledge_graph/KG-2.17-Memory_Hygiene.md)
- [Evidence-Weighted Memory](2_epistemic_knowledge_graph/KG-2.18-Evidence_Weighted_Memory.md)
- [Self-Curating Wiki](2_epistemic_knowledge_graph/KG-2.19-Self_Curating_Wiki.md)
- [Mementified Context Management](2_epistemic_knowledge_graph/KG-2.20-Mementified_Context_Management.md)
- [KG-2.22 — Pack-Driven Retrieval Signals](2_epistemic_knowledge_graph/KG-2.22-Pack_Driven_Retrieval_Signals.md)
- [AU-KG.research.zero-llm-pack-link — Zero-LLM Pack-Driven Link Inference](2_epistemic_knowledge_graph/KG-2.33-Zero_LLM_Link_Inference.md)
- [AU-KG.retrieval.relational-intent-retrieval — Relational-Intent Retrieval](2_epistemic_knowledge_graph/KG-2.34-Relational_Intent_Retrieval.md)
- [AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle and Audit](2_epistemic_knowledge_graph/KG-2.35-Schema_Pack_Lifecycle_And_Audit.md)
- [KG-2.36 — Pack-Driven OWL Closure](2_epistemic_knowledge_graph/KG-2.36-Pack_Driven_OWL_Closure.md)
- [AU-KG.research.research-state-domain-pack — Research-State Domain Pack](2_epistemic_knowledge_graph/KG-2.37-Research_State_Domain_Pack.md)
- [Contextual-Retrieval Enrichment](2_epistemic_knowledge_graph/KG-2.50-Contextual_Retrieval_Enrichment.md)
- [MCP Tool Source Connector](2_epistemic_knowledge_graph/KG-2.59-MCP_Tool_Source_Connector.md)

## Pillar 2 deep-dive — Company Brain

The operational state layer that turns the epistemic graph into a multi-writer, multi-reader, multi-tenant organizational memory — architecture, ontology, permissions, provenance, and concurrency.

- [Company Brain Documentation](2_epistemic_knowledge_graph/company_brain/00_index.md)
- [Company Brain Architecture](2_epistemic_knowledge_graph/company_brain/architecture.md)
- [OWL Ontology](2_epistemic_knowledge_graph/company_brain/ontology.md)
- [Data-Level Permissions](2_epistemic_knowledge_graph/company_brain/permissions.md)
- [Provenance Tracking](2_epistemic_knowledge_graph/company_brain/provenance.md)
- [Concurrency Control](2_epistemic_knowledge_graph/company_brain/concurrency.md)
- [Conflict Resolution](2_epistemic_knowledge_graph/company_brain/conflict_resolution.md)
- [Event Streaming](2_epistemic_knowledge_graph/company_brain/event_streaming.md)
- [Multi-Tenancy](2_epistemic_knowledge_graph/company_brain/multi_tenancy.md)
- [Gap Analysis & Maturity Scorecard](2_epistemic_knowledge_graph/company_brain/gap_analysis.md)
- [Roadmap](2_epistemic_knowledge_graph/company_brain/roadmap.md)

## Pillar 3 — Agentic Harness Engineering

Continuous evaluation, interpretability, and self-improvement loops — the evolution engine, backtesting/curriculum, and distributed agent state.

- [Agentic Harness Engineering](3_agentic_harness_engineering.md)
- [Decomposed Reward Signals](3_agentic_harness_engineering/AHE-3.1-Continuous_Evaluation_Engine.md)
- [Agent Config Versioning](3_agentic_harness_engineering/AHE-3.2-Agentic_Evolution_Engine.md)
- [Distributed Agentic Evolution](3_agentic_harness_engineering/AHE-3.4-Distributed_Agentic_Evolution.md)
- [Heavy Thinking Orchestration](3_agentic_harness_engineering/AHE-3.5-Heavy_Thinking_And_Background_Intelligence.md)
- [Backtest Evaluation Harness](3_agentic_harness_engineering/AHE-3.6-Backtest_And_Curriculum.md)
- [Distributed Agent State Concurrency](3_agentic_harness_engineering/AHE-3.7-Distributed_State_Manager.md)
- [LongMemEval-S Validation Harness](3_agentic_harness_engineering/AHE-3.12-LongMemEval_S_Validation_Harness.md)

## Pillar 4 — Ecosystem & Peripherals

Dynamic capability discovery, MCP servers, the hardened multiplexer, connectors, and governance policy for everything around the core.

- [Ecosystem & Peripherals](4_ecosystem_peripherals.md)
- [Provider Prompt Adaptation](4_ecosystem_peripherals/ECO-4.0-Tool_Interface_And_MCP_Factory.md)
- [Community Telemetry](4_ecosystem_peripherals/ECO-4.2-Community_Telemetry_And_Ecosystem_Map.md)
- [AU-ECO.toolkit.journey-map-milestones — Messaging Configuration Guide](4_ecosystem_peripherals/ECO-4.5-Messaging_Configuration_Guide.md)
- [AU-ECO.toolkit.journey-map-milestones — Native Messaging Backend Abstraction](4_ecosystem_peripherals/ECO-4.5-Native_Messaging_Backend.md)
- [AU-ECO.mcp.toolkit-live-discovery: Unified Agent Toolkit Ingestor](4_ecosystem_peripherals/ECO-4.6-Agent_Toolkit_Ingestor.md)
- [AU-ECO.mcp.toolkit-live-discovery: MCP Live Tool Discovery](4_ecosystem_peripherals/ECO-4.6-MCP_Live_Discovery.md)
- [Autonomous Trading Ecosystem](4_ecosystem_peripherals/ECO-4.9-Autonomous_Trading_Ecosystem.md)
- [Document-Source Connector Framework](4_ecosystem_peripherals/ECO-4.25-Document_Source_Connector_Framework.md)
- [Checkpointed Incremental Poll](4_ecosystem_peripherals/ECO-4.26-Checkpointed_Incremental_Poll.md)
- [Connector Registry + Factory](4_ecosystem_peripherals/ECO-4.27-Connector_Registry_And_Factory.md)
- [External Permission Sync](4_ecosystem_peripherals/ECO-4.28-External_Permission_Sync.md)
- [MCP Agent-Package Connector Adapter](4_ecosystem_peripherals/ECO-4.29-MCP_Agent_Package_Connector_Adapter.md)
- [Media Generation Gateway](4_ecosystem_peripherals/ECO-4.30-Media_Generation_Gateway.md)
- [Media Transcription Bridge](4_ecosystem_peripherals/ECO-4.31-Media_Transcription_Bridge.md)
- [Query Analysis](4_ecosystem_peripherals/ECO-4.32-Query_Analysis.md)
- [Native Database Traversal](4_ecosystem_peripherals/ECO-4.33-Native_Database_Traversal.md)

## Pillar 5 — Agent OS Infrastructure

Kernel, server-minted identity, externalized state, engine sharding, fleet autonomy, observability, and safety sandboxes.

- [Agent OS Infrastructure](5_agent_os_infrastructure.md)
- [Massive Scale Architecture & Sandbox](5_agent_os_infrastructure/OS-5.5-Massive_Scale_Architecture.md)
- [Reactive Budget Guardrails](5_agent_os_infrastructure/OS-5.5-Reactive_Budget_Guardrails.md)
- [OS-5.6 — Distributed Replay, Sandboxing, & Epistemic Resource Scheduling](5_agent_os_infrastructure/OS-5.6-Distributed_Replay_And_Coordination.md)
- [Hardened WASM Sandbox Executor](5_agent_os_infrastructure/OS-5.7-Hardened_WASM_Executor.md)
- [Epistemic Resource Scheduler](5_agent_os_infrastructure/OS-5.8-Epistemic_Resource_Scheduler.md)
- [Gateway Service Dashboard](5_agent_os_infrastructure/OS-5.9-Gateway_Service_Dashboard.md)
- [Ontological Guardrail Engine](5_agent_os_infrastructure/OS-5.9-Ontological_Guardrail_Engine.md)

## Pillar 6 — GeniusBot Cockpit

The desktop cockpit frontend for AI agents.

- [GeniusBot Cockpit](6_geniusbot_cockpit.md)

## Documentation standards for this tree

We employ a strict Concept ID Registry to ensure 1:1:1 traceability between
**Code** (docstrings), **Tests**, and **Documentation**. If you are contributing
a concept doc, follow the standard file naming conventions:

- Pillar summary: `{N}_{pillar_name}.md`
- Concept reference: `{ID}-{Name}.md` (e.g. `KG-2.5-Topological_Analysis.md`)

All new concept proposals go through the DSTDD design phase — see
[`.specify/design/_template.md`](../../.specify/design/_template.md) for the
required KG analysis, and [Status — the Codex](../status.md) for the generated,
always-current concept count (never hand-typed here or anywhere else).
