---
name: agent-utilities-pillars
skill_type: skill
description: >-
  The 5-pillar architecture reference for agent-utilities.
  Each pillar has its own concept namespace and module area.
tags: [architecture, pillars, orch, kg, ahe, eco, os]
---

# agent-utilities — 5-Pillar Architecture

## Pillar 1: Graph Orchestration Engine (AU-ORCH.planning.orchestration-overview)

**Purpose**: Multi-agent coordination, HTN planning, specialist routing, workflow execution.

| Concept | Name | Key Module |
|---------|------|-----------|
| `ORCH-1.0` | Intelligence Graph Core | `knowledge_graph/core/engine.py` |
| `ORCH-1.1` | HTN Planning Pipeline | `graph/planner.py` |
| `ORCH-1.2` | Specialist Routing & Discovery | `graph/routing.py` |
| `ORCH-1.3` | Execution Safety & State | `graph/executor.py`, `graph/hsm.py` |
| `ORCH-1.4` | Capability Wiring Engine | `graph/nodes.py` |
| `AU-ORCH.planning.legal-automation-roadmap` | Agent Orchestrator | `graph/agent_orchestrator.py` |
| `AU-ORCH.planning.legal-automation-roadmap` | DSTDD Pipeline | `sdd/orchestrator.py` |
| `ORCH-1.8` | Workflow Distillation | `workflows/distillation_hook.py` |
| `AU-ORCH.execution.rlm-synthesis-failed-falling` | Workflow-TeamConfig Unification | `workflows/skill_compiler.py` |
| `AU-ORCH.execution.autonomous-department-orchestration` | Autonomous Department Orchestration | `models/company.py` |

**Architecture**: Tasks flow through the HTN planner → routing selects specialist
agents → executor manages state via HSM → results feed back to KG.

---

## Pillar 2: Epistemic Knowledge Graph (AU-KG.compute.kg-x)

**Purpose**: Semantic storage, OWL reasoning, tiered memory, retrieval, domain knowledge.

| Concept | Name | Key Module |
|---------|------|-----------|
| `KG-2.0` | Active Knowledge Graph | `knowledge_graph/core/engine.py` |
| `KG-2.1` | Tiered Memory & Context | `knowledge_graph/memory/` |
| `KG-2.2` | Ontology & Epistemics | `knowledge_graph/ontology*.ttl` |
| `KG-2.3` | Graph Integrity & Retrieval | `knowledge_graph/retrieval/` |
| `KG-2.4` | Inductive Knowledge | `knowledge_graph/core/ar_graph.py` |
| `KG-2.5` | Topological Analysis | `knowledge_graph/core/topological_analysis_engine.py` |
| `KG-2.6` | Domain: Finance | `domains/finance/` |
| `KG-2.7` | Research Intelligence | `knowledge_graph/research/` |
| `KG-2.12` | Company Operations Domain | `ontology_company.ttl`, `models/company.py` |
| `KG-2.13` | Company Intelligence Graph | `models/company.py` |
| `AU-KG.memory.ground-truth-preamble-declaring` | Skill-Graph ↔ KG Sync | `skill-graph-builder` |

**Architecture**: NetworkX in-memory graph + optional LadybugDB backend.
OWL ontologies loaded via RDFLib/OWLReady2. Tiered memory: episodic →
semantic → procedural with decay/consolidation.

---

## Pillar 3: Agentic Harness Engineering (AU-AHE.optimization.telemetry-optimization)

**Purpose**: Self-improvement, testing, evolution, team composition, curriculum learning.

| Concept | Name | Key Module |
|---------|------|-----------|
| `AHE-3.0` | Agentic Harness Core | `harness/` |
| `AHE-3.1` | Continuous Evaluation Engine | `harness/evaluation_engine.py` |
| `AHE-3.2` | Agentic Evolution Engine | `harness/agentic_evolution_engine.py` |
| `AHE-3.3` | Team & Synergy Optimization | `graph/team_composer.py` |
| `AHE-3.4` | Distributed Agentic Evolution | `harness/distributed_state_manager.py` |
| `AU-AHE.harness.self-evolution-narrative` | Heavy Thinking & Background Intelligence | `agentic_evolution/forge.py` |

**Architecture**: Evaluation engine scores agent outputs → evolution engine
proposes improvements → team composer optimizes coalition membership →
successful teams promoted to TeamConfigNode for reuse.

---

## Pillar 4: Ecosystem & Peripherals (ECO-4.x)

**Purpose**: MCP servers, API clients, messaging, tools, external integrations.

| Concept | Name | Key Module |
|---------|------|-----------|
| `ECO-4.0` | Tool Interface & MCP Factory | `mcp/server_factory.py` |
| `ECO-4.1` | A2A Network & Consensus | `protocols/a2a_graph_skill.py` |
| `AU-ECO.toolkit.journey-map-milestones` | Native Messaging Backend | `ecosystem/bridge.py` |
| `AU-ECO.toolkit.self-documenting-plugin-bundle` | Self-Documenting Skill-Graph | `skill_graphs/agent-utilities/` |
| `AU-OS.deployment.infra-orchestration` | Company Infrastructure Orchestration | `ontology_company_infra.ttl` |
| `AU-OS.deployment.blueprint-library` | Infrastructure Blueprint Library | `skill_graphs/infrastructure-blueprints/` |

**Architecture**: MCP servers expose tools → server_factory creates
standardized endpoints → tool_filtering selects relevant tools for each agent.

---

## Pillar 5: Agent OS Infrastructure (OS-5.x)

**Purpose**: Security, auth, scheduling, guardrails, telemetry, XDG paths.

| Concept | Name | Key Module |
|---------|------|-----------|
| `OS-5.0` | Agent OS Kernel & XDG Paths | `core/paths.py`, `core/config.py` |
| `OS-5.1` | Security & Auth | `security/guardrails.py`, `security/tool_guard.py` |
| `OS-5.2` | Resource Scheduling | `core/cognitive_scheduler.py` |
| `AU-OS.governance.reactive-multi-axis-budget` | Guardrails & Safety | `security/guardrails.py` |
| `AU-OS.governance.wasm-micro-agent-sandbox` | Telemetry & Observability | `observability/token_tracker.py` |

**Architecture**: XDG-compliant paths → config.json at `~/.config/agent-utilities/` →
cognitive scheduler manages LLM load → guardrails enforce safety contracts.
