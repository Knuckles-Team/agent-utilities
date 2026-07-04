---
name: agent-utilities
description: >-
  The agent-utilities framework — a 5-pillar self-evolving AI architecture
  providing Knowledge Graph, Orchestration, Agentic Evolution, Ecosystem
  peripherals, and Agent OS. Install this skill to gain full context about
  how to use, develop for, and extend agent-utilities.
tags: [agent-utilities, knowledge-graph, orchestration, evolution, mcp, owl, pydantic-ai]
concept: AU-ECO.toolkit.self-documenting-plugin-bundle
---

# agent-utilities — Master Skill Reference

**CONCEPT:AU-ECO.toolkit.self-documenting-plugin-bundle — Self-Documenting Skill-Graph**

agent-utilities is a self-evolving AI framework built on 5 pillars. This skill-graph
provides agents (Antigravity, Claude Code, Windsurf, OpenCode) with complete context
for using, developing, and extending the framework.

## 🏗️ Architecture — The 5 Pillars

| Pillar | ID Prefix | Purpose | Key Modules |
|--------|-----------|---------|-------------|
| **Graph Orchestration** | `AU-ORCH.planning.orchestration-overview` | Multi-agent coordination, HTN planning, routing | `graph/`, `orchestration/` |
| **Epistemic Knowledge Graph** | `KG-2.x` | Semantic storage, OWL ontologies, retrieval, memory | `knowledge_graph/` |
| **Agentic Harness Engineering** | `AU-AHE.optimization.telemetry-optimization` | Self-improvement, testing, evolution | `harness/`, `agentic_evolution/` |
| **Ecosystem & Peripherals** | `ECO-4.x` | MCP servers, API clients, messaging, tools | `mcp/`, `tools/`, `ecosystem/` |
| **Agent OS Infrastructure** | `OS-5.x` | Security, scheduling, identity, guardrails | `core/`, `security/`, `observability/` |

## 📍 Quick Navigation

| Need | Skill |
|------|-------|
| **Deploy** agent-utilities (zero-infra default, MCP servers, gateway, prod) | → [deployment/SKILL.md](deployment/SKILL.md) |
| Understand how to **develop** for agent-utilities | → [development/SKILL.md](development/SKILL.md) |
| Use agent-utilities **tools** (X search, KG, workflows) | → [tools/SKILL.md](tools/SKILL.md) |
| Work with **OWL ontologies** | → [ontology/SKILL.md](ontology/SKILL.md) |
| Understand the **5-pillar architecture** | → [pillars/SKILL.md](pillars/SKILL.md) |

## 🧬 Concept ID System

Every feature in agent-utilities has a `CONCEPT:X.Y` tag providing 1:1:1 traceability:

```
CONCEPT:AU-ORCH.adapter.hot-cache-invalidation  →  code: graph/routing.py
                  →  test: tests/unit/graph/test_routing.py
                  →  docs: docs/pillars/1_graph_orchestration/ORCH-1.2.md
```

**40 canonical concepts** across 5 pillars. All new features require DSTDD design phase.
See `docs/concept_map.md` for the complete registry.

## 🔧 Core Dependencies

- **Pydantic AI**: Agent framework with structured output
- **NetworkX**: Graph engine for KG operations
- **RDFLib/OWLReady2**: OWL ontology reasoning
- **LM Studio**: Default local LLM backend (vllm.arpa)

## 🤖 Agent Usage Guide

- When developing FOR agent-utilities, consult [development/SKILL.md](development/SKILL.md)
- When USING agent-utilities tools, consult [tools/SKILL.md](tools/SKILL.md)
- Every code module has a `CONCEPT:X.Y` tag in its docstring — use this for traceability
- All prompts live in `agent_utilities/prompts/*.json` — reference by agent name
- OWL ontologies are in `knowledge_graph/ontology_*.ttl`
- The KG MCP server (`graph-os`) exposes `graph_query`, `graph_search`, `graph_write`,
  `graph_analyze`, `graph_ingest`, and `graph_orchestrate`

## 📦 Package Structure

```
agent_utilities/
├── core/              # OS-5.x: Config, paths, model factory, scheduler
├── graph/             # AU-ORCH.planning.orchestration-overview: Planner, executor, routing, HSM, lifecycle
├── knowledge_graph/   # KG-2.x: Engine, memory, ontologies, KB, retrieval
├── harness/           # AU-AHE.optimization.telemetry-optimization: Evaluation engine, evolution engine
├── mcp/               # ECO-4.x: Server factory, KG server
├── tools/             # ECO-4.x: X search, tool filtering
├── security/          # OS-5.x: Guardrails, tool guard
├── observability/     # OS-5.x: Token tracker, audit logger
├── models/            # Pydantic models: graph, knowledge_graph, company
├── prompts/           # Agent prompt JSON definitions
├── workflows/         # Workflow compiler, skill compiler, distillation
└── docs/              # Pillar docs, concept map, guides
```
