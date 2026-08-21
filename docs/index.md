# Agent Utilities

agent-utilities is a Python harness for building, orchestrating, and running AI
agents against a shared knowledge graph. Install it, point it at a task, and it
plans, executes, and remembers — backed by epistemic-graph or an in-process
store, your choice. This page gets you running in 5 minutes; the architecture,
concept registry, and full pillar reference are one click away.

## Stand it up

```bash
pip install agent-utilities          # zero external *service* deps to start
```

```bash
setup-config generate --profile tiny     # complete config.json (every option)
graph-os &                                # KG MCP server — no database needed
agent-utilities-doctor                    # one health sweep across every subsystem
```

Or let the installer do all three plus skill wiring in one shot:

```bash
curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh
```

## Choose your path

| Path | Where to go |
|:-----|:-------------|
| 🚀 **Try it in 5 minutes** | [Start Here](start-here.md) / [Quick Start](guides/quick-start.md) |
| 📦 **Deploy it** | [Supported Deployment Configurations (the ladder)](guides/deployment-configurations.md) |
| 🤖 **I'm an AI agent integrating with this repo** | [For AI Agents](for-ai-agents.md) → `AGENTS.md` |
| 🏛️ **Understand the architecture** | [Pillar Reference](pillars/index.md) |

## The 5-Pillar Architecture

The entire ecosystem is organized into five foundational pillars, each handling a distinct layer of organizational intelligence.

| # | Pillar | Summary | Key Capability |
|:-:|:-------|:--------|:---------------|
| **1** | **[Graph Orchestration](pillars/1_graph_orchestration.md)** | Routing, planning, execution, and state management via directed acyclic graphs. | Routes work to the right agent/model |
| **2** | **[Epistemic Knowledge Graph](pillars/2_epistemic_knowledge_graph.md)** | The Single Company Brain: Memory, ontology, retrieval, and structural reasoning. | Maintains organizational state with provenance |
| **3** | **[Agentic Harness](pillars/3_agentic_harness_engineering.md)** | Continuous evaluation, interpretability, and self-improvement loops. | Makes the system smarter over time |
| **4** | **[Ecosystem & Peripherals](pillars/4_ecosystem_peripherals.md)** | Dynamic capability discovery, MCP servers, the hardened multiplexer, connectors, and governance policy. | Connects to external systems securely |
| **5** | **[Agent OS Infrastructure](pillars/5_agent_os_infrastructure.md)** | Kernel, server-minted identity, externalized state, engine sharding, fleet autonomy, Prometheus observability, and safety sandboxes. | Wraps everything in policy and compliance |

→ Full curated reference (grouped by subsystem, not a flat list): **[Architecture Reference](architecture/index.md)** · **[Pillar Reference](pillars/index.md)**.

## Go deeper

- **[Company Brain Architecture](pillars/2_epistemic_knowledge_graph/company_brain/architecture.md)** — the operational state layer deep-dive: how the Epistemic Knowledge Graph becomes a multi-writer, multi-reader, multi-tenant organizational memory.
- **[Read the story](journey.md)** — *The Narrative Journey*, an optional technical novel tracing all 5 pillars through a real end-to-end scenario. Not required reading — a way in if you learn better from a worked example than a reference table.
- **[Status — the Codex](status.md)** — the generated, honesty-first concept/capability registry. Every concept/capability count on this site is computed from here, never hand-typed; if a claim elsewhere disagrees, this page is the one to trust.
- **[Documentation Catalog](reference/documentation-catalog.md)** — every publishable page in this site, generated, including the ones not promoted into the left nav.
- **[The wider ecosystem](ecosystem.md)** — agent-utilities is the entrypoint/harness for the wider `agent-packages` ecosystem: 65 connector packages, three frontends (geniusbot, agent-webui, agent-terminal-ui), a skill library, and ontologies.
