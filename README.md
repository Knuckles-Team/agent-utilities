# Agent Utilities - AGI Harness

![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/agent-utilities)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/agent-utilities)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/agent-utilities)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/agent-utilities)
![PyPI - License](https://img.shields.io/pypi/l/agent-utilities)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/agent-utilities)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/agent-utilities)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/agent-utilities)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/agent-utilities)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/agent-utilities)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/agent-utilities)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/agent-utilities)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/agent-utilities)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/agent-utilities)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/agent-utilities)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/agent-utilities)

*Version: 0.11.1*

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Intelligence Graph](#-intelligence-graph)
- [First Principles Architecture](#-first-principles-architecture)
- [Concept Map](#-concept-map)
- [Architecture & Orchestration](#architecture--orchestration)
- [Multi-Model Config & Secret Storage](#multi-model-config--secret-storage)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Creating an Agent](#creating-an-agent)
- [Building MCP Servers](#building-mcp-servers--api-wrappers)
- [API Documentation](#api-documentation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🌌 Mission & Future State: Distributed Evolution
The core vision for `agent-utilities` transcends being just an execution harness—it is the bedrock for **Distributed Agentic Evolution**.

As autonomous agents leverage this ecosystem to solve complex problems, they continuously learn, adapt, and refine their own capabilities. Our future state envisions a community of independent, self-improving agents that not only run on this harness but dynamically contribute their localized evolutionary breakthroughs—new skills, optimized TeamConfigs, refined prompts, and advanced reasoning traces—back to the open-source collective.

By tying our unified Knowledge Graph, capability auto-activation, and cross-agent communication protocols together, `agent-utilities` becomes an interconnected hive mind where the evolution of one agent elevates the intelligence of all. The harness is not just a way to run an agent; it is the heartbeat of a distributed, self-evolving intelligence network.

## Key Features

- **[Multi-Domain Expert System](docs/guides/features.md#comprehensive-feature-list)**: Scale across finance, medical, and scientific domains using Vectorized Topological Memory and specialized MCP tools.
- **[Unified Intelligence Graph](docs/guides/features.md#comprehensive-feature-list)**: A topological pipeline combining in-memory NetworkX analysis with persistent Cypher (LadybugDB/Neo4j/Falkor) backends.
- **[Spec-Driven Development (SDD)](docs/guides/features.md#spec-driven-development-sdd-lifecycle)**: High-fidelity orchestration that decomposes goals into structured specs, implementation plans, and parallel tasks.
- **[Emergent Architecture](docs/guides/features.md#emergent-architecture-conceptkg-20-through-conceptorch-12)**: Dynamic AgentCapability auto-activation, TeamConfig coalition promotion, and evolutionary skill refinement via self-models.
- **[Agent OS & Safety](docs/guides/features.md#human-in-the-loop--tool-safety)**: Built-in Universal Tool Guards, structural vulnerability scanning, and transparent process lifecycle management.

> 📖 **[View the Comprehensive Feature List & Architecture Deep Dives](docs/guides/features.md)**

## 🗺 Concept Map

Consolidated from 169 tags into **34 canonical concepts** across **5 Core Pillars**.

→ **Full Concept Map**: [docs/concept_map.md](docs/concept_map.md) — canonical concept registry (single source of truth).
→ **Concept Index**: [docs/overview.md](docs/overview.md#concept-index) — all pillars with descriptions and code paths.

| Pillar | ID Range | Count | Focus |
|:------|:---------|:---:|:------|
| **ORCH-1** Graph Orchestration | ORCH-1.0 – 1.6 | 7 | Intelligence graph, HTN planning, routing, execution safety, DSTDD |
| **KG-2** Knowledge Graph | KG-2.0 – 2.8 | 9 | Active KG, memory, ontology, retrieval, research, finance, enterprise |
| **AHE-3** Agentic Harness | AHE-3.0 – 3.6 | 7 | Evaluation, evolution, teams, heavy thinking, backtest |
| **ECO-4** Ecosystem | ECO-4.0 – 4.5 | 6 | MCP, A2A, telemetry, connectors, KG server, terminal agent launcher |
| **OS-5** Agent OS | OS-5.0 – 5.4 | 5 | Kernel, security, scheduling, guardrails, observability |

## 🏗️ Architecture & Pillar Reference

The detailed architectural diagrams and deep-dive documentation for `agent-utilities` have been moved to their respective Pillar documentation pages in `/docs`.

* **[1. Graph Orchestration & Planning](docs/pillars/1_graph_orchestration.md)**
  * *Contains: First Principles Architecture, SDD Lifecycle, Execution Flow (Dynamic Multi-Layer Parallelism).*
* **[2. Epistemic Knowledge Graph](docs/pillars/2_epistemic_knowledge_graph.md)**
  * *Contains: Graph-OS Native Ingestion Pipeline, MAGMA Reasoning Views, Persistent Task Tracking.*
* **[3. Agentic Harness Engineering](docs/pillars/3_agentic_harness_engineering.md)**
  * *Contains: Self-Models, Evolution, Evaluation.*
* **[4. Ecosystem Peripherals](docs/pillars/4_ecosystem_peripherals.md)**
  * *Contains: graph-os MCP Tools, Server Endpoints, MCP Loading & Registry Architecture.*
* **[5. Agent OS Infrastructure](docs/pillars/5_agent_os_infrastructure.md)**
  * *Contains: Human-in-the-Loop Tool Approval, Process Lifecycle, Auth/Security.*
* **[C4 Architecture Diagrams](docs/pillars/architecture_c4.md)**
  * *Contains: Ecosystem Dependency Graph, C4 Container Diagram, Cross-Pillar Data Flows.*
* **[Memory Architecture](docs/pillars/memory_architecture.md)**
  * *Contains: Multi-Timescale Memory, Memento Context Management, Observational Memory Bridge.*

## External Agent Discovery (mcp_config.json)

Register the Knowledge Graph in your IDE's `mcp_config.json` using the standard CLI pattern:

```json
{
  "mcpServers": {
    "graph-os": {
      "command": "uv",
      "args": ["run", "graph-os"],
      "env": {
        "LITE_LLM_PROVIDER": "openai",
        "LLM_API_KEY": "sk-..."
      }
    }
  }
}
```

> **Note:** Model selection, routing logic, and system configurations are centralized in your XDG `~/.config/agent-utilities/config.json`. Only sensitive tokens like `LLM_API_KEY` remain in the environment.

## 📚 Guides & Tutorials

For detailed tutorials, installation options, and configuration guides, refer to the `docs/guides/` directory:

* **[Quick Start](docs/guides/quick-start.md)**
* **[Installation Guide](docs/guides/installation.md)**
  * *Bare-metal, pip packages, Docker*
* **[Configuration & Environment Variables](docs/guides/configuration.md)**
  * *Multi-tiered LLM setup, Models Config*
* **[Local Secret Storage (Vault & SQLite)](docs/guides/secrets-auth.md)**
* **[Creating an Agent](docs/guides/creating-an-agent.md)**
* **[Building MCP Servers & API Wrappers](docs/guides/building-mcp-servers.md)**
* **[API Documentation & Swagger](docs/guides/development.md)**

## Documentation

Comprehensive system documentation is available in the [`docs/`](docs/) directory:

> **New to the project?** Start with the [**Concept Overview Map**](docs/overview.md) to get oriented.

### Core References

| Guide | Description |
| :--- | :--- |
| [Overview Map](docs/overview.md) | The Concept Galaxy — 34 canonical concepts, query lifecycle, concept index |
| [Concept Map](docs/concept_map.md) | Canonical concept registry (single source of truth) |
| [C4 Architecture](docs/pillars/architecture_c4.md) | System context, container, and component diagrams |
| [Evolution Pipeline](docs/overview.md#evolution-pipeline--super-assimilation-architecture) | Assimilation governance, wire-or-discard heuristic, 4-phase pipeline |

### Pillar Deep-Dives

| Pillar | Guide |
| :--- | :--- |
| Graph Orchestration | [docs/pillars/1_graph_orchestration.md](docs/pillars/1_graph_orchestration.md) |
| Epistemic Knowledge Graph | [docs/pillars/2_epistemic_knowledge_graph.md](docs/pillars/2_epistemic_knowledge_graph.md) |
| Agentic Harness Engineering | [docs/pillars/3_agentic_harness_engineering.md](docs/pillars/3_agentic_harness_engineering.md) |
| Ecosystem & Peripherals | [docs/pillars/4_ecosystem_peripherals.md](docs/pillars/4_ecosystem_peripherals.md) |
| Agent OS Infrastructure | [docs/pillars/5_agent_os_infrastructure.md](docs/pillars/5_agent_os_infrastructure.md) |

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch.
2. **Write tests** for new functionality — all tests must include assertions.
3. **Follow existing patterns** — use the established Pydantic models, structured prompts, and concept markers.
4. **Run the test suite** before submitting: `uv run pytest tests/ -q`.
   > **Note:** All tests are strictly bounded by a 60-second timeout via `pytest-timeout`. Any test that sleeps or hangs indefinitely will fail automatically. Ensure you don't use `time.sleep` without bounds.
5. **Update documentation** in `docs/` if your changes affect public APIs.

See [AGENTS.md](AGENTS.md) for project-specific conventions and architecture rules.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
