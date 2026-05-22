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

*Version: 0.11.2*

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
        "AGENT_ID": "local-developer",
        "WORKSPACE_PATH": "${workspaceFolder}"
      }
    }
  }
}
```

> **Note:** Model selection, routing logic, and system configurations are centralized in your XDG `~/.config/agent-utilities/config.json`. Only local workspace paths, local agent IDs, or environment overrides remain in the environment.

## Multi-Model Config & Secret Storage

All LLM providers, model registries, safety guardrails, and scheduler policies are managed centrally via the XDG-compliant configuration file at `~/.config/agent-utilities/config.json`.

Every field in the `config.json` has a 1-to-1 environment variable override. The environment variables (detailed in `.env.example`) act as secondary overrides for all settings.

### Centralized `config.json` Template

Here is a fully-populated and production-ready `config.json` file representing the absolute source of truth for the `agent-utilities` Pydantic `AgentConfig` schema:

```json
{
  "default_agent_name": "Agent",
  "agent_description": "AI Agent",
  "agent_system_prompt": null,

  "host": "0.0.0.0",
  "port": 9000,
  "debug": false,
  "enable_web_ui": false,
  "enable_terminal_ui": false,
  "enable_web_logs": true,
  "enable_acp": false,
  "acp_port": 8001,
  "acp_session_root": ".acp-sessions",
  "mcp_config": null,
  "max_upload_size": 10485760,

  "agent_api_key": null,
  "enable_api_auth": false,
  "auth_jwt_jwks_uri": null,
  "auth_jwt_issuer": null,
  "auth_jwt_audience": null,
  "allowed_origins": null,
  "allowed_hosts": null,
  "tool_guard_mode": "strict",
  "sensitive_tool_patterns": [
    ".*delete.*",
    ".*remove.*",
    ".*rm_.*",
    ".*prune.*",
    ".*kill.*",
    ".*exec.*",
    ".*run_command.*"
  ],

  "secrets_backend": "inmemory",
  "secrets_sqlite_path": null,
  "secrets_vault_url": null,
  "secrets_vault_mount": "secret",

  "routing_strategy": "hybrid",
  "graph_persistence_type": "file",
  "graph_persistence_path": "~/.local/share/agent-utilities/graph_state",
  "enable_llm_validation": false,
  "graph_router_timeout": 300.0,
  "graph_verifier_timeout": 300.0,
  "graph_direct_execution": true,
  "min_confidence": 0.4,
  "validation_mode": false,
  "approval_timeout": 0.0,

  "enable_kg_embeddings": true,
  "kg_backups": 3,
  "knowledge_graph_sync_background": true,

  "enable_otel": true,
  "otel_exporter_otlp_endpoint": "http://langfuse.arpa/api/public/otel",
  "otel_exporter_otlp_headers": null,
  "otel_exporter_otlp_public_key": "lf_pk_...",
  "otel_exporter_otlp_secret_key": "lf_sk_...",
  "otel_exporter_otlp_protocol": "http/protobuf",
  "langfuse_host": "http://langfuse.arpa",
  "langfuse_public_key": "lf_pk_...",
  "langfuse_secret_key": "lf_sk_...",
  "langfuse_dataset_capture_threshold": 0.0,

  "a2a_broker": "in-memory",
  "a2a_broker_url": null,
  "a2a_storage": "in-memory",
  "a2a_storage_url": null,
  "a2a_config": null,
  "a2a_refresh_interval": 300,

  "max_tokens": 16384,
  "temperature": 0.7,
  "top_p": 1.0,
  "timeout": 32400.0,
  "tool_timeout": 32400.0,
  "parallel_tool_calls": true,
  "seed": null,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "logit_bias": null,
  "stop_sequences": null,
  "extra_headers": null,
  "extra_body": null,

  "cognitive_scheduler_enabled": true,
  "max_concurrent_agents": 5,
  "agent_token_quota": 100000,
  "preemption_threshold_pct": 0.85,
  "agent_policies_path": null,
  "permissions_signing_key": null,
  "specialist_registry_path": null,
  "homeostatic_downgrade_enabled": true,
  "adversarial_verification": false,
  "maintenance_token_budget": 0,
  "maintenance_priority": "LOW",
  "watchdog_patterns": [
    "pyproject.toml",
    "mcp_config.json",
    "requirements*.txt"
  ],

  "custom_skills_directory": null,
  "skill_types": null,

  "chat_models": [
    {
      "id": "qwen/qwen3.5-9b",
      "provider": "openai",
      "base_url": "http://10.0.0.18:1234/v1",
      "supports_json": false,
      "vision": true,
      "reasoning": true,
      "tools_enabled": true,
      "parallel_instances": 3,
      "context_window": 256000,
      "intelligence_level": "normal",
      "can_route": true,
      "can_kg": true
    }
  ],
  "embedding_models": [
    {
      "id": "text-embedding-nomic-embed-text-v2-moe",
      "provider": "openai",
      "base_url": "http://10.0.0.18:1234/v1",
      "parallel_instances": 4,
      "chunk_size": 768
    }
  ],

  "workspace_path": "/home/apps/workspace",
  "agent_utilities_config_dir": "~/.config/agent-utilities"
}
```

> **Note:** JSON does not support comments. The JSON key names correspond exactly to their uppercase environment variable overrides (e.g. `default_agent_name` → `DEFAULT_AGENT_NAME`).

For comprehensive definitions and capabilities of specific variables, see the [Configuration Guide](docs/guides/configuration.md) and [Local Secret Storage Guide](docs/guides/secrets-auth.md).


## Installation

Install via pip:

```bash
pip install agent-utilities
```

To install with all optional dependencies (including MCP servers, UI, and external graph backends):

```bash
pip install "agent-utilities[all]"
```

For more details, see the [Installation Guide](docs/guides/installation.md).

## Quick Start

You can quickly launch the graph-os MCP server:

```bash
uv run graph-os
```

Or start the standalone agent from your code:

```python
from agent_utilities.core.config import config
from agent_utilities.agent.factory import AgentFactory

# Configuration is automatically loaded from config.json
agent = AgentFactory.create_agent("MyAgent")
response = agent.run_sync("Analyze the knowledge graph for recent updates.")
print(response.data)
```

For a comprehensive walkthrough, see the [Quick Start Guide](docs/guides/quick-start.md).

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
