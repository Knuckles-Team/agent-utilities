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

*Version: 0.47.0*

> **Documentation** — The complete guides, architecture references, and pillar
> documentation are published as the
> [official documentation](https://knuckles-team.github.io/agent-utilities/).

## ⚡ 5-Minute Quickstart

`agent-utilities` is a batteries-included harness for building Pydantic-AI agents.
The **zero-infra default needs no databases or external services** — the knowledge
graph runs in-process.

```bash
pip install agent-utilities
```

Point it at any model provider (e.g. `OPENAI_API_KEY`, or a local vLLM/Ollama
endpoint via `.env`), then:

```python
from agent_utilities.agent.factory import create_agent

# A ready-to-run agent: skills, universal tools, and the in-process knowledge graph.
agent, toolsets = create_agent(name="assistant")
result = agent.run_sync("What can you do?")
print(result.output)
```

That's the whole "hello world". From there, attach MCP toolsets, swap the graph
backend, or wire up the full 5-pillar platform — see [Quick Start](#quick-start),
the [reference agent](examples/reference_agent/), and the [pillar docs](docs/pillars).

> **Heads-up — this is two repos.** The heavy graph compute lives in a **separate**
> Rust engine, [`epistemic-graph`](https://github.com/Knuckles-Team/epistemic-graph)
> (reached out-of-process over MessagePack/UDS — **no PyO3**). `agent-utilities`
> ships a pure-Python client for it, so you don't need Rust to get started.
> Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md).

## Table of Contents

- [The Technical Novel: Narrative Journey](docs/journey.md)
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

---

## 🌌 The Journey of Agent Utilities: The Technical Novel

> [!NOTE]
> **New to the project?** Rather than reading dry configuration tables, experience `agent-utilities` live!
> Read our comprehensive technical biography tracing the lifecycle of a high-stakes quantitative rebalancing mandate.
>
> 📖 **[Read the Immersive Narrative Journey (docs/journey.md)](docs/journey.md)**

---

## 🌌 Mission & Future State: Distributed Evolution

The core vision for `agent-utilities` transcends being just an execution harness—it is the bedrock for **Distributed Agentic Evolution** and the substrate for the **AI-First Autonomous Organization**.

As autonomous agents leverage this ecosystem to solve complex problems, they continuously learn, adapt, and refine their own capabilities. Our future state envisions a community of independent, self-improving agents that not only run on this harness but dynamically contribute their localized evolutionary breakthroughs—new skills, optimized TeamConfigs, refined prompts, and advanced reasoning traces—back to the open-source collective.

By tying our unified Knowledge Graph, capability auto-activation, and cross-agent communication protocols together, `agent-utilities` becomes an interconnected hive mind where the evolution of one agent elevates the intelligence of all. The harness is not just a way to run an agent; it is the heartbeat of a distributed, self-evolving intelligence network.

## Key Features

- **[Multi-Domain Expert System](docs/guides/features.md#comprehensive-feature-list)**: Scale across finance, medical, and scientific domains using Temporally-Aware Epistemic Memory (TKG) and specialized MCP tools.
- **[Schema-Pack 2.0 — pluggable domain profiles](docs/pillars/2_epistemic_knowledge_graph/KG-2.37-Research_State_Domain_Pack.md)** (CONCEPT:KG-2.22–KG-2.37): Swap the active domain pack (`research-state`, `finance`, `biomedical`, …) to retune the whole KG — zero-LLM typed-edge extraction (supports/weakens/cites), relational-intent recall, recency/source-trust ranking, autocut, and **transitive/inverse OWL closure** + bi-temporal `as_of` "literature state" that flat brain layers can't provide. Set `GRAPH_SCHEMA_PACK` or `graph_configure(action="schema_pack")`.
- **[Unified Intelligence Graph](docs/guides/features.md#comprehensive-feature-list)**: A tiered pipeline combining native Rust in-memory processing (`EpistemicGraphBackend`, the default L1 working store) with a durable PostgreSQL/pggraph persistence tier and OWL (Apache Jena Fuseki) semantics. (LadybugDB / Neo4j / FalkorDB Cypher backends remain available under `backends/contrib/`.)
- **[Centralized Sessions & Goals (API-First Gateway)](docs/centralized_kg_coordination.md#7-centralized-sessions--autonomous-goal-coordination)**: A highly-resilient, centralized REST API gateway running on Port `8100` that handles background goal loops, durable turns, and user session reply orchestration.
- **[High-Performance Rust Compute Engine](pillars/5_agent_os_infrastructure/OS-5.5-Massive_Scale_Architecture.md) 🔬**: A compiled Rust Graph Compute Engine via `epistemic-graph` running over high-speed Unix Sockets (length-prefixed MessagePack — **no PyO3/FFI**), providing fast AST parsing, VF2 subgraph matching, and a Redpanda-backed Reactive State Ledger. The architecture is **modeled** to scale toward 100,000,000 concurrent agents — each working a *bounded subgraph*, not one monolithic graph — via a [Capacity Model](docs/scaling/capacity_model.md) whose anchors are now **measured** — per-shard linear write throughput and a ~52 kB/agent working-set footprint (`epistemic-graph/docs/benchmarks.md`). The 100M target is a **measured projection** (~78 hosts at 64 GB/host), **not a full-scale load test** — we do not claim 100M has been run.
- **[Spec-Driven Development (SDD)](docs/guides/features.md#spec-driven-development-sdd-lifecycle)**: High-fidelity orchestration that decomposes goals into structured specs, implementation plans, and parallel tasks.
- **[Emergent Architecture](docs/guides/features.md#emergent-architecture-conceptkg-20-through-conceptorch-12)**: Dynamic AgentCapability auto-activation, TeamConfig coalition promotion, and evolutionary skill refinement via self-models.
- **[Agent OS & Safety](docs/guides/features.md#human-in-the-loop--tool-safety)**: Built-in Universal Tool Guards, structural vulnerability scanning, and transparent process lifecycle management.
- **[Dynamic Company Brain Ingestion](docs/guides/features.md#company-intelligence-graph)**: Dynamic data ingestion from external platforms like Jira, GitLab/GitHub, enterprise architecture repositories (e.g., Essential Project, Archi), and databases with automatic ontology alignment and GraphQL/REST extraction.
- **[Document-Source Connector Framework](docs/pillars/4_ecosystem_peripherals/ECO-4.25-Document_Source_Connector_Framework.md)** (CONCEPT:ECO-4.25–4.32): A `load`/`poll`/`slim` connector abstraction (web crawler, filesystem, REST/JSON, **databases — Postgres/MariaDB/MSSQL/Oracle/SQLite/Mongo**, and the entire `agent-packages/agents/*` MCP fleet) with resumable checkpoints, **external permission sync** into the entailment-aware ACL gate, **contextual-retrieval enrichment** (KG-2.50), and query analysis. Ports the Onyx connector surface onto the semantic core — ingested documents become first-class `Document`+`Chunk` ontology objects with OWL semantics, bitemporal slicing, and reified links a flat index can't offer. Includes an explicit **Onyx connector-parity map**.
- **[Media Generation & Transcription](docs/pillars/4_ecosystem_peripherals/ECO-4.30-Media_Generation_Gateway.md)** (CONCEPT:ECO-4.30/4.31): Self-hosted image (`flux.2` + Stable Diffusion 3.5), video (`hunyuanvideo`), speech synthesis (`xtts`), and transcription (`faster-whisper`) exposed as agent tools under the `MEDIA_TOOLS` gate.
- **[Company Brain Runtime (Trust, Permissions, Feedback)](docs/architecture/company_brain_runtime.md)**: The 6-layer "Single Company Brain" wired end-to-end behind `KG_BRAIN_ENFORCE` — **source-authority conflict resolution with trust decay** and **field-level survivorship** (durable per-attribute provenance / MDM golden record), **data-level ACLs + tenant scoping + read audit** on the retrieval path, a **human-correction → durable rule → eval** feedback loop, and **token-budgeted, task-scoped retrieval**.
- **[Vendor-Neutral Enterprise Ontology](docs/architecture/vendor_neutral_enterprise_ontology.md)**: One ArchiMate-aligned upper ontology + crosswalk so ServiceNow↔ERPNext, Camunda↔Archi, etc. are interchangeable — a single query resolves all sources regardless of which vendor tool produced the data.
- **[Enterprise Agent Governance](docs/pillars/4_ecosystem_peripherals.md#-enterprise-agent-governance-eco-416--eco-422)**: Production-grade mutation governance with risk-scored change proposals, human-in-the-loop approval gates, AGENTS.md self-improvement, lint enforcement hooks, plugin bundle distribution, permission policies, staleness auditing, and unified governance workflow pipeline.
- **[Global Workspace Attention & Social-System Swarm](docs/architecture/global_workspace_attention.md)**: After every multi-agent wave the parallel engine scores, selects, and broadcasts winning specialist proposals (read back as runtime standing, with an engine-mismatch telemetry guard) and snapshots [Multi-Agent Social System](docs/architecture/multi_agent_social_system.md) health (archetype heterogeneity, co-evolution, Wasserstein drift) into `ExecutionResult.telemetry`.
- **[In-House Training Substrate](docs/architecture/in_house_training_substrate.md)**: Fine-tune the framework's own open-weight models end-to-end — a deterministic reward/data engine, torch/PEFT SFT/DPO/GRPO trainers (`data-science-mcp[training]`), a pure-Rust loss/optimizer performance path (`epistemic-graph`), checkpoint→reliability-suite eval hooks, and a model-registry role deploy seam that goes live with no hot-path edit. Build-now / run-later on the GB10 (first run: OpenSeeker SFT).
- **[Ontology System — Palantir-Foundry-parity, graph-native](docs/pillars/2_epistemic_knowledge_graph.md#ontology-system-palantir-foundry-parity)** (`agent_utilities/knowledge_graph/ontology/`, `kg.ontology`): A first-class object/link/function/action layer reaching the *same* epistemic backend the rest of the KG uses — **interfaces** (CONCEPT:KG-2.38, abstract shapes + implementer resolution), **value types** (KG-2.39, constrained semantic types compiled to SHACL/OWL), **property types** (KG-2.47, the scalar/geo/vector/struct type vocabulary that drives column DDL + write coercion), **derived properties** (KG-2.40, read-time FUNCTION/CYPHER/SPARQL/EMBEDDING compute), **functions** (KG-2.41, typed/versioned/governed PLAIN/ON_OBJECTS/QUERY runtime), **action types** (KG-2.42, submission-criteria-gated, side-effecting, batchable, revertable), **durable object edits** (KG-2.43, bitemporal edit ledger with revert), **indexing lifecycle** (KG-2.44, content-hashed object funnel + staleness ledger), **object sets** (KG-2.45, composable filter/search/search-around/pivot/aggregate handles), **fine-grained permissioning** (KG-2.46, entailment-aware ACL marking propagation + row-drop enforcement), and **document processing** (KG-2.48, chunk→embed→link) over **reified junction links** (KG-2.26). Unique value-adds vs Foundry: OWL/SHACL-backed interfaces+value-types, embedding/cypher/sparql-backed derived properties, reified many-to-many links, entailment-aware marking propagation, bitemporal edit history, a self-evolving ontology, and the Rust epistemic engine. Exposed over the `ontology_*` MCP tools and the operator **Object Explorer / Object / Vertex** views in the [web UI](docs/pillars/4_ecosystem_peripherals.md) (`/api/enhanced/ontology/*`).

> 📖 **[View the Comprehensive Feature List & Architecture Deep Dives](docs/guides/features.md)**

## 🗺 Concept Map

→ **Full Concept Map**: [docs/concept_map.md](docs/concept_map.md) — canonical concept registry.
→ **Single Source of Truth**: [docs/concepts.yaml](docs/concepts.yaml) — machine-generated registry of every concept marker in code.
→ **Concept Index**: [docs/overview.md](docs/overview.md#concept-index) — all pillars with descriptions and code paths.

<!-- BEGIN GENERATED: concepts -->

Synthesized from concept markers in the codebase into **147 canonical concepts** across **18 pillars**.

> This count and the table below are generated from `docs/concepts.yaml` by `scripts/gen_docs.py`. Do not edit by hand.

| Pillar | ID Range | Count | Focus |
|:------|:---------|:---:|:------|
| **AHE-3** Agentic Harness Engineering | AHE-3.x – AHE-3.17 | 14 | Telemetry-Driven Optimization, Agentic Harness Engineering / Evolution, Adversarial verification passed — no issues found, Optional convergence monitor for multi-loop tasks, Check for matching TeamConfig before LLM planning, Detected mathematical/quantitative topology. Escalate to reasoning model, Distills updated tool description back into Python function docstring, GitOps Git Commit Automation |
| **CTX-1** Context Management | CTX-1.0 | 1 | Nested Subfolder Instructions |
| **ECO-4** Ecosystem & Peripherals | ECO-4.0 – ECO-4.33 | 23 | Register PlannerGraphSkill when graph_bundle is available, Live MCP server connection for tool metadata caching, Company Infrastructure Orchestration, Infrastructure Blueprint Library, Pluggable Event Queue Backend, Team-Specific Startup Context, Deterministic Lint Enforcement Hook, Plugin Bundle Distribution System |
| **EE-033** EE-033 | EE-033 | 1 | closes the priors→weights loop |
| **EE-034** EE-034 | EE-034 | 1 | the expert agent writes one per decision; a nightly distill |
| **EE-036** EE-036 | EE-036 | 1 | concrete subclasses |
| **EE-037** EE-037 | EE-037 | 1 | microstructure, trading, pricing |
| **EE-039** EE-039 | EE-039 | 1 | EE-039 |
| **EG-009** EG-009 | EG-009 | 1 | then by its node |
| **KG-1** Knowledge Graph Core | KG-1.0 | 1 | Centralized KG Coordination Protocol |
| **KG-2** Epistemic Knowledge Graph | KG-2.0 – KG-2.50 | 52 | Provides git-like transactional mutation for KG evolution, routes skill evolution through the single graph-native, the self-bootstrapping ontology agent applied to ingest, Lazy embedding model — defer HTTP connection to first use, Compute positional interaction encoding for structural generalization, /2.15/2.34/2.35 — Topological Analysis Engine, Generates actionable LLM artifacts from KG-ingested research, / KG-2.10 — research assimilation + orchestration synthesis |
| **LGC-1** Logic & Governance Core | LGC-1.0 | 1 | Logic & Governance Core |
| **ORCH-1** Graph Orchestration | ORCH-1.0 – ORCH-1.40 | 32 | Inject signal board observations from prior adaptive_agent_router, Current nesting depth for recursive graph orchestration, Invalidate hot cache so routing reflects new self-knowledge, Visibility allow-list of upstream step ids whose results, Session ID of the parent graph if this state was forked, Dependency cycle detected — falling back, Autonomous Department Orchestration, Graph-Native Reactive Event Sourcing and OS Guardrails |
| **ORCH-2** Orchestration Extensions | ORCH-2.0 | 1 | Orchestration Engine |
| **ORCH-5** Orchestration Runtime | ORCH-5.0 | 1 | / TUI-20 |
| **OS-5** Agent OS Infrastructure | OS-5.0 – OS-5.13 | 13 | FileWatcher — watchdog-triggered graph execution, refactoring. This module re-exports it to avoid breaking, MaintenanceCron — scheduled autonomous maintenance, Reactive Multi-Axis Budget Guardrails, WASM Micro-Agent Sandbox & Runner, Distributed Coordinator with Semantic Sharding, Deterministic Replay Engine, Epistemic dynamic priority & quota scaling based on KG Centrality |
| **SAFE-1** Safety & Guardrails | SAFE-1.0 | 1 | Tool-Agnostic File Safety Hooks |
| **UTIL-1** Shared Utilities | UTIL-1.0 | 1 | Data Type Conversion |

<!-- END GENERATED: concepts -->

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
* **[6. GeniusBot Desktop Cockpit](docs/pillars/6_geniusbot_cockpit.md)**
  * *Contains: Premium Systems Cockpit, swappable plugins tab matrix, sandboxed terminal widget, visual finance trading dashboard.*
* **[C4 Architecture Diagrams](docs/pillars/architecture_c4.md)**
  * *Contains: Ecosystem Dependency Graph, C4 Container Diagram, Cross-Pillar Data Flows.*
* **[Memory Architecture](docs/pillars/memory_architecture.md)**
  * *Contains: Multi-Timescale Memory, Memento Context Management, Observational Memory Bridge.*
* **[Company Brain Runtime](docs/architecture/company_brain_runtime.md)**
  * *Contains: the 6-layer model wired end-to-end — trust/conflict resolution & field-level survivorship, data permissions/tenancy/audit, feedback→rule→eval, retrieval budget, streams, `KG_BRAIN_ENFORCE`.*
* **[Vendor-Neutral Enterprise Ontology](docs/architecture/vendor_neutral_enterprise_ontology.md)**
  * *Contains: the canonical ArchiMate crosswalk, vendor adapters, code→capability realization, and virtual REST federation.*

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
      "base_url": "http://vllm.arpa/v1",
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
      "base_url": "http://vllm-embed.arpa/v1",
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

### Zero-infrastructure by default

Out of the box, agent-utilities runs as a **single self-contained binary with no
external system dependencies**. The default knowledge-graph backend is `tiered` —
the always-included Rust-native `epistemic_graph` (L1) in front of an embedded
**LadybugDB** (L2). No Postgres/Neo4j server is required to get started.

To use a durable PostgreSQL tier in production, just set a DSN — the tiered L2
auto-switches to Postgres (your existing configuration keeps working unchanged):

```bash
export GRAPH_BACKEND=tiered
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5432/agent_kg
```

## Deployment

Full deployment instructions — running `graph-os` and `mcp-multiplexer` as
standard **stdio** or **streamable-http** servers, the centralized REST API
gateway, Docker composes, and production hardening — are in the
**[Deployment Guide](docs/guides/deployment.md)**.

## Quick Start

You can quickly launch the graph-os MCP server (a thin FastMCP wrapper):

```bash
uv run graph-os                       # stdio (default)
uv run graph-os --transport streamable-http --host 0.0.0.0 --port 8004
```

Or start the standalone agent from your code:

```python
from agent_utilities.core.config import config
from agent_utilities.agent.factory import create_agent

# Configuration is automatically loaded from config.json
agent = create_agent(name="MyAgent")
response = agent.run_sync("Analyze the knowledge graph for recent updates.")
print(response.data)
```

For a comprehensive walkthrough, see the [Quick Start Guide](docs/guides/quick-start.md).

## 📚 Guides & Tutorials

For detailed tutorials, installation options, and configuration guides, refer to the `docs/guides/` directory:

* **[Quick Start](docs/guides/quick-start.md)**
* **[Installation Guide](docs/guides/installation.md)**
  * *Bare-metal, pip packages, Docker*
* **[Deployment Guide](docs/guides/deployment.md)**
  * *Zero-infra default, graph-os & multiplexer (stdio/streamable-http), API gateway, production hardening*
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
| [Overview Map](docs/overview.md) | The Concept Galaxy — canonical concepts (see the Concept Map above for the authoritative count), query lifecycle, concept index |
| [Concept Map](docs/concept_map.md) | Canonical concept registry (single source of truth) |
| [C4 Architecture](docs/pillars/architecture_c4.md) | System context, container, and component diagrams |
| [Company Brain Runtime](docs/architecture/company_brain_runtime.md) | The 6-layer brain wired end-to-end: trust/survivorship, permissions, feedback→rule→eval, retrieval budget (`KG_BRAIN_ENFORCE`) |
| [Vendor-Neutral Enterprise Ontology](docs/architecture/vendor_neutral_enterprise_ontology.md) | ArchiMate crosswalk + vendor adapters making ServiceNow↔ERPNext↔Camunda interchangeable |
| [Global Workspace Attention](docs/architecture/global_workspace_attention.md) | GWT loop: score→select→broadcast specialist proposals + `get_attention_score` read-back + engine-mismatch telemetry |
| [Multi-Agent Social System](docs/architecture/multi_agent_social_system.md) | Swarm as `S=(f,g,G)`: archetypes, local observability, co-evolution, P1–P4 swarm health |
| [In-House Training Substrate](docs/architecture/in_house_training_substrate.md) | Cross-repo: reward/data engine → torch/PEFT trainers → Rust kernels → deploy seam (GB10 fine-tunes) |
| [Graph-Native Assimilation Engine](docs/architecture/assimilation_engine.md) | Self-evolution loop: ingest papers/OSS/repos/docs → dedup → gap → synergy → rank → grounded plans; idempotent, runs via `graph_orchestrate(action="assimilate")` + golden-loop daemon |
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
