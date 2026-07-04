# Agent Utilities

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

*Version: 1.3.0*

> **New here?** Read **[docs/start-here.md](docs/start-here.md)** — one page: what
> this is, the three ways to use it, and the zero-infra knowledge graph. For AIs,
> **[llms.txt](llms.txt)** is the entry index. Full docs are
> [published here](https://knuckles-team.github.io/agent-utilities/).

> 🤖 **AI agents — this repo is self-deploying.** If you were pointed here to set it
> up: read the **[🚀 Zero-to-deployed](AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator)**
> procedure in `AGENTS.md`, loop **[`genesis.yaml`](genesis.yaml)**, and run the
> `agent-utilities-deployment` skill (or `agent-os-genesis` for enterprise). Ask the
> operator one question — *homelab or enterprise?* — and take it from there.

> 🧰 **Install the skills first — they unlock how to use everything else.** Once
> `agent-utilities` is installed, run **`agent-utilities install-skills`** to drop the
> skill toolkit — including the **`agent-utilities` skill-graph** (the platform's own
> reference manual) plus the deployment/evolution/KG skills — into the calling agent
> tool (Claude Code, etc.) and the agent-utilities XDG skills dir, where agents
> auto-load them. With no flags it installs into every detected tool; `--tool claude`
> or `--path <dir>` targets one. `agent-utilities-doctor` flags it if the toolkit is
> missing.

## 🚀 Deploy in one command

Point yourself (or any agent) at this repo, or just run:

```bash
# macOS / Linux
curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh

# Windows (PowerShell)
irm https://knuckles-team.github.io/agent-utilities/install.ps1 | iex
```

The installer checks your host, installs agent-utilities, drops the skill toolkit
into every AI tool you have (Claude Code, Cursor, Codex, Windsurf, …), wires the
knowledge-graph MCP server, and hands off to a guided deployment. Pick your shape:

| You are… | Profile | What you get |
|---|---|---|
| A homelab / self-hoster | `tiny` | Zero-infra, all-local. No databases, no Docker. |
| One durable server | `single-node-prod` | Postgres/pg-age + the core MCP connector fleet. |
| An enterprise | `enterprise` | Multi-host Swarm, **everything wired** — Vault, SSO, DNS, ingress, observability, all 50+ connectors. |

```bash
curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh -s -- --profile enterprise
```

→ Full procedure: **[Zero-to-deployed](AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator)** · manifest: **[`genesis.yaml`](genesis.yaml)**.

## ⚡ What it is

`agent-utilities` is a **batteries-included harness for building Pydantic-AI
agents** that ship with a knowledge graph, orchestration, memory, and tools. The
**zero-infra default needs no databases or external services** — the knowledge
graph runs in-process. Use it three ways:

| You want to… | Use | Start |
|---|---|---|
| Build a standalone agent in Python | **Library** | `from agent_utilities import create_agent` |
| Give an existing agent (Claude Code/Cursor/yours) the KG + tools | **MCP `graph-os`** | `uvx --from agent-utilities graph-os` |
| Share one KG backend across many clients/containers | **MCP HTTP / REST gateway** | `uvx --from agent-utilities graph-os --transport streamable-http` / `python -m agent_utilities` (REST, default `:9000`) |

→ Full trade-offs: **[Consumption Models](docs/guides/consumption-models.md)**.

### The 30-second mental model

All four surfaces talk to one gateway; the gateway owns one knowledge-graph facade;
the facade fronts **one engine — the authority** (a fast Rust engine that does
compute, cache, semantics, AND durable persistence). Writes fan out to optional
durable mirrors. Everything below the gateway is shared — the surfaces are just
different windows onto the same brain.

```mermaid
flowchart TD
    subgraph Surfaces["Four ways in"]
        WEB[agent-webui] & TUI[agent-terminal-ui] & GB[geniusbot] & IDE[Claude Code / Cursor]
    end
    Surfaces -->|REST / MCP| GW["graph-os MCP + REST gateway<br/>(identity · ActionPolicy · metrics)"]
    GW --> KG["KnowledgeGraph facade<br/>(ontology · routing · memory)"]
    KG --> ENG["epistemic-graph Rust engine — THE authority<br/>compute + cache + semantic + durable (MessagePack/UDS)"]
    ENG -->|async lossless fan-out| MIR["optional mirrors<br/>Postgres / pg-age · Neo4j · FalkorDB · LadybugDB"]
    KG --> OWL["OWL / SHACL · Fuseki<br/>semantics (local SPARQL over the engine)"]
    GW -. fleet events / autonomy .-> FLEET["reconciler · playbooks · autoscaler"]
```

## ⚡ 5-Minute Quickstart

```bash
pip install agent-utilities          # zero external *service* deps to start
```

Point it at any model provider (`OPENAI_API_KEY`, or a local vLLM/Ollama endpoint
via `.env`), then create an agent — skills, tools, and the in-process KG included:

```python
from agent_utilities import create_agent

agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("What can you do?").output)
```

### Stand it up and verify — 3 commands, zero infra

```bash
setup-config generate --profile tiny     # complete config.json (every option)
graph-os &                                # KG MCP server — no database needed
agent-utilities-doctor                    # one health sweep across every subsystem
```

Scale up (`--profile single-node-prod`/`enterprise`), add Stardog + pg-age, or let
Claude set itself up — all in the **[Quick Start](docs/guides/quick-start.md)** and
**[Self-Setup](docs/guides/self-setup.md)** guides.

### Use the knowledge graph natively — for free, no database

```python
from agent_utilities.mcp import kg_server   # GRAPH_BACKEND=epistemic_graph is the default

# Add knowledge...
await kg_server._execute_tool("graph_write", action="add_node",
    node_id="svc:payments", node_type="Service",
    properties='{"team":"fintech","tier":"critical"}')

# ...and query it back — in-process, no server required.
res = await kg_server._execute_tool("graph_query",
    cypher="MATCH (n:Service) WHERE n.tier='critical' RETURN n")
```

→ The full capability catalog (search, ingest, orchestrate, ontology, memory) is
in **[docs/capabilities.md](docs/capabilities.md)**; runnable code is in the
[reference agent](examples/reference_agent/).

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

## 🧭 Roadmap & Vision

> This section is **aspirational direction**, not shipped behavior — it's here so
> you know where the project is heading. For what works *today*, see
> [Capabilities](docs/capabilities.md).

The direction beyond a single agent harness is **distributed agentic evolution**:
agents that learn from their own failures (the [harness-engineering pillar](docs/pillars/3_agentic_harness_engineering.md)
ships the evolution loop today) and, over time, contribute reusable breakthroughs
— new skills, TeamConfigs, refined prompts — back to a shared knowledge graph so
improvements compound across agents. The building blocks that exist now (unified
KG, capability auto-activation, cross-agent protocols) are the substrate; the
"agents improving each other at scale" end-state is roadmap.

Designed-but-not-yet-running roadmap items (designs/specs exist; do not expect
these to work out of the box today):

- **[Media Generation & Transcription](docs/pillars/4_ecosystem_peripherals/ECO-4.30-Media_Generation_Gateway.md)** (CONCEPT:ECO-4.30/4.31): Self-hosted image (`flux.2` + Stable Diffusion 3.5), video (`hunyuanvideo`), speech synthesis (`xtts`), and transcription (`faster-whisper`) exposed as agent tools under the `MEDIA_TOOLS` gate — requires the corresponding self-hosted model services to be deployed.
- **[In-House Training Substrate](docs/architecture/in_house_training_substrate.md)**: Fine-tune the framework's own open-weight models end-to-end — a deterministic reward/data engine, torch/PEFT SFT/DPO/GRPO trainers (`data-science-mcp[training]`), a pure-Rust loss/optimizer performance path (`epistemic-graph`), checkpoint→reliability-suite eval hooks, and a model-registry role deploy seam. Build-now / run-later on the GB10 (first run: OpenSeeker SFT).

## Key Features

Grouped by what they do. Each line links to the deep-dive; the full catalog with
every concept ID is in **[docs/guides/features.md](docs/guides/features.md)**.

**🧠 Knowledge graph & memory** — the zero-infra core.
**One engine — the authority** (native Rust: compute + cache + OWL semantics +
durable persistence; writes fan out to optional Postgres/pg-age, Neo4j, FalkorDB,
LadybugDB mirrors) with **[Schema-Pack domain profiles](docs/pillars/2_epistemic_knowledge_graph/KG-2.37-Research_State_Domain_Pack.md)**
(KG-2.22–2.37: zero-LLM typed-edge extraction, transitive/inverse OWL closure,
bitemporal `as_of` recall) over a **[high-performance Rust compute engine](docs/pillars/5_agent_os_infrastructure/OS-5.5-Massive_Scale_Architecture.md)**
(MessagePack/UDS, **no PyO3**; measured ~52 kB/agent — see the
[capacity model](docs/scaling/capacity_model.md) for the honestly-projected 100M figure).

**🗂 Ontology system (Palantir-Foundry parity, graph-native)** — the structured layer.
**[Objects, links, interfaces, value/property types, derived properties, functions,
action types, durable edits, object sets & fine-grained permissioning](docs/pillars/2_epistemic_knowledge_graph.md#ontology-system-palantir-foundry-parity)**
(KG-2.26, 2.38–2.48) — OWL/SHACL-backed, reified many-to-many links, bitemporal edit
history, exposed over `ontology_*` MCP tools and the web-UI Object Explorer.
A **[vendor-neutral ArchiMate upper ontology](docs/architecture/vendor_neutral_enterprise_ontology.md)**
makes ServiceNow↔ERPNext↔Camunda interchangeable in one query.

**🔀 Orchestration & self-evolution** — how work gets planned and improved.
**[Spec-Driven Development](docs/guides/features.md#spec-driven-development-sdd-lifecycle)**,
**[emergent architecture](docs/guides/features.md#emergent-architecture-conceptkg-20-through-conceptorch-12)**
(capability auto-activation, TeamConfig coalitions),
**[global-workspace attention](docs/architecture/global_workspace_attention.md)** over
multi-agent waves, **[ontology-to-workflow execution](docs/pillars/1_graph_orchestration.md)**
(KG-2.52/53, ORCH-1.41–43: lift a descriptive process into an executable plan), and
**[governed evolution-to-branch publication](docs/guides/autonomous-evolution.md)**
(AHE-3.18–21: propose-only, governance + regression gated, **never auto-pushed**).

**🏢 Enterprise integration (Company Brain)** — getting your systems in.
A **[document-source connector framework](docs/pillars/4_ecosystem_peripherals/ECO-4.25-Document_Source_Connector_Framework.md)**
(ECO-4.25–4.32, KG-2.59) — native Postgres/filesystem/REST/web-crawl, with **every
other system riding the ~58-server MCP fleet** via the universal `mcp_tool` source —
feeding the **[6-layer Company Brain runtime](docs/architecture/company_brain_runtime.md)**
(`KG_BRAIN_ENFORCE`: trust-decay conflict resolution, field-level survivorship,
data ACLs + tenant scoping, human-correction→rule→eval feedback).

**📈 Scale-out planes (all opt-in; default stays zero-infra)** — how it grows.
**[Externalized durable state](docs/architecture/state_externalization.md)** (one
`STATE_DB_URI`, OS-5.16–18), **[tenant-sharded engines](docs/architecture/engine_sharding.md)**
(HRW routing, KG-2.58/OS-5.28), **[Kafka ingest scale-out](docs/architecture/event_backbone_architecture.md)**
(KG-2.55–57, fail-loud), **[queue-driven agent dispatch](docs/architecture/agent_dispatch.md)**
(ORCH-1.45), and **[gateway scaling + Prometheus `/metrics`](docs/architecture/gateway_scaling.md)**
(OS-5.23, per-tenant rate limits, circuit breakers, `GATEWAY_WORKERS`).

**⚡ Inference & numeric acceleration (all opt-in)** — reuse compute instead of recomputing it.
**[KV-cache layering (vLLM → LMCache → epistemic-graph)](docs/guides/kvcache-vllm-lmcache.md)** — pool
and dedup vLLM's KV cache into the engine so inference workers share prefill by token-hash: the
**universal `LMCacheMPConnector`** (dense **and** hybrid Mamba/GDN) over an **L1 CPU + L2 engine** tier
stack (`EpistemicGraphL2Connector` KG-2.311 / `EpistemicGraphKVBackend` KG-2.306 → engine EG-185/186/187),
steered per-execution by a **[dynamic KV-layering policy](docs/architecture/kv-cache-layering-policy.md)**
(ORCH-1.105: cache-worthiness scoring). Plus the numeric **`xp` numpy-shim** (`agent_utilities/numeric/`, KG-2.312) — a numpy-compatible namespace
that routes reductions/linalg/random through the BLAS/LAPACK-free epistemic-graph numeric kernel (Surface A
of the engine's [Analytics Program](https://knuckles-team.github.io/epistemic-graph/architecture/numeric-kernel/),
EG-321) when the compiled kernel is present and **transparently falls back to numpy** when it is not.

**🛡 Autonomy & governance** — how it acts safely.
A **[fleet-autonomy control plane](docs/architecture/fleet_autonomy.md)** (OS-5.15,
5.24–27, 5.29: `POST /api/fleet/events` → fail-closed **ActionPolicy** gate → reconciler,
remediation playbooks, health-gated deploy-watch + rollback, reactive autoscaler),
**[server-minted identity & fail-closed permissioning](docs/architecture/gateway_scaling.md)**
(OS-5.14, JWT `ActorContext`, HMAC engine auth), enterprise mutation governance, and a
**[hardened MCP multiplexer](docs/pillars/4_ecosystem_peripherals.md)** (ECO-4.34:
per-child limits, circuit breakers, restart-on-crash).

Shipped but lightly documented (real code, importable today):

- **Causal reasoning**: structural-causal-model types, d-separation, and formal reasoning over KG subgraphs — `agent_utilities/knowledge_graph/core/formal_reasoning_core.py`.
- **Skill compiler** (CONCEPT:ORCH-1.8): compiles `SKILL.md` prose (+ optional `references/team.yaml`) into executable `GraphPlan` workflows — `agent_utilities/workflows/skill_compiler.py`.
- **Evolutionary memory & aggregation** (CONCEPT:KG-2.1): the self-curating CRUD insight/skill memory banks (`agent_utilities/harness/evolving_memory.py`) plus the global-workspace score→select→broadcast aggregation over multi-agent waves (`agent_utilities/graph/workspace_attention.py`).
- **KG auto-routing**: the strategy-based router (`agent_utilities/graph/routing/` — fast-path, workflow-context, and policy strategies) backed by capability designation + reward write-back (`agent_utilities/knowledge_graph/retrieval/capability_index.py`).
- **Reactive framework** (CONCEPT:ORCH-1.10): graph-native event sourcing, dynamic behavioral dispatch, and multi-axis budget guardrails — `agent_utilities/graph/reactive/`.

> 📖 **[View the Comprehensive Feature List & Architecture Deep Dives](docs/guides/features.md)**

## 🗺 Concept Map

→ **Full Concept Map**: [docs/concept_map.md](docs/concept_map.md) — canonical concept registry.
→ **Single Source of Truth**: [docs/concepts.yaml](docs/concepts.yaml) — machine-generated registry of every concept marker in code.
→ **Concept Index**: [docs/overview.md](docs/overview.md#concept-index) — all pillars with descriptions and code paths.

<!-- BEGIN GENERATED: concepts -->

Synthesized from concept markers in the codebase into **607 canonical concepts** across **28 pillars**.

> This count and the table below are generated from `docs/concepts.yaml` by `scripts/gen_docs.py`. Do not edit by hand.

| Pillar | ID Range | Count | Focus |
|:------|:---------|:---:|:------|
| **AHE-3** Agentic Harness Engineering | AHE-3.x – AHE-3.74 | 70 | Telemetry-Driven Optimization, Agentic Harness Engineering / Evolution, / AHE-3.40 — generalized from the original system-prompt-only, Optional convergence monitor for multi-loop tasks, Check for matching TeamConfig before LLM planning, Detected mathematical/quantitative topology. Escalate to reasoning model, Distills updated tool description back into Python function docstring, GitOps Git Commit Automation |
| **CE-038** CE-038 | CE-038 | 1 | periodic code-health |
| **CTX-1** Context Management | CTX-1.0 | 1 | Nested Subfolder Instructions |
| **ECO-4** Ecosystem & Peripherals | ECO-4.0 – ECO-4.99 | 83 | / Universal-capability — the orchestrator emits ONE Markdown answer; each, Live MCP server connection for tool metadata caching, Company Infrastructure Orchestration, Infrastructure Blueprint Library, Pluggable Event Queue Backend, Team-Specific Startup Context, Deterministic Lint Enforcement Hook, Plugin Bundle Distribution System |
| **EE-033** EE-033 | EE-033 | 1 | closes the priors→weights loop |
| **EE-034** EE-034 | EE-034 | 1 | the expert agent writes one per decision; a nightly distill |
| **EE-036** EE-036 | EE-036 | 1 | concrete subclasses |
| **EE-037** EE-037 | EE-037 | 1 | microstructure, trading, pricing |
| **EE-039** EE-039 | EE-039 | 1 | EE-039 |
| **EG-009** EG-009 | EG-009 | 1 | then by its node |
| **EG-010** EG-010 | EG-010 | 1 | ORCH-1.73 — this module is now PURELY STRUCTURAL. The old hardcoded |
| **EG-037** EG-037 | EG-037 | 1 | This engine |
| **EG-160** EG-160 | EG-160 | 1 | EG-160 |
| **EG-185** EG-185 | EG-185 | 1 | so the wrapper logs that L2 delete is |
| **EG-186** EG-186 | EG-186 | 1 | over a small HTTP surface |
| **EG-187** EG-187 | EG-187 | 1 | is configured with, so a co-located deploy shares one source of |
| **EG-321** EG-321 | EG-321 | 1 | EG-321 |
| **EG-346** EG-346 | EG-346 | 1 | is installed, the kernel-discovery loop below finds |
| **KG-1** Knowledge Graph Core | KG-1.0 | 1 | Centralized KG Coordination Protocol |
| **KG-2** Epistemic Knowledge Graph | KG-2.0 – KG-2.323 | 261 | / KG-2.106 — code AST parsing delegated to the epistemic-graph engine, routes skill evolution through the single graph-native, the self-bootstrapping ontology agent applied to ingest, Lazy embedding model — defer HTTP connection to first use, Compute positional interaction encoding for structural generalization, /2.15/2.34/2.35 — Topological Analysis Engine, Lazy symbol loading. The finance domain pulls heavy optional, / KG-2.10 / KG-2.78 — research assimilation + orchestration synthesis |
| **LGC-1** Logic & Governance Core | LGC-1.0 | 1 | Logic & Governance Core |
| **ML-011** ML-011 | ML-011 | 1 | join inference |
| **ORCH-1** Graph Orchestration | ORCH-1.0 – ORCH-1.105 | 94 | Inject signal board observations from prior adaptive_agent_router, Current nesting depth for recursive graph orchestration, Invalidate hot cache so routing reflects new self-knowledge, Visibility allow-list of upstream step ids whose results, Session ID of the parent graph if this state was forked, Dependency cycle detected — falling back, Autonomous Department Orchestration, Graph-Native Reactive Event Sourcing and OS Guardrails |
| **ORCH-2** Orchestration Extensions | ORCH-2.0 | 1 | Orchestration Engine |
| **ORCH-5** Orchestration Runtime | ORCH-5.0 | 1 | Durable session and autonomous goal persistence with iterative background goal loops |
| **OS-5** Agent OS Infrastructure | OS-5.0 – OS-5.77 | 68 | FileWatcher — watchdog-triggered graph execution, refactoring. This module re-exports it to avoid breaking, MaintenanceCron — scheduled autonomous maintenance, Reactive Multi-Axis Budget Guardrails, WASM Micro-Agent Sandbox & Runner, Distributed Coordinator with Semantic Sharding, Deterministic Replay Engine, Epistemic dynamic priority & quota scaling based on KG Centrality |
| **SAFE-1** Safety & Guardrails | SAFE-1.0 – SAFE-1.8 | 9 | Tool-Agnostic File Safety Hooks, non-saturating superhuman progress tracking via relative scorers and a saturation detector that keep producing signal past the human or known-answer ceiling so a genuine capability jump is distinguishable from metric saturation, a multi-agent scaling-law harness that sweeps collective size over a fixed task and fits capability ~ N^alpha so the platform can measure whether adding agents helps super- or sub-linearly instead of assuming it does, recursive-improvement velocity tracker that surfaces whether the loop is still improving and flags a non-positive derivative as a research-gets-harder signal, a model-collapse guard for the self-generated training corpus that rejects near-duplicate or distributionally-narrowing rows and caps the synthetic-to-human fraction so recursive distillation cannot quietly degenerate, objective-level safety primitives for rising autonomy, Unattended-session stop-on-ask containment |
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
* **[Multi-Tenant graph-os over Streamable-HTTP](docs/architecture/multi_tenant_streamable_http.md)**
  * *Contains: hierarchical org→user isolation, private-by-default + commons/markings sharing, the five isolation layers (identity → named-graph → scope/visibility → Postgres RLS → audit), tenant-scoped fleet, and the elastic per-tenant engine pool.*

## External Agent Discovery (mcp_config.json)

Register the platform in your IDE's `mcp_config.json` using the standard CLI pattern.
**Generate it with `setup-config mcp` (doctor-driven) — don't hand-write it.** Pick
which server to keep: **`graph-os`** exposes *just the Knowledge Graph* (the `go__*`
tools of one KG backend), while **`mcp-multiplexer`** fronts *the whole fleet* — it
runs in dynamic mode and loads graph-os plus every `*-mcp` server on demand.

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
    },
    "mcp-multiplexer": {
      "command": "uv",
      "args": ["run", "mcp-multiplexer"],
      "env": {
        "AGENT_ID": "local-developer",
        "WORKSPACE_PATH": "${workspaceFolder}",
        "MCP_MULTIPLEXER_MODE": "dynamic",
        "MCP_CONFIG": "${workspaceFolder}/mcp_config.json"
      }
    }
  }
}
```

> **Note:** Model selection, routing logic, and system configurations are centralized in your XDG `~/.config/agent-utilities/config.json`. Only local workspace paths, local agent IDs, or environment overrides remain in the environment.

## Multi-Model Config & Secret Storage

All LLM providers, model registries, safety guardrails, and scheduler policies are managed centrally via the XDG-compliant configuration file at `~/.config/agent-utilities/config.json`.

Every field in the `config.json` has a 1-to-1 environment variable override. The environment variables (detailed in `.env.example`) act as secondary overrides for all settings.

### Minimal `config.json`

You only need to declare your model providers; every other field has a sensible
default. A minimal working config:

```json
{
  "chat_models": [
    {"id": "qwen/qwen3.6-27b", "provider": "openai", "base_url": "http://vllm.arpa/v1",
     "tools_enabled": true, "can_route": true, "can_kg": true}
  ],
  "embedding_models": [
    {"id": "nomic-embed-text-v2", "provider": "openai", "base_url": "http://vllm-embed.arpa/v1"}
  ]
}
```

> Every `config.json` key maps 1-to-1 to an uppercase environment-variable override
> (`default_agent_name` → `DEFAULT_AGENT_NAME`). JSON has no comments — keep notes in
> the guides. The **fully-populated production template** (auth, secrets, routing,
> scheduler, OTEL/Langfuse, A2A, sampling) lives in
> [docs/examples/config.json](docs/examples/config.json).

For comprehensive definitions and capabilities of specific variables, see the [Configuration Guide](docs/guides/configuration.md) and [Local Secret Storage Guide](docs/guides/secrets-auth.md). The authoritative **per-flag inventory and audit** (every `KG_*`/`GRAPH_*`/`EPISTEMIC_*` flag, its default, and whether it should exist at all) is [docs/architecture/configuration.md](docs/architecture/configuration.md).


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

Out of the box, agent-utilities runs as a **single self-contained install with no
external *service* dependencies** (no database or graph server to stand up;
Python package dependencies still apply). The default knowledge-graph backend is
`epistemic_graph` — the always-included Rust-native engine that is **the one
authority** (compute + cache + semantic + durable persistence). No Postgres/Neo4j
server is required to get started.

To add a durable PostgreSQL **mirror** in production (for interop/BI/DR), turn on
fan-out and name the mirror — the engine stays the read authority and Postgres
receives the replicated write stream:

```bash
export GRAPH_BACKEND=fanout
export GRAPH_MIRROR_TARGETS='["pg-age"]'
export KG_CONNECTIONS='[{"name":"pg-age","backend":"age","uri":"postgresql://agent:agent@localhost:5432/agent_kg"}]'
```

## Deployment

Full deployment instructions — running `graph-os` and `mcp-multiplexer` as
standard **stdio** or **streamable-http** servers, the centralized REST API
gateway, Docker composes, and production hardening — are in the
**[Deployment Guide](docs/guides/deployment.md)**. The flagship
**[Deployment Configurations](docs/guides/deployment-configurations.md)** guide
walks every shape from the zero-infra laptop default to a sharded,
queue-driven, policy-governed fleet (`STATE_DB_URI`, `GRAPH_SERVICE_ENDPOINTS`,
`TASK_QUEUE_BACKEND`, `AGENT_DISPATCH_BACKEND`, `GATEWAY_WORKERS`).

> **Already deployed and want to turn the enterprise/autonomy capabilities on?**
> They ship off-by-default so the laptop experience stays zero-infra. The
> **[Enterprise Enablement Runbook](docs/guides/enterprise-enablement-runbook.md)**
> is the ordered push → deploy → flag-enablement sequence (security → state → sharding
> → brain → autonomy), each stage independently reversible and verified.

> **Serving thousands of tenants over streamable-HTTP?** The
> **[Multi-Tenant graph-os](docs/architecture/multi_tenant_streamable_http.md)**
> architecture covers hierarchical org→user isolation, private-by-default sharing
> with explicit commons/markings promotion, full tenant-stamped audit, and the
> elastic per-tenant engine pool — with ready-to-edit k8s and Swarm manifests in
> [`deploy/`](deploy/README.md).

## Quick Start

You can quickly launch the graph-os MCP server (a thin FastMCP wrapper):

```bash
uvx --from agent-utilities graph-os                       # stdio (default)
uvx --from agent-utilities graph-os --transport streamable-http --host 0.0.0.0 --port 8004
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
  * *Multi-tiered LLM setup, Models Config; the per-flag audit lives in [docs/architecture/configuration.md](docs/architecture/configuration.md)*
* **[Local Secret Storage (Vault & SQLite)](docs/guides/secrets-auth.md)**
* **[Creating an Agent](docs/guides/creating-an-agent.md)**
* **[Building MCP Servers & API Wrappers](docs/guides/building-mcp-servers.md)**
* **[API Documentation & Swagger](docs/guides/development.md)**

## 🌌 The Technical Novel

> [!NOTE]
> Prefer a story to config tables? **[The Immersive Narrative Journey (docs/journey.md)](docs/journey.md)**
> traces `agent-utilities` live through the lifecycle of a high-stakes quantitative
> rebalancing mandate — a guided tour of the whole platform in motion.

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
| [In-House Training Substrate](docs/architecture/in_house_training_substrate.md) | **Roadmap** — cross-repo design: reward/data engine → torch/PEFT trainers → Rust kernels → deploy seam (GB10 fine-tunes) |
| [Graph-Native Assimilation Engine](docs/architecture/assimilation_engine.md) | Self-evolution loop: ingest papers/OSS/repos/docs → dedup → gap → synergy → rank → grounded plans; idempotent, runs via `graph_orchestrate(action="assimilate")` + golden-loop daemon |
| [Evolution Pipeline](docs/overview.md#evolution-pipeline--super-assimilation-architecture) | Assimilation governance, wire-or-discard heuristic, 4-phase pipeline |
| [State Externalization](docs/architecture/state_externalization.md) | `STATE_DB_URI` shared Postgres state, SKIP LOCKED queue claims, advisory-lock daemon leadership, fleet pagination (OS-5.16–5.18, KG-2.54) |
| [Engine Sharding](docs/architecture/engine_sharding.md) | Tenant-partitioned engine shards behind client-side HRW routing + topology visibility (KG-2.58, OS-5.28) |
| [Event Backbone](docs/architecture/event_backbone_architecture.md) | Kafka event backbone + ingest task-queue scale-out: fail-loud selection, keyed partitions, `kg-ingest` consumer group (KG-2.55–2.57) |
| [Agent Dispatch](docs/architecture/agent_dispatch.md) | Queue-driven agent dispatch: session-keyed `agent_turns` queue + stateless worker fleet (ORCH-1.45) |
| [Fleet Autonomy](docs/architecture/fleet_autonomy.md) | ActionPolicy decision point, fleet reconciler, remediation playbooks, deploy watch, autoscaler (OS-5.24–5.27, OS-5.29) |
| [Gateway Scaling](docs/architecture/gateway_scaling.md) | `GATEWAY_WORKERS` pre-fork, per-tenant rate limiting, engine circuit breaker, Prometheus `/metrics` (OS-5.23) |
| [Autonomous Evolution](docs/guides/autonomous-evolution.md) | The governed self-evolution chain: propose-only loops → governance validator → regression gate → policy-gated branch publication (AHE-3.18–3.21) |
| [Metrics Reference](docs/reference/metrics.md) | Catalog of every `agent_utilities_*` Prometheus series |

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
