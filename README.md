# Agent Utilities

A batteries-included Python harness for building, orchestrating, and running AI
agents against a shared knowledge graph — zero-infra by default.

[![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Build](https://img.shields.io/github/actions/workflow/status/Knuckles-Team/agent-utilities/release.yml?branch=main)](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml)
[![PyPI - License](https://img.shields.io/pypi/l/agent-utilities)](LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Docs](https://img.shields.io/badge/docs-published-blue)](https://knuckles-team.github.io/agent-utilities/)
[![Engine: epistemic-graph](https://img.shields.io/badge/engine-epistemic--graph-6f42c1)](https://github.com/Knuckles-Team/epistemic-graph)

*Version: 2.5.0*

> **New here?** Read **[docs/start-here.md](docs/start-here.md)** — one page,
> the real onboarding entry point. For AIs, **[llms.txt](llms.txt)** is the
> entry index.

## What it is

`agent-utilities` is a batteries-included harness for building Pydantic-AI
agents that ship with a knowledge graph, orchestration, memory, and tools
built in. Install it and use it three ways: as a **library** you import into
Python code (`from agent_utilities import create_agent`), as an **MCP server**
(`graph-os`) that hands an existing agent — Claude Code, Cursor, your own —
the knowledge graph and tool surface, or as an **HTTP/REST gateway** sharing
one KG backend across many clients. All three sit on one engine — a fast Rust
knowledge-graph engine that does compute, caching, semantics, and durable
storage — so whichever surface you pick, you're talking to the same brain.
Writes fan out asynchronously to optional durable mirrors (Postgres/pg-age,
Neo4j, FalkorDB) for interop, BI, or disaster recovery, but the engine stays
the one read/write authority; nothing downstream of it changes when you add
one. The default needs no databases or external services: the knowledge graph
runs in-process, so a fresh checkout can create, query, and persist a graph
before you've installed anything else — no Postgres, no Neo4j, no separate
graph server to stand up first. Full trade-offs:
**[Consumption Models](docs/guides/consumption-models.md)**.

## Quickstart

```bash
pip install agent-utilities          # zero external *service* deps to start
```

Point it at any model provider (`OPENAI_API_KEY`, or a local vLLM/Ollama
endpoint), then create an agent — skills, tools, and the in-process KG included:

```python
from agent_utilities import create_agent

agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("What can you do?").output)
```

Or stand up the whole platform and verify it — three commands, zero infra:

```bash
setup-config generate --profile tiny     # complete config.json (every option)
graph-os &                                # KG MCP server — no database needed
agent-utilities-doctor                    # one health sweep across every subsystem
```

Or work with the knowledge graph directly, no database required:

```python
from agent_utilities.mcp import kg_server   # epistemic-graph is the default — zero-infra

await kg_server._execute_tool("graph_write", action="add_node",
    node_id="svc:payments", node_type="Service",
    properties='{"team":"fintech","tier":"critical"}')

res = await kg_server._execute_tool("graph_query",
    cypher="MATCH (n:Service) WHERE n.tier='critical' RETURN n")
```

Scale up with `--profile single-node-prod`/`enterprise`, add Stardog + pg-age,
or let Claude set itself up — all in the **[Quick Start Guide](docs/guides/quick-start.md)**
and **[Self-Setup Guide](docs/guides/self-setup.md)**. The full capability
catalog (search, ingest, orchestrate, ontology, memory) is in
**[docs/capabilities.md](docs/capabilities.md)**; runnable code is in the
[reference agent](examples/reference_agent/).

> **Heads-up — this is two repos.** The heavy graph compute lives in a
> **separate** Rust engine, [`epistemic-graph`](https://github.com/Knuckles-Team/epistemic-graph)
> (reached out-of-process over MessagePack/UDS — no PyO3, no Rust toolchain
> needed here). AI agents pointed here to deploy this: follow **[Zero-to-deployed](AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator)**
> in `AGENTS.md`.

## Key Features

Grouped by what they do — link out for the full catalog with every concept ID:
**[docs/guides/features.md](docs/guides/features.md)**.

- **Knowledge graph & memory** — one Rust engine is the authority for compute,
  cache, OWL semantics, and durable persistence; optional Postgres/Neo4j/
  FalkorDB mirrors fan out writes.
- **Ontology system** (Palantir-Foundry parity) — objects, links, interfaces,
  derived properties, action types, and object-set permissioning, graph-native.
- **Orchestration & self-evolution** — Spec-Driven Development, capability
  auto-activation, and a governed evolution loop that proposes changes for
  review and never auto-pushes them.
- **Enterprise integration (Company Brain)** — a document-source connector
  framework plus the ~58-server MCP fleet feed a 6-layer ingestion runtime
  with trust-decay conflict resolution, field-level survivorship, data ACLs
  and tenant scoping, and a human-correction→rule→eval feedback loop.
- **Scale-out planes, all opt-in** — externalized durable state, tenant-sharded
  engines with HRW routing, Kafka ingest scale-out, queue-driven agent
  dispatch, and a Prometheus-instrumented gateway; the zero-infra default is
  unchanged until you turn these on.
- **Inference acceleration** — KV-cache layering across vLLM/LMCache/the
  engine, plus a numpy-compatible numeric shim backed by the engine's own
  kernel.
- **Autonomy & governance** — a fail-closed `ActionPolicy` gate, server-minted
  identity, and a hardened MCP fleet gateway built into `graph-os`.

Benchmarked against a conventional stitched memory stack (separate vector DB +
BM25 + app-level fusion), the unified KG memory matches recall (1.000) while
retrieving ~3.6× faster and surviving a restart via a durable KV cold-tier
(100% survival) — full scorecard in the
[Phase-2 benchmark report](https://github.com/knuckles-team/epistemic-graph/blob/main/docs/benchmarks.md#phase-2-agent-memory--kv-cache-benchmark-measured).
A handful of capabilities are real and importable today but lightly
documented — causal reasoning, a `SKILL.md`-to-`GraphPlan` compiler, and
graph-native event sourcing among them; see
**[docs/guides/features.md](docs/guides/features.md)** for the full list.

## Architecture at a glance

<!-- BEGIN GENERATED: concepts -->

Synthesized from concept markers in the codebase into **1216 canonical concepts** across **9 pillars**.

> This count is generated from `docs/concepts.yaml` by `scripts/gen_docs.py` — do not edit by hand. The table below covers the 5 pillars agent-utilities itself owns; the other 4 (37 concepts) belong to the epistemic-graph engine's own pillar set. Live per-pillar status: [docs/status.md](docs/status.md).

| # | Pillar | Focus | Concepts | Docs |
|:-:|:-------|:------|:--------:|:-----|
| 1 | Graph Orchestration | Planning, SDD lifecycle, dynamic multi-layer execution | 219 | [docs/pillars/1_graph_orchestration.md](docs/pillars/1_graph_orchestration.md) |
| 2 | Epistemic Knowledge Graph | The one engine authority — ingestion, ontology, ETL, reasoning | 512 | [docs/pillars/2_epistemic_knowledge_graph.md](docs/pillars/2_epistemic_knowledge_graph.md) |
| 3 | Agentic Harness Engineering | Self-models, evaluation, governed self-evolution | 120 | [docs/pillars/3_agentic_harness_engineering.md](docs/pillars/3_agentic_harness_engineering.md) |
| 4 | Ecosystem & Peripherals | MCP fleet, messaging, connectors, UI surfaces | 141 | [docs/pillars/4_ecosystem_peripherals.md](docs/pillars/4_ecosystem_peripherals.md) |
| 5 | Agent OS Infrastructure | Auth, governance, deployment, scaling | 187 | [docs/pillars/5_agent_os_infrastructure.md](docs/pillars/5_agent_os_infrastructure.md) |

<!-- END GENERATED: concepts -->

All four consumption surfaces (library, `graph-os` MCP, REST gateway, and any
IDE/agent) talk to **one** `graph-os` MCP server — it serves the knowledge
graph natively and doubles as the fleet gateway for the other ~58 MCP servers,
loading them on demand via `find_tools`/`load_tools` so hundreds of fleet
tools stay out of context until asked for. Wiring it into Claude Code/Cursor/
etc. via `mcp_config.json` (generate with `setup-config mcp` — don't
hand-write it), self-contained vs. shared-engine configs, Keycloak-protected
fleets, and every env var: **[Consumption Models](docs/guides/consumption-models.md)**.
Full architecture, every pillar deep-dive, C4 diagrams, and the Company Brain
and Vendor-Neutral Enterprise Ontology write-ups: **[docs/index.md](docs/index.md)**.

## Installation & Deployment

```bash
pip install agent-utilities              # add "[all]" for MCP servers, UI, and external graph backends
```

Out of the box it runs zero-infrastructure — no database or graph server to
stand up; the bundled Rust `epistemic-graph` engine is the one authority for
compute, cache, and durable persistence. Add a durable Postgres mirror later
(`GRAPH_MIRROR_TARGETS`/`KG_CONNECTIONS`) if you need one for interop/BI/DR.
See the **[Installation Guide](docs/guides/installation.md)**.

Model providers, routing, and secrets are configured centrally via
`~/.config/agent-utilities/config.json` (every field has a matching
environment-variable override) — generate one with `setup-config generate`,
then see the **[Configuration Guide](docs/guides/configuration.md)** and the
**[Local Secret Storage Guide](docs/guides/secrets-auth.md)**.

To go beyond a laptop — `graph-os` over stdio/streamable-HTTP, the REST
gateway, Docker composes, and sharded/queue-driven production shapes — the
**[Deployment Configurations](docs/guides/deployment-configurations.md)**
guide walks every step from zero-infra to a governed, multi-tenant fleet. The
**[Enterprise Enablement Runbook](docs/guides/enterprise-enablement-runbook.md)**
is the ordered push → deploy → flag-enablement sequence for turning on the
opt-in scale-out and autonomy planes once you're already deployed. Once
installed, run **`agent-utilities install`** to drop the skill toolkit into
your agent tool (Claude Code, Cursor, etc.) — it unlocks the deployment,
evolution, and knowledge-graph skills the rest of this README assumes.

## Documentation

- **[docs/start-here.md](docs/start-here.md)** — the real onboarding entry
  point: what this is, the three ways to use it, and the zero-infra knowledge
  graph, in one page.
- **[AGENTS.md](AGENTS.md)** — contributor/agent working discipline:
  architecture reference, coding conventions, the branching/merge-queue
  workflow, and the zero-to-deployed genesis procedure.
- **[docs/status.md](docs/status.md)** — the live concept/capability status
  page: what's built vs. roadmap, by pillar.
- **[docs/journey.md](docs/journey.md)** — *optional deep-dive*: a narrative
  walkthrough of the platform in motion, for readers who prefer a story to
  config tables.
- **[CHANGELOG.md](CHANGELOG.md)** — release history and the roadmap direction
  beyond a single agent harness (distributed agentic evolution).

Everything else — architecture, pillar deep-dives, guides — is indexed from
**[docs/index.md](docs/index.md)**.

agent-utilities is also the entrypoint for the wider `agent-packages`
ecosystem — 65 connector packages, three frontends (geniusbot, agent-webui,
agent-terminal-ui), a skill library, and ontologies. See
**[The wider ecosystem](docs/ecosystem.md)**.

## Contributing

Fork the repo, write tests for new functionality (assertions required, not
just coverage), follow the established Pydantic models/structured
prompts/concept markers, and run `uv run pytest tests/ -q` before submitting
— a 60-second timeout applies to every test, so an unbounded `time.sleep`
fails automatically. Update `docs/` if your change affects a public API. See
**[CONTRIBUTING.md](CONTRIBUTING.md)** and **[AGENTS.md](AGENTS.md)** for the
full conventions and architecture rules.

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.
