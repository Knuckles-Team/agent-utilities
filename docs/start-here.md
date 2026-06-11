# Start Here — What agent-utilities Is & How to Use It

> The single page to read first. If you are an AI agent or a developer who just
> wants to *use* this, everything you need is below or one click away.

## What it is, in one paragraph

**agent-utilities is a batteries-included harness for building Pydantic-AI agents
that come with a knowledge graph, orchestration, memory, and tools out of the
box.** The heavy graph compute runs in a separate Rust engine
([`epistemic-graph`](ecosystem.md)) reached out-of-process over a socket — but you
don't need Rust, Postgres, or any server to start: **the default knowledge graph
runs in-process and costs you nothing to turn on.** You can consume it three ways:
import it as a **library**, run it as an **MCP server** (`graph-os`), or call its
**REST gateway**.

## The 5 pillars (what's inside)

| Pillar | What it gives you | Deep dive |
|---|---|---|
| **1. Graph Orchestration** | A router→planner→dispatcher that turns a goal into a coordinated team/swarm of agents at runtime | [pillar 1](pillars/1_graph_orchestration.md) |
| **2. Epistemic Knowledge Graph** | A temporal, OWL-aware KG with ingestion, hybrid search, and a Palantir-parity ontology — the agent's memory and world model | [pillar 2](pillars/2_epistemic_knowledge_graph.md) |
| **3. Agentic Harness Engineering** | Self-models, evaluation, and evolution-from-failure for the agents themselves | [pillar 3](pillars/3_agentic_harness_engineering.md) |
| **4. Ecosystem & Peripherals** | The `graph-os` MCP tools, the hardened MCP multiplexer, and connectors to the wider `*-mcp` fleet | [pillar 4](pillars/4_ecosystem_peripherals.md) |
| **5. Agent OS** | Sessions, goals, the REST gateway, server-minted JWT identity, the fleet supervisor + autonomy control plane, Prometheus metrics, tool safety | [pillar 5](pillars/5_agent_os_infrastructure.md) |

## Three ways to use it (pick one)

See [Consumption Models](guides/consumption-models.md) for the full trade-offs.
The short version:

| You want to… | Use | One-liner |
|---|---|---|
| Build a standalone agent in Python | **Library** | `from agent_utilities import create_agent` |
| Give an existing agent (Claude Code, Cursor, your own) KG + tools | **MCP `graph-os`** | `uv run graph-os` (stdio) |
| Share one KG/agent backend across many clients/containers | **MCP over HTTP** or **REST gateway** | `uv run graph-os --transport streamable-http` / `python -m agent_utilities` (REST, default port 9000) |

### 1. As a library (standalone agent)

```python
from agent_utilities import create_agent

# Skills + universal tools + the in-process knowledge graph, ready to run.
agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("What can you do?").output)
```

### 2. As an MCP server (give any agent the KG + tools)

```bash
uv run graph-os                       # stdio — for Claude Code / Cursor / IDEs
uv run graph-os --transport streamable-http --host 0.0.0.0 --port 8004   # HTTP
```

Register it in your client's `mcp_config.json`:

```json
{ "mcpServers": { "graph-os": { "command": "uv", "args": ["run", "graph-os"] } } }
```

The agent now has `graph_query`, `graph_search`, `graph_ingest`, `graph_orchestrate`,
`ontology_*`, and more — see [Capabilities](capabilities.md).

### 3. As a REST gateway (one backend, many clients)

```bash
python -m agent_utilities             # REST gateway, default :9000
curl -s localhost:9000/api/graph/query -d '{"cypher":"MATCH (n) RETURN n LIMIT 5"}'
```

## The knowledge graph is free and native

You do **not** need a database to use the KG. The default backend is `tiered`:
the Rust `epistemic_graph` working store (L1) in front of an embedded LadybugDB
(L2). Zero servers, zero config:

```bash
export GRAPH_BACKEND=tiered     # this is already the default
```

When you outgrow it, point `GRAPH_DB_URI` at Postgres/pggraph and the durable
tier switches automatically — nothing else changes. See
[Deployment Recipes](recipes/tiny.md) for tiny → single-node → enterprise.

## When one host is not enough

Every scale-out lever is opt-in and leaves the zero-infra default untouched:
one shared Postgres state store (`STATE_DB_URI`), tenant-sharded KG engines
behind client-side HRW routing (`GRAPH_SERVICE_ENDPOINTS`), Kafka-backed
ingest workers (`TASK_QUEUE_BACKEND=kafka` + `kg-ingest-worker`), a
queue-driven agent-dispatch fleet (`AGENT_DISPATCH_BACKEND=queue` +
`agent-dispatch-worker`), and pre-forked gateway workers with per-tenant rate
limiting (`GATEWAY_WORKERS`). The flagship guide walks every configuration
from laptop to fleet: **[Deployment Configurations](guides/deployment-configurations.md)**.

## Where to go next

- **[Capabilities](capabilities.md)** — the concrete list of what an agent can do, with copy-paste snippets.
- **[Consumption Models](guides/consumption-models.md)** — library vs MCP stdio vs MCP HTTP vs REST.
- **[Deployment Configurations](guides/deployment-configurations.md)** — the flagship guide: every deployment shape from zero-infra laptop to sharded, queue-driven fleet.
- **[Ecosystem](ecosystem.md)** — how agent-utilities anchors the wider `agent-packages/*` fleet.
- **[Day-0 Deployment](guides/day0.md)** — from `scripts/bootstrap.sh` to a full enterprise swarm.
- **[Operational examples](examples/)** — focused walkthroughs: [ontology→workflow](examples/ontology-to-workflow.md), [fleet events wiring](examples/fleet-events-wiring.md), [action-policy postures](examples/action-policy-postures.md), [autoscaling signals](examples/autoscaling-signals.md), [engine sharding](examples/sharding-walkthrough.md), [queue dispatch](examples/queue-dispatch-walkthrough.md), [evolution publication](examples/evolution-publication.md), [observability](examples/observability.md), [identity/JWT](examples/identity-jwt.md), [MCP consumption](examples/mcp-consumption.md).
- **[Metrics reference](reference/metrics.md)** — every Prometheus series the platform emits.
- **[Reference agent](https://github.com/Knuckles-Team/agent-utilities/tree/main/examples/reference_agent)** — runnable end-to-end examples.
- **[AGENTS.md](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md)** — conventions & architecture rules for contributors/AIs editing the repo.
