# Start Here — What agent-utilities Is & How to Use It

> The single page to read first. If you are an AI agent or a developer who just
> wants to *use* this, everything you need is below or one click away.

> 🧰 **Install the skills first — they unlock how to use everything else.** After
> `pip install "agent-utilities[serving]"`, run **`agent-utilities install`**. It installs
> the ten-skill workflow toolkit for graph domains plus development, deployment, and
> evolution into a validated provider-owned XDG generation and the detected calling
> agent tools (Claude Code, etc.). `agent-utilities-doctor` flags it if the toolkit is
> missing.

## What it is, in one paragraph

**agent-utilities is a batteries-included harness for building Pydantic-AI agents
that come with a knowledge graph, orchestration, memory, and tools out of the
box.** The heavy graph compute runs in a separate Rust engine
([`epistemic-graph`](ecosystem.md)) reached out-of-process over a socket — but you
don't need Rust, Postgres, or a separately managed server to start: **GraphOS
supervises the packaged engine over a private local transport by default.** You
can consume it three ways:
import it as a **library**, run it as an **MCP server** (`graph-os`), or call its
**REST gateway**.

## One ontology over the whole ecosystem

agent-utilities maps the **entire ecosystem — `agent-packages/agents/*` + `services/*` +
enterprise systems + research papers — into ONE ontology-driven knowledge-graph
orchestration system.** OWL/RDF reasoning runs over all of it to *extrapolate new
relationships* (transitive/inverse/subclass/property-chain closures) across domains that
were never explicitly linked — research connects to the real deployed estate, not a silo.
Long-running objectives (**Loops** — research, develop, or skill execution) make that
reasoning the *engine*: each cycle promotes new information, reasons over the ecosystem,
and harvests the inferred cross-domain relationships back as the next iteration's inputs.
Automated research produces **Agent-Native Research Artifacts (ARA)** — OWL-native,
4-layer, ecosystem-grounded, OWL/SHACL-sealed — exposed via the `research_artifact` MCP
tool and `POST /api/research/*`. See [OWL/RDF Layer](architecture/owl_rdf_layer.md).

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

The zero-infrastructure path is deliberately narrow: `graph-os --transport stdio`
with `DEPLOYMENT_PROFILE=tiny`, no `GRAPH_SERVICE_ENDPOINTS`, and no
`KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`. It creates a neutral, short-lived
bootstrap JWT and key in memory as a one-time proof, validates the token through
the normal verifier, destroys both, and returns a process-lifetime session
without persisting personal, host, endpoint, filesystem, token, or proof data.
Run `agent-utilities-doctor --only graph_identity auth` before launch.
Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external process identity plus its validation
policy; acquisition or validation failure never falls back locally. External
stdio authority is bounded by a renewable shared expiry lease: identity drift is
rejected, failed renewal never extends the lease, and graph work fails closed at
expiry.

| You want to… | Use | One-liner |
|---|---|---|
| Build a standalone agent in Python | **Library** | `from agent_utilities import create_agent` |
| Give an existing agent (Claude Code, Cursor, your own) KG + tools | **MCP `graph-os`** | `graph-os` (stdio) |
| Share one KG/agent backend across many clients/containers | **MCP over HTTP** or **REST gateway** | `graph-os --transport streamable-http` / `python -m agent_utilities` (REST, default port 9000) |

### 1. As a library (standalone agent)

```python
from agent_utilities import create_agent

# Skills + universal tools + the supervised knowledge graph, ready to run.
agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("What can you do?").output)
```

### 2. As an MCP server (give any agent the KG + tools)

```bash
graph-os                       # stdio — for Claude Code / Cursor / IDEs
graph-os --transport streamable-http --host 127.0.0.1 --port 8004 # local HTTP
```

For a remote bind, configure JWT/OIDC authentication and trusted TLS
termination. An unauthenticated non-loopback MCP listener is rejected.

Register it in Codex through the native MCP command:

```bash
setup-config codex
# Equivalent: codex mcp add graph-os -- graph-os --transport stdio
```

The launcher remains machine-neutral. Engine topology, identity, TLS, and secret
references belong in AgentConfig, not Codex's `config.toml`. Use each other MCP
client's native registration mechanism for the same command and arguments.

The agent now has `graph_query`, `graph_search`, `graph_ingest`, `graph_orchestrate`,
`ontology_*`, and more — see [Capabilities](capabilities.md).

### 3. As a REST gateway (one backend, many clients)

```bash
python -m agent_utilities             # REST gateway, default :9000
curl -s localhost:9000/api/graph/query -d '{"cypher":"MATCH (n) RETURN n LIMIT 5"}'
```

## The knowledge graph is free and native

You do **not** need a database to use the KG. The default backend is
`epistemic_graph`: the Rust engine is the one authority — compute, cache,
semantic, and durable persistence in a single store. Zero separately managed
servers, zero connector config:

Epistemic-graph is always the authority, so no backend selector is required.

When you want optional projections, point `GRAPH_MIRROR_TARGETS` at
Postgres/pg-age (or other) mirror connections; the
engine stays the authority and fans writes out to the mirrors. See
[Deployment Recipes](recipes/tiny.md) for tiny → single-node → enterprise, and
[Stardog + pg-age databases](recipes/databases.md) to push your ontology to
Stardog (or a local SPARQL endpoint) and backfill relationships into Apache AGE
through runtime connection-profile references in one command. For a config-complete, end-to-end install (the
path Claude follows to set itself up), see the [Self-Setup guide](guides/self-setup.md)
— one command generates a `config.json` covering every option and a `doctor`
validates the deployment.

External Neo4j/openCypher, AGE, LadybugDB/Kuzu, remote epistemic-graph, and
GraphQL sources use one reference-only
[discovery, mapping, approval, and ingestion lifecycle](architecture/universal-external-graph-connectors.md).
Agent Utilities ships the native connection points and governance contracts, not an
environment-specific endpoint, query, schema profile, or ontology.

## When one host is not enough

Every scale-out lever is opt-in and leaves the zero-infra default untouched:
one shared Postgres state store (`STATE_DB_URI`), an Epistemic Graph cell whose
placement catalog routes tenant graphs to fenced MultiRaft groups through a stable
coordinator (`GRAPH_SERVICE_ENDPOINTS`), Kafka-backed
ingest workers (`TASK_QUEUE_BACKEND=kafka` + `kg-ingest-worker`), a
queue-driven agent-dispatch fleet (`agent-dispatch-worker`), and pre-forked
gateway workers with per-tenant rate
limiting (`GATEWAY_WORKERS`). The flagship guide walks every configuration
from laptop to fleet: **[Deployment Configurations](guides/deployment-configurations.md)**.

## Where to go next

- **[Capabilities](capabilities.md)** — the concrete list of what an agent can do, with copy-paste snippets.
- **[Consumption Models](guides/consumption-models.md)** — library vs MCP stdio vs MCP HTTP vs REST.
- **[Universal External Graph Connectors](architecture/universal-external-graph-connectors.md)** — schema discovery, digest-bound mapping approval, and governed ingestion.
- **[Loop Engine](guides/loop-engine.md)** — run self-improvement, research, and goal loops through the `graph_loops` entry point and autonomous daemon tick.
- **[Deployment Configurations](guides/deployment-configurations.md)** — the flagship guide: every deployment shape from zero-infra laptop to sharded, queue-driven fleet.
- **[Ecosystem](ecosystem.md)** — how agent-utilities anchors the wider `agent-packages/*` fleet.
- **[Day-0 Deployment](guides/day0.md)** — from `scripts/bootstrap.sh` to a full enterprise swarm.
- **[Operational examples](examples/mcp-consumption.md)** — focused walkthroughs: [ontology→workflow](examples/ontology-to-workflow.md), [fleet events wiring](examples/fleet-events-wiring.md), [action-policy postures](examples/action-policy-postures.md), [autoscaling signals](examples/autoscaling-signals.md), [engine sharding](examples/sharding-walkthrough.md), [queue dispatch](examples/queue-dispatch-walkthrough.md), [evolution publication](examples/evolution-publication.md), [observability](examples/observability.md), [identity/JWT](examples/identity-jwt.md), [MCP consumption](examples/mcp-consumption.md).
- **[Metrics reference](reference/metrics.md)** — every Prometheus series the platform emits.
- **[Reference agent](https://github.com/Knuckles-Team/agent-utilities/tree/main/examples/reference_agent)** — runnable end-to-end examples.
- **[AGENTS.md](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md)** — conventions & architecture rules for contributors/AIs editing the repo.
