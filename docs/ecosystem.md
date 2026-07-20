# The Ecosystem — How agent-utilities Fits

agent-utilities is the **spine** of a larger `agent-packages/*` ecosystem: the
shared library + knowledge graph + orchestration that every other piece builds
on. This page maps the pieces and how a request flows through them.

> Hostnames below are generalized placeholders (`*.example`). Substitute
> your own. No secrets or real endpoints appear here.

## The pieces (one-liners)

### Core
| Project | Role |
|---|---|
| **agent-utilities** | The foundational library: Pydantic-AI harness, graph orchestration, KG facades, ontology, config, the GraphOS MCP/fleet surface, and REST gateway. |
| **epistemic-graph** | The Rust-native graph engine — **the ONE database / authority** (compute + in-memory cache + OWL/Datalog reasoning + durable persistence), reached out-of-process over MessagePack/UDS — **no PyO3**. Writes fan out to optional durable **mirrors** (Postgres/pg-age, Neo4j, FalkorDB, LadybugDB) for interop/BI/DR. |

### Frontends (all consume the agent-utilities REST gateway / MCP)
| Project | Role |
|---|---|
| **agent-webui** | React web dashboard — chat, graph explorer, ontology Object/Vertex views, and the **Fleet Supervisor** (swarm health, topology, pause/kill, approvals). |
| **agent-terminal-ui** | Textual TUI — sessions, goals, durable task queue, multi-session agent view. |
| **geniusbot** | PySide6 desktop cockpit — service/finance/infra dashboards + embedded terminal. |

### Capabilities & connectors
| Project | Role |
|---|---|
| **agents/&ast;** (the `*-mcp` fleet) | 65 MCP connectors to enterprise systems (ServiceNow, ERPNext, GitLab/GitHub, LeanIX, ArchiMate, Twenty CRM, Camunda, Keycloak, OpenBao, Technitium DNS, Portainer, Kafka, …). Each runs as a streamable-http container; all template off `create_mcp_server()`. |
| **universal-skills** | 40+ reusable agent skills (deployment, infra, security, workflows) — including the day-0 bootstrap workflow. |
| **skill-graphs** | Generates skill-graph definitions and capability composition. |

## Provider dependency contract

The provider inventory is defined by repository-manager's `workspace.yml`; package
discovery must not depend on a separately maintained list. Every declared provider uses
the publishable range `agent-utilities>=1.27.1,<2.0.0` while local ecosystem development
resolves the exact sibling checkout with this uv source:

```toml
[tool.uv.sources]
agent-utilities = { path = "../../agent-utilities", editable = true }
```

Agent Utilities has a hard base dependency on the one supported
`epistemic-graph[full]` artifact. The `[mcp]` extra is connector-focused and adds the MCP
serving surface; `[agent-runtime]` additionally adds model orchestration. Neither extra
selects or owns a different engine build. Provider documentation must not describe the
graph engine as exclusive to the agent runtime or absent from MCP installations.

Validate all 65 checkouts, dependency declarations, registry-safe source declarations, and MkDocs
content from the ecosystem workspace:

```bash
python scripts/check_provider_fleet_contract.py
```

### Enterprise service layer (optional, à-la-carte)
| Service | Role | When |
|---|---|---|
| **Keycloak** | OIDC/SAML SSO — root of auth trust | enterprise |
| **OpenBao** | Secrets engine / vault | single-node prod + enterprise |
| **Technitium DNS** | Authoritative `.example` zone | enterprise (swarm) |
| **Caddy** | HTTPS ingress / reverse proxy | single-node prod + enterprise |
| **Langfuse** | LLM observability / tracing | any (optional) |
| **LGTM** | Prometheus/Loki/Grafana/Tempo observability | enterprise |
| **Postgres/pg-age** | Durable KG **mirror** (engine fan-out target); also the shared fleet state store (`STATE_DB_URI`) | single-node prod + enterprise |
| **Kafka** | Event backbone + `kg_tasks`/`agent_turns` work queues for ingest and dispatch workers | enterprise (optional) |

### Scale-out workers (optional, any host)

| Process | Role | Flag |
|---|---|---|
| **engine cluster** | Tenant-partitioned `epistemic-graph`; the engine catalog routes fenced MultiRaft groups | stable `GRAPH_SERVICE_ENDPOINTS` coordinator |
| **kg-ingest-worker** | Joins the `kg-ingest` consumer group and drains the ingest task queue as an engine client | `TASK_QUEUE_BACKEND=kafka` (or `postgres`) |
| **agent-dispatch-worker** | Claims session-keyed agent turns and executes them through fenced WorkItems | always queue-driven |

## How a request flows

```mermaid
flowchart LR
    U[User / external agent] -->|MCP or REST + JWT| GW["graph-os MCP / REST gateway<br/>(GATEWAY_WORKERS, /metrics)"]
    GW --> ENG[("epistemic-graph cell<br/>catalog-routed MultiRaft groups")]
    GW --> ORCH["Orchestrator<br/>router → planner → swarm"]
    GW -->|enqueue turns| Q[("agent_turns / kg_tasks queues<br/>Kafka / Postgres / SQLite")]
    Q --> DW[agent-dispatch-worker fleet]
    Q --> IW[kg-ingest-worker fleet]
    DW --> ENG
    IW --> ENG
    ORCH -->|spawns| AG[Agents / teams]
    AG -->|tools via GraphOS| FLEET[*-mcp connector fleet]
    FLEET --> EXT[("ServiceNow / ERPNext /<br/>GitLab / Kafka / …")]
    GW --> SUP["Fleet supervisor + autonomy plane<br/>/api/fleet/* → ActionPolicy"]
    SUP --> UI[agent-webui / TUI / geniusbot]
    GW -.traces.-> LF[Langfuse]
    GW -.metrics.-> PROM[Prometheus]
```

1. A user or external agent calls **graph-os** (MCP) or the **REST gateway**;
   requests are scoped to a server-minted `ActorContext` (JWT identity,
   OS-5.14) and rate-limited per tenant.
2. The engine handles KG reads/writes — one local engine by default, or an
   engine cell with catalog-owned placement epochs and fenced MultiRaft groups at scale; the
   **orchestrator** decomposes goals into teams/swarms. In queue mode, agent
   turns and ingest tasks flow through durable queues to stateless
   **dispatch/ingest worker fleets** on any host.
3. Spawned agents reach external systems through the **`*-mcp` fleet**, federated
   by GraphOS (per-child limits, circuit breakers, restart-on-crash).
4. The **fleet supervisor** surfaces health/topology/events/approvals to the
   UIs, and the opt-in **autonomy control plane** (ActionPolicy-gated
   reconciler, playbooks, deploy watch, autoscaler) acts on them; traces flow
   to **Langfuse** and metrics to **Prometheus** when configured.

## Deploying the ecosystem

The connector fleet and the backend are deployed by profile — see
[Day-0 Deployment](guides/day0.md) and the recipes
([tiny](recipes/tiny.md) · [single-node prod](recipes/single-node-prod.md) ·
[enterprise](recipes/enterprise.md)). The canonical service list lives in the
generated `mcp-fleet.registry.yml`.
