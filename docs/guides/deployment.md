# Deploying agent-utilities

This guide covers everything needed to deploy `agent-utilities` — from a
self-contained installation with **no external services** up to a distributed,
production-grade cluster. It documents the sole `graph-os` MCP process over
**stdio** or **streamable-http**, including its built-in fleet gateway, and where
the REST API lives.

> **CONCEPT:AU-ECO.messaging.native-backend-abstraction / OS-5.x** — MCP standardized interfaces + Agent OS deploy.

---

## 1. Install

```bash
# Core Python library and mandatory full epistemic-graph engine
pip install agent-utilities

# Supported headless GraphOS serving runtime
pip install "agent-utilities[serving]"

# Every optional integration, only when this host needs all of them
pip install "agent-utilities[all]"
```

Useful extras compose with `agent-utilities[serving,...]`:

| Extra | Adds |
|-------|------|
| `mcp` | FastMCP server stack (`graph-os`) |
| `graph` | Graph compute / retrieval deps |
| `postgresql` | `psycopg` driver for the optional Postgres/pg-age mirror |
| `owl` / `stardog` | OWL ontology + SPARQL reasoning |
| `auth` / `vault` | JWT/OIDC auth + OpenBao secrets |

The epistemic-graph engine is the one database and ships with the wheel, so the
out-of-box experience needs no external database server. Mirror drivers
(`postgresql`, `neo4j`, `falkordb`) are only required when you fan writes out to
those mirrors.

---

## 2. The out-of-box default: a self-contained installation

Agent-utilities always runs the **epistemic-graph** engine as the one database
and authority for all reads and writes:

```
epistemic-graph engine   (Rust-native, always included)
  · authority / system of record (durable persistence)
  · in-memory cache + native graph compute + semantic/ontology reasoning
```

GraphOS runs as the Python console entry point and supervises the bundled Rust
`epistemic-graph-server` as an out-of-process child over a private socket. The two
processes come from one installation and need **no external service** (no Postgres,
Neo4j, or FalkorDB server required). It is the recommended default for
local development, edge/offline agents, demos, single-node, and most production
deployments.

```bash
# Generate the exact zero-infrastructure authority boundary, then run:
setup-config generate --profile tiny
agent-utilities-doctor --only graph_identity auth
graph-os --transport stdio
```

For this one shape, leave `GRAPH_SERVICE_ENDPOINTS`, `KG_AUTH_TOKEN_REF`, and
`KG_IDENTITY_OAUTH2` unset. GraphOS signs and validates a short-lived JWT using an
in-memory key and fixed neutral service claims as a one-time proof, destroys the
key and token, and returns a process-lifetime session. It persists no personal,
host, endpoint, filesystem, credential, or proof data.

Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point must instead configure exactly one `KG_AUTH_TOKEN_REF` or
`KG_IDENTITY_OAUTH2` in XDG AgentConfig plus the external JWT validation policy.
Raw tokens and OAuth2 client secrets are rejected as durable configuration.
GraphOS aborts when identity, audience, tenant authority, JWKS, or policy pinning
cannot be established; failure never falls back to local authority.
External stdio authority remains bounded by a renewable shared expiry-only lease.
Renewal must preserve the original identity and capabilities; drift is rejected,
failed renewal never extends the lease, and graph work fails closed at expiry.

The engine's durable store lives under the discovered XDG data directory; inject
`GRAPH_SERVICE_PERSIST_DIR` at runtime to override it without recording a machine path
in project configuration.

### Backend selection cheat-sheet

| Goal | AgentConfig |
|------|-----|
| Default — the engine only (self-contained, zero-infra) | *(unset)* |
| Isolated tests/CI | The real ephemeral epistemic-graph fixture |
| Engine + projections for interop/BI/DR | `GRAPH_MIRROR_TARGETS=postgresql` + `GRAPH_DB_CONNECTION_PROFILE_REF=secret://graph/mirror-profile` |

> The engine is always the authority and always serves every read. A mirror
> (Postgres, Neo4j, FalkorDB, Ladybug) only receives an **asynchronous, lossless**
> copy of committed writes via a durable replay-on-reconnect outbox — it is never
> on the read path. Enable mirrors only when you need external query, business
> intelligence, or disaster recovery.

---

## 3. Running `graph-os` (the Knowledge Graph MCP server)

`graph-os` is a **thin FastMCP wrapper** following the standard `mcp_server.py`
template. It serves **only the MCP tool surface**; the REST API is centralized in
the API gateway (see §5).

```bash
# stdio (local agent integration)
graph-os --transport stdio

# streamable-http (local loopback)
graph-os --transport streamable-http --host 127.0.0.1 --port 8004
```

Non-loopback streamable HTTP requires configured JWT/OIDC authentication and
trusted TLS termination. The server rejects unauthenticated remote binding.

GraphOS accepts `--transport {stdio,streamable-http}`, `--host`, `--port`, plus
auth/eunomia flags. A liveness endpoint is served at `GET /health` under the
streamable-HTTP transport.

Tools exposed: `graph_query`, `graph_search`, `graph_write`, `graph_ingest`,
`graph_analyze`, `graph_orchestrate`, `graph_configure`, `graph_sessions`,
`graph_goals`, `source_sync`, `graph_feedback`.

---

## 4. Built-in MCP fleet gateway

GraphOS is the only MCP process clients launch. It reads the configured
`mcpServers` catalog, exposes the bounded discovery tools (`find_tools`,
`list_catalog`, `load_tools`, and `unload_tools`), and mounts child tools on
demand. Per-child enable/disable filters, timeouts, concurrency limits, pools,
restart supervision, and circuit breakers remain internal GraphOS behavior.

```bash
graph-os --transport stdio
graph-os --transport streamable-http --host 127.0.0.1 --port 8004
```

Point local clients at the installed `graph-os` command or remote clients at
an authenticated, TLS-protected GraphOS `/mcp` URL. Both modes use the same fleet
catalog; there is no secondary MCP gateway deployment.

---

## 5. The centralized REST API (API gateway)

All Knowledge Graph **REST** endpoints are served by the API gateway
(`agent_utilities.server.app`), not by the `graph-os` MCP server. Funnelling
every client (UIs, subagents, ingestion scripts) through one persistent process
provides one authenticated policy and session boundary.

```bash
python -m agent_utilities --host 127.0.0.1 --port 9000
```

Mounted under `/api`:

- `/api/graph/query`, `/api/graph/search`, `/api/graph/write`,
  `/api/graph/ingest`, `/api/graph/analyze`, `/api/graph/orchestrate`,
  `/api/graph/configure` (+ their granular sub-routes)
- `/api/sessions`, `/api/goals`, `/api/tools`

Cypher is available as a language inside the typed `graph_query` operation. The
gateway intentionally exposes no raw query-language route; reads and mutations
therefore retain the same `GraphSession`, tenant, policy, audit, and durability
semantics as every other graph operation.

The single background KG daemon is role-gated by `KG_DAEMON_ROLE`:

| Role | Behavior |
|------|----------|
| `host` | The gateway process: runs workers, drains the work queue |
| `client` | MCP servers/agents: enqueue work, do not spawn workers |
| `auto` | Pick based on context (default) |

`graph-os-daemon` is the standalone headless host for that queue, maintenance,
and background-work loop when the REST gateway is not the host. It serves no HTTP
API. Use `python -m agent_utilities` for REST and `graph-os` for MCP.

---

## 6. MCP client wiring

Register the portable stdio launcher in Codex through its native command:

```bash
setup-config codex
# Equivalent: codex mcp add graph-os -- graph-os --transport stdio
```

Codex stores this registration in `config.toml`; do not create a Codex
`mcp_config.json`. Other clients should register the same command and arguments
through their native configuration surface. GraphOS independently reads an
optional XDG fleet catalog for progressive-disclosure tools. Keep topology,
identity, credential references, and TLS-profile references in AgentConfig.

---

## 7. Docker

Compose files live under `docker/`:

| File | Purpose |
|------|---------|
| `docker/mcp.compose.yml` | `graph-os` MCP server (streamable-http) |
| `docker/pg-age.compose.yml` | PostgreSQL + pgvector + pg-age (optional mirror) |
| `docker/neo4j.compose.yml`, `docker/falkordb.compose.yml` | Optional mirrors |
| `docker/kafka-kraft.compose.yml` | Redpanda/Kafka reactive event ledger |

```bash
# Self-contained: MCP server plus its packaged, supervised engine
docker compose -f docker/mcp.compose.yml up -d

# Add a Postgres/pg-age mirror for interop/BI/DR:
docker compose -f docker/pg-age.compose.yml up -d
# then declare the projection (fan-out enables automatically):
#   GRAPH_MIRROR_TARGETS=postgresql
#   GRAPH_DB_CONNECTION_PROFILE_REF=secret://graph/mirror-profile
```

---

## 8. External sources and Langfuse

External Neo4j/openCypher, AGE, LadybugDB/Kuzu, remote epistemic-graph, and
GraphQL sources are declared through reference-only `EXTERNAL_GRAPH_CONNECTORS`.
Deployment supplies the referenced connection, authentication, TLS, variables, and
mapping-policy documents; the repository retains no source-specific profile. Run the
bounded `discover → propose → approve → external_graph_doctor → ingest` lifecycle
described in [Universal External Graph Connectors](../architecture/universal-external-graph-connectors.md).

For Langfuse, configure `LANGFUSE_HOST`, both credential references, and a verified
TLS-profile reference in AgentConfig. The native MCP child and propose-only failure
evolution auto-enable when both credential references resolve unless explicitly
disabled. `TRACE_EXPORT_ENABLED`, `LANGFUSE_CAPTURE_CONTENT`, and
`LANGFUSE_KG_AUTO_INGEST` remain explicit gates; auto-ingestion also requires an
independent persistence HMAC-key reference.

```bash
agent-utilities-doctor --only config transport_security graph_connections langfuse
agent-utilities-doctor --live
```

---

## 9. Production hardening

Set `APP_PROFILE` to `production` to enable the profile guard
(`agent_utilities.core.profile_guard`). In production it **refuses in-memory
defaults** and requires a durable, shardable engine plus a real event broker:

- a durable engine — the packaged, supervised `epistemic_graph` engine (default), or a
  shared/remote engine via `GRAPH_SERVICE_ENDPOINTS`; the pure `memory` backend
  is rejected.
- `a2a_broker` = `redis`/`postgres`, `a2a_storage` = `redis`/`postgres`.
  These are the adapter names currently constructed by `server/app.py`; an
  engine-native broker/storage pair remains an identified implementation gap.
- `kafka_bootstrap_servers` set (the reactive event ledger needs a real broker).

```json
{
  "APP_PROFILE": "production",
  "GRAPH_MIRROR_TARGETS": ["postgresql"],
  "GRAPH_DB_CONNECTION_PROFILE_REF": "secret://graph/mirror-profile",
  "KAFKA_BOOTSTRAP_SERVERS": "broker.example.invalid:9092"
}
```

The guard raises `ProductionProfileError` listing every offending setting so an
operator can fix them all at once.

---

## 10. Verify a deployment

```bash
# Resolve the active backend (should print the epistemic-graph engine by default)
python -c "from agent_utilities.knowledge_graph.backends import create_backend as c; \
b=c(); print(type(b).__name__)"

# graph-os exposes the standard args
graph-os --help

# Validate static configuration, identity, engine, connectors, and observability
agent-utilities-doctor

# Health (HTTP transport)
curl -s localhost:8004/health

# REST via the gateway
curl -s -XPOST localhost:9000/api/graph/query -d '{"cypher":"MATCH (n) RETURN count(n)"}'
```

See also: [Configuration](configuration.md) · [Graph Engine (Authority + Mirrors)](graph_engine.md)
· [Deploying Graph Databases](graph-db-deployment.md) ·
[Universal External Graph Connectors](../architecture/universal-external-graph-connectors.md).
