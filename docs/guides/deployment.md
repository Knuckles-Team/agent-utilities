# Deploying agent-utilities

This guide covers everything needed to deploy `agent-utilities` — from a single
self-contained binary with **no external dependencies** up to a distributed,
production-grade cluster. It also documents how the two MCP servers (`graph-os`
and `mcp-multiplexer`) are run as standard **stdio** or **streamable-http**
servers, and where the REST API lives.

> **CONCEPT:AU-ECO.messaging.native-backend-abstraction / OS-5.x** — MCP standardized interfaces + Agent OS deploy.

---

## 1. Install

```bash
# Core library (zero-infra default backend works out of the box)
pip install agent-utilities

# Everything: MCP servers, UI, embeddings, graph backends, messaging, auth…
pip install "agent-utilities[all]"
```

Useful extras (compose with `agent-utilities[mcp,graph,postgresql]`):

| Extra | Adds |
|-------|------|
| `mcp` | FastMCP server stack (`graph-os`, `mcp-multiplexer`) |
| `graph` | Graph compute / retrieval deps |
| `postgresql` | `psycopg` driver for the optional Postgres/pg-age mirror |
| `owl` / `stardog` | OWL ontology + SPARQL reasoning |
| `auth` / `vault` | JWT/OIDC auth + OpenBao secrets |

The epistemic-graph engine is the one database and ships with the wheel, so the
out-of-box experience needs no external database server. Mirror drivers
(`postgresql`, `neo4j`, `falkordb`) are only required when you fan writes out to
those mirrors.

---

## 2. The out-of-box default: a single self-contained binary

When `GRAPH_BACKEND` is unset, agent-utilities runs the **`epistemic_graph`**
engine alone — the one database, the authority for all reads and writes:

```
epistemic-graph engine   (Rust-native, always included)
  · authority / system of record (durable persistence)
  · in-memory cache + native graph compute + semantic/ontology reasoning
```

This runs entirely in one process with **no external system dependencies** (no
Postgres, Neo4j, or FalkorDB server required). It is the recommended default for
local development, edge/offline agents, demos, single-node, and most production
deployments.

```bash
# Nothing to configure — just run:
graph-os                     # or: python -m agent_utilities.mcp.kg_server
```

The engine's durable store lives at the XDG path
`~/.local/share/agent-utilities/kg/` (override with `GRAPH_DB_PATH`).

### Backend selection cheat-sheet

| Goal | Env |
|------|-----|
| Default — the engine only (self-contained, zero-infra) | *(unset)* |
| Pure ephemeral, in-memory (tests/CI) | `GRAPH_BACKEND=memory` |
| Engine + mirrors for interop/BI/DR | `GRAPH_BACKEND=fanout` + `GRAPH_MIRROR_TARGETS=postgresql` + `GRAPH_DB_URI=postgresql://…` |

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
# stdio (local agent integration — Claude Code, the multiplexer, etc.)
graph-os --transport stdio

# streamable-http (remote / containerized)
graph-os --transport streamable-http --host 0.0.0.0 --port 8004
```

Standard args (from `create_mcp_server`): `--transport {stdio,streamable-http,sse}`,
`--host`, `--port`, plus auth/eunomia flags. A liveness endpoint is served at
`GET /health` under HTTP transports.

Tools exposed: `graph_query`, `graph_search`, `graph_write`, `graph_ingest`,
`graph_analyze`, `graph_orchestrate`, `graph_configure`, `graph_sessions`,
`graph_goals`, `graph_hydrate`, `graph_feedback`.

---

## 4. Running `mcp-multiplexer` (one server, many tools)

The multiplexer aggregates many child MCP servers (declared in an
`mcp_config.json`) into one unified server, namespacing each child's tools with a
short, host-aware prefix. It is also a standard FastMCP server:

```bash
# stdio
mcp-multiplexer --config /path/to/mcp_config.json --transport stdio

# streamable-http
mcp-multiplexer --config mcp_config.json --transport streamable-http --host 0.0.0.0 --port 8005
```

`--config` defaults to `$MCP_CONFIG`, then a discovery list
(`~/.config/agent-utilities/mcp_config.json`, `./mcp_config.json`, …). Per-child
`enabledTools` / `disabledTools` (fnmatch) and `timeout` are honored; the
multiplexer skips itself and any `disabled` server to avoid recursion.

**Eager vs. dynamic (`MCP_MULTIPLEXER_MODE`).** The default `eager` mode mounts
every child's tools up front. `dynamic` mode (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog) boots with only the
meta-tools `find_tools` / `load_tools` / `unload_tools` and lazily mounts child tools
at runtime via FastMCP `tools/list_changed` — use it when the aggregated fleet would
otherwise blow past a client's tool-count limit.

### Four ways to wire the multiplexer into a client

Like any MCP server, the multiplexer can be consumed four ways. The child
`mcp_config.json` it aggregates is the same file in every case (mount it for the
container options).

=== "1. stdio (client launches it)"

    The common case — the client spawns the multiplexer and reads its consolidated
    tool surface over stdio:

    ```json
    {
      "mcpServers": {
        "mcp-multiplexer": {
          "command": "uvx",
          "args": ["--from", "agent-utilities", "mcp-multiplexer",
                   "--config", "mcp_config.json", "--transport", "stdio"],
          "env": { "MCP_MULTIPLEXER_MODE": "dynamic" }
        }
      }
    }
    ```

=== "2. streamable-http (local process)"

    Run it as a long-lived HTTP process, then point the client at the URL:

    ```bash
    mcp-multiplexer --config mcp_config.json --transport streamable-http --host 0.0.0.0 --port 8005
    curl -s http://localhost:8005/health        # {"status":"OK"}
    ```

    ```json
    { "mcpServers": { "mcp-multiplexer": { "url": "http://localhost:8005/mcp" } } }
    ```

=== "3. Local container / uv"

    Build the image from this repo's `docker/Dockerfile` (or run via `uv`), mounting
    the child config. Launch directly from `mcp_config.json` (swap `docker`→`podman`):

    ```json
    {
      "mcpServers": {
        "mcp-multiplexer": {
          "command": "docker",
          "args": [
            "run", "-i", "--rm",
            "-e", "TRANSPORT=stdio",
            "-e", "MCP_CONFIG=/config/mcp_config.json",
            "-v", "./mcp_config.json:/config/mcp_config.json:ro",
            "agent-utilities:local", "mcp-multiplexer"
          ]
        }
      }
    }
    ```

    Or run a local streamable-http container and connect by `url`
    (`uv run mcp-multiplexer --transport streamable-http --port 8005` for the uv variant).

=== "4. Remote URL (deployed gateway)"

    When the multiplexer is deployed remotely (e.g. as a streamable-http service
    fronted by Caddy on the internal `*.arpa` zone), connect with the `"url"` key — no
    local process or image required:

    ```json
    { "mcpServers": { "mcp-multiplexer": { "url": "http://mcp-gateway.arpa/mcp" } } }
    ```

    Fronting it with Caddy follows the same reverse-proxy → `:8005` pattern as the
    connector fleet (`http://<host>.arpa` → container port).

---

## 5. The centralized REST API (API gateway)

All Knowledge Graph **REST** endpoints are served by the API gateway
(`agent_utilities.server.app`), not by the `graph-os` MCP server. Funnelling
every client (UIs, subagents, ingestion scripts) through one persistent process
eliminates embedded-DB file-lock contention.

Mounted under `/api`:

- `/api/graph/query`, `/api/graph/search`, `/api/graph/write`,
  `/api/graph/ingest`, `/api/graph/analyze`, `/api/graph/orchestrate`,
  `/api/graph/configure` (+ their granular sub-routes)
- `/api/sessions`, `/api/goals`, `/api/tools`
- `POST /cypher` — lock-bypassing direct-Cypher fast path (backpressure + read cache)

The single background KG daemon is role-gated by `KG_DAEMON_ROLE`:

| Role | Behavior |
|------|----------|
| `host` | The gateway process: runs workers, drains the work queue |
| `client` | MCP servers/agents: enqueue work, do not spawn workers |
| `auto` | Pick based on context (default) |

---

## 6. MCP client wiring (`mcp_config.json`)

Point a client (Claude Code, Antigravity, Windsurf, OpenCode) at the servers:

```json
{
  "mcpServers": {
    "graph-os": {
      "command": "graph-os",
      "args": ["--transport", "stdio"],
      "env": { "GRAPH_BACKEND": "epistemic_graph" }
    }
  }
}
```

To consolidate many servers behind one (recommended when a client has a tool-count
limit), point the client at `mcp-multiplexer` and list the children in the same
`mcp_config.json`.

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
# Zero-infra: just the MCP server (the engine is the database, inside the container)
docker compose -f docker/mcp.compose.yml up -d

# Add a Postgres/pg-age mirror for interop/BI/DR:
docker compose -f docker/pg-age.compose.yml up -d
# then enable fan-out:
#   GRAPH_BACKEND=fanout
#   GRAPH_MIRROR_TARGETS=postgresql
#   GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg
```

---

## 8. Production hardening

Set `APP_PROFILE=production` to enable the profile guard
(`agent_utilities.core.profile_guard`). In production it **refuses in-memory
defaults** and requires a durable, shardable engine plus a real event broker:

- a durable engine — the embedded `epistemic_graph` engine (default), or a
  shared/remote engine via `GRAPH_SERVICE_ENDPOINTS`; the pure `memory` backend
  is rejected.
- `a2a_broker` = `kafka`/`nats`, `a2a_storage` = `postgresql`/`redis`.
- `kafka_bootstrap_servers` set (the reactive event ledger needs a real broker).

```bash
export APP_PROFILE=production
export GRAPH_BACKEND=fanout        # engine authority + mirrors for interop/DR
export GRAPH_MIRROR_TARGETS=postgresql
export GRAPH_DB_URI=postgresql://agent:agent@pg-age.internal:5432/agent_kg
export KAFKA_BOOTSTRAP_SERVERS=redpanda-0:9092,redpanda-1:9092
```

The guard raises `ProductionProfileError` listing every offending setting so an
operator can fix them all at once.

---

## 9. Verify a deployment

```bash
# Resolve the active backend (should print the epistemic-graph engine by default)
python -c "from agent_utilities.knowledge_graph.backends import create_backend as c; \
b=c(); print(type(b).__name__)"

# graph-os exposes the standard args
graph-os --help

# Health (HTTP transport)
curl -s localhost:8004/health

# REST via the gateway
curl -s -XPOST localhost:8000/api/graph/query -d '{"cypher":"MATCH (n) RETURN count(n)"}'
```

See also: [Configuration](configuration.md) · [Graph Engine (Authority + Mirrors)](graph_engine.md)
· [Deploying Graph Databases](graph-db-deployment.md).
