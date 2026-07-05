# Consumption Models — Library vs MCP vs REST

agent-utilities can be consumed four ways. They all funnel through the **same
in-process engine** (`_execute_tool` → `IntelligenceGraphEngine`), so capabilities
are identical — you're only choosing the transport and process boundary.

| Model | Entry point | Process boundary | Best for | Trade-off |
|---|---|---|---|---|
| **Library** | `from agent_utilities import create_agent` | In-process (yours) | Building a standalone agent/app in Python | You own the process lifecycle |
| **MCP — stdio** | `uv run graph-os` | Subprocess of the client | Claude Code, Cursor, IDE agents (single-user, spawns its own graph-os) | One client per process; subprocess overhead |
| **MCP — streamable-http** | `uv run graph-os --transport streamable-http` | Standalone server | Remote/containerized agents; many clients | Network + auth to manage |
| **REST gateway** | `python -m agent_utilities` (`PORT`, default `:9000`) | Standalone server | UIs, scripts, non-MCP HTTP clients; one shared KG host | Plain HTTP, not MCP tool-discovery |

## 1. Library (standalone agent)

Direct, lowest-latency, full graph access. You manage the process.

```python
from agent_utilities import create_agent

agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
result = await agent.run("Summarize the latest changes in the codebase")
print(result.content)
```

When to use: you're writing the agent/app yourself and want direct calls.

## 2. MCP over stdio

Give an *existing* agent (Claude Code, Cursor, your own MCP client) the full KG
tool surface. The client spawns `graph-os` as a subprocess.

```json
{
  "mcpServers": {
    "graph-os": {
      "command": "uv",
      "args": ["run", "graph-os"],
      "env": { "AGENT_ID": "local-developer", "WORKSPACE_PATH": "${workspaceFolder}" }
    }
  }
}
```

When to use: IDE / desktop agents; each spawns its own graph-os that fronts the
whole fleet — see [one gateway, every client](#the-fleet-gateway-is-built-into-graph-os).

## 3. MCP over streamable-http

Same tools, but a long-lived HTTP server — ideal for containers and remote
agents. This is how the `*-mcp` connector fleet is deployed.

```bash
uv run graph-os --transport streamable-http --host 0.0.0.0 --port 8004
```

When to use: Docker/Portainer deployment; multiple remote agents sharing one
server. See [Day-0](day0.md).

## 4. REST gateway (`python -m agent_utilities`, default port 9000)

A FastAPI gateway exposing every tool as a REST route (`/api/graph/*`,
`/api/sessions`, `/api/goals`, `/api/ontology/*`, `/api/fleet/*`,
`/api/dashboard/*`, plus Prometheus `/metrics`). It also runs the single
consolidated KG host daemon, so many `KG_DAEMON_ROLE=client` processes share
one engine without file-lock contention. (`graph-os-daemon` is a separate
headless console script — it holds the host lock and drains the task queue but
serves **no** HTTP.)

```bash
python -m agent_utilities          # binds HOST:PORT (defaults 0.0.0.0:9000)
curl -s localhost:9000/api/graph/search -d '{"action":"hybrid","query":"payments"}'
```

Scale it with `GATEWAY_WORKERS` and front it with Caddy/nginx — see
[Scaling the Gateway](../architecture/gateway_scaling.md) and the
[deployment configuration ladder](deployment-configurations.md).

When to use: web UIs (agent-webui consumes this), scripts, the fleet supervisor,
and any non-MCP HTTP client.

> The REST surface and the MCP tool surface are kept at **strict 1:1 parity** by
> a contract test (`tests/unit/test_gateway_mcp_parity.py`) — anything callable
> over MCP is callable over REST and vice-versa.

## The fleet gateway is built into graph-os (one gateway, every client)

There is **no separate `mcp-multiplexer` process anymore** — it is absorbed into
graph-os via the in-process fleet loader (`attach_fleet_loader`). A single
`graph-os` serves its own KG/engine tools **and** lazily fronts the entire `*-mcp`
fleet declared in its `MCP_CONFIG`, mounted on demand via `find_tools` /
`list_catalog` / `load_tools`. Point every client at graph-os — never at a
standalone multiplexer.

### Shared instance vs single-user — same engine, same fleet

The two MCP transports above are just two ways onto the **one** graph-os:

- **Shared instance (streamable-http):** `http://graph-os.arpa/mcp` — one
  long-lived, JWT-gated gateway that many deployed clients share.
- **Single-user (stdio):** each interactive client (Claude Code, opencode, an
  agent) spawns its **own** local `graph-os` process. This is the standard for
  interactive tools because they cannot mint/rotate the gateway's JWT — the local
  process performs the OIDC client-credentials flow itself. It is **not** a second
  KG: `ENGINE_MODE=remote` + `ENGINE_ENDPOINT=tcp://<engine>:9100` point every
  stdio client at the **same shared engine**, and `MCP_CONFIG` at the **same
  canonical fleet list**. A single-user shim and the shared gateway resolve to
  identical data.

See the [ecosystem map](../ecosystem.md) for the connector fleet, and
[MCP auth](../architecture/mcp_auth.md) for the inbound-JWT / outbound-client-credentials
wiring.
