# Consumption Models — Library vs MCP vs REST

agent-utilities can be consumed four ways. They all funnel through the **same
in-process engine** (`_execute_tool` → `IntelligenceGraphEngine`), so capabilities
are identical — you're only choosing the transport and process boundary.

| Model | Entry point | Process boundary | Best for | Trade-off |
|---|---|---|---|---|
| **Library** | `from agent_utilities import create_agent` | In-process (yours) | Building a standalone agent/app in Python | You own the process lifecycle |
| **MCP — stdio** | `uv run graph-os` | Subprocess of the client | Claude Code, Cursor, IDE agents, the multiplexer | One client per process; subprocess overhead |
| **MCP — streamable-http** | `uv run graph-os --transport streamable-http` | Standalone server | Remote/containerized agents; many clients | Network + auth to manage |
| **REST gateway** | `uv run graph-os-daemon` (`:8100`) | Standalone server | UIs, scripts, non-MCP HTTP clients; one shared KG host | Plain HTTP, not MCP tool-discovery |

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

When to use: IDE / desktop agents; aggregating many MCP servers under the
[multiplexer](#bonus-the-multiplexer).

## 3. MCP over streamable-http

Same tools, but a long-lived HTTP server — ideal for containers and remote
agents. This is how the `*-mcp` connector fleet is deployed.

```bash
uv run graph-os --transport streamable-http --host 0.0.0.0 --port 8004
```

When to use: Docker/Portainer deployment; multiple remote agents sharing one
server. See [Day-0](day0.md).

## 4. REST gateway (`graph-os-daemon`, port 8100)

A FastAPI gateway exposing every tool as a REST route (`/api/graph/*`,
`/api/sessions`, `/api/goals`, `/api/ontology/*`, `/api/fleet/*`). It also runs
the single consolidated KG host daemon, so many `KG_DAEMON_ROLE=client` processes
share one engine without file-lock contention.

```bash
uv run graph-os-daemon
curl -s localhost:8100/api/graph/search -d '{"action":"hybrid","query":"payments"}'
```

When to use: web UIs (agent-webui consumes this), scripts, the fleet supervisor,
and any non-MCP HTTP client.

> The REST surface and the MCP tool surface are kept at **strict 1:1 parity** by
> a contract test (`tests/unit/test_gateway_mcp_parity.py`) — anything callable
> over MCP is callable over REST and vice-versa.

## Bonus: the multiplexer

`mcp-multiplexer` aggregates many child MCP servers (graph-os + the whole
`*-mcp` fleet) into one unified MCP endpoint, so an agent sees every tool through
a single connection:

```bash
mcp-multiplexer --config ./mcp_config.json --transport stdio
```

See the [ecosystem map](../ecosystem.md) for the connector fleet it federates.
