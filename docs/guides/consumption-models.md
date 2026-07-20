# Consumption Models — Library vs MCP vs REST

agent-utilities can be consumed four ways. They all reach the same GraphOS
execution surface and authoritative `epistemic-graph` engine, so capabilities
are identical — you're choosing the client transport and process boundary. The
Rust engine remains out of process over its authenticated local or remote wire
transport; none of these modes embeds it in Python.

| Model | Entry point | Process boundary | Best for | Trade-off |
|---|---|---|---|---|
| **Library** | `from agent_utilities import create_agent` | In-process (yours) | Building a standalone agent/app in Python | You own the process lifecycle |
| **MCP — stdio** | `graph-os` | Subprocess of the client | Claude Code, Cursor, IDE agents (single-user, spawns its own graph-os) | One client per process; subprocess overhead |
| **MCP — streamable-http** | `graph-os --transport streamable-http` | Standalone server | Remote/containerized agents; many clients | Network + auth to manage |
| **REST gateway** | `python -m agent_utilities` (`PORT`, default `:9000`) | Standalone server | UIs, scripts, non-MCP HTTP clients; one shared KG host | HTTP API rather than MCP discovery; remote binds require identity and TLS |

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
      "command": "graph-os",
      "args": ["--transport", "stdio"]
    }
  }
}
```

The launcher remains machine-neutral: no host path, endpoint, certificate, or
credential is committed.
With `DEPLOYMENT_PROFILE=tiny`, no `GRAPH_SERVICE_ENDPOINTS`, and neither
`KG_AUTH_TOKEN_REF` nor `KG_IDENTITY_OAUTH2`, this exact GraphOS stdio boundary
uses a neutral in-memory bootstrap JWT as a one-time proof. Its key and token are
destroyed before a process-lifetime session is returned, and no personal, host,
endpoint, filesystem, credential, or proof data is persisted. Run
`agent-utilities-doctor --only graph_identity auth` before launch.

Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external process-identity source. The OAuth2
shape must contain a runtime-resolved client-secret reference. Never put the
resulting token or raw client secret in this launcher block. Failure never falls
back to the local authority. External stdio authority uses a shared in-memory
expiry-only lease. Renewals must preserve the original identity and capabilities;
drift is rejected, failed renewal does not extend the lease, and tool plus
background graph work fail closed at expiry while renewal retries.

When to use: IDE / desktop agents; each spawns its own graph-os that fronts the
whole fleet — see [one gateway, every client](#fleet-gateway).

## 3. MCP over streamable-http

Same tools, but a long-lived HTTP server — ideal for containers and remote
agents. This is how the `*-mcp` connector fleet is deployed.

```bash
graph-os --transport streamable-http --host 127.0.0.1 --port 8004
```

This is a loopback example. For Docker/Portainer or multiple remote agents,
configure JWT/OIDC authentication and trusted TLS termination before binding a
non-loopback address. See [Day-0](day0.md).

## 4. REST gateway (`python -m agent_utilities`, default port 9000)

A FastAPI gateway exposing every tool as a REST route (`/api/graph/*`,
`/api/sessions`, `/api/goals`, `/api/ontology/*`, `/api/fleet/*`,
`/api/dashboard/*`, plus Prometheus `/metrics`). It also runs the single
consolidated KG host daemon, so many `KG_DAEMON_ROLE=client` processes share
one engine without file-lock contention. (`graph-os-daemon` is a separate
headless console script — it holds the host lock and drains the task queue but
serves **no** HTTP.)

```bash
python -m agent_utilities --host 127.0.0.1 --port 9000
curl -s localhost:9000/api/graph/search -d '{"action":"hybrid","query":"payments"}'
```

Configure JWT/OIDC authentication, allowed hosts/origins, and trusted TLS before a
non-loopback bind.

Scale it with `GATEWAY_WORKERS` and front it with Caddy/nginx — see
[Scaling the Gateway](../architecture/gateway_scaling.md) and the
[deployment configuration ladder](deployment-configurations.md).

When to use: web UIs (agent-webui consumes this), scripts, the fleet supervisor,
and any non-MCP HTTP client.

> The REST surface and the MCP tool surface are kept at **strict 1:1 parity** by
> a contract test (`tests/unit/test_gateway_mcp_parity.py`) — anything callable
> over MCP is callable over REST and vice-versa.

## The fleet gateway is built into graph-os (one gateway, every client) { #fleet-gateway }

GraphOS owns the in-process fleet loader. A single `graph-os` serves its own
KG/engine tools **and** lazily fronts the entire `*-mcp`
fleet declared in its `MCP_CONFIG`, mounted on demand via `find_tools` /
`list_catalog` / `load_tools`. Point every client at graph-os — never at a
second fleet-gateway process.

### Shared instance vs single-user — same engine, same fleet

The two MCP transports above are just two ways onto the **one** graph-os:

- **Shared instance (streamable-http):** `https://graph-os.example.test/mcp` — one
  long-lived, JWT-gated gateway that many deployed clients share.
- **Single-user (stdio):** each interactive client (Claude Code, opencode, an
  agent) spawns its **own** local `graph-os` process. This is the standard for
  interactive tools. A tiny packaged-local instance uses the neutral ephemeral
  authority described above. A stdio process pointed at a shared engine has an
  explicit `GRAPH_SERVICE_ENDPOINTS`, so it must use one external process-identity
  source (normally the OAuth2 client-credentials flow). It is **not** a second KG:
  `GRAPH_SERVICE_ENDPOINTS=tls://<engine>:9100` plus a
  runtime `ENGINE_TLS_PROFILE_REF` point every stdio client at the **same shared
  engine**, and `MCP_CONFIG` at the **same
  canonical fleet list**. A single-user shim and the shared gateway resolve to
  identical data.

See the [ecosystem map](../ecosystem.md) for the connector fleet, and
[MCP auth](../architecture/mcp_auth.md) for the inbound-JWT / outbound-client-credentials
wiring.
