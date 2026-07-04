# Scalable Frontends — one shared backend, many thin instances

> **The standard.** A frontend is a *thin client* over a **shared** agent-utilities
> backend. The heavy parts — the Rust epistemic-graph engine, the embedding model,
> the consolidated background daemon — live in **one** backend (or a sharded set),
> never co-located per frontend instance. This is what lets you run many frontend
> instances cheaply (the agent-terminal-ui pattern): one node hosts the engine,
> dozens of light frontends fan out against it.

The anti-pattern is the opposite: a frontend that imports the engine in-process and
forces itself to be the KG host. Then every instance carries a full engine + daemon,
RSS balloons (hundreds of MB to >1 GB each), and you cannot scale horizontally.

## The five-point checklist

A frontend is "scale-many-instances ready" when:

1. **Talks to a shared backend over the wire** — HTTP to the gateway (`AGENT_URL`)
   and/or the engine socket (`GRAPH_SERVICE_ENDPOINTS`), never an in-process engine
   it owns exclusively.
2. **Never forces itself to be the KG host** — runs as `KG_DAEMON_ROLE=client` so
   exactly one shared host runs the consolidated daemon (queue drain, graph writer,
   workers, scheduler, file-watch); clients reach it over the socket.
3. **Lazy heavy imports** — any `agent_utilities.knowledge_graph` / engine import is
   inside the function that uses it, never at module top, so importing the frontend
   does not eagerly load the engine.
4. **Slim runtime image** — multi-stage build, runtime deps only, no test/dev
   extras, non-root, no caches. (See the `agent-terminal-ui/Dockerfile` template.)
5. **A lightweight/headless mode** where the surface supports it (CLI/server), for
   many concurrent non-interactive instances against one backend.

## Shared env knobs (the same on every thin instance)

| Variable | Thin-instance value | Effect |
|---|---|---|
| `KG_DAEMON_ROLE` | `client` | Do not run the in-process host daemon; reach the shared host. |
| `GRAPH_SERVICE_ENDPOINTS` | `unix:///run/eg-0.sock,…` (or host:port) | Where the shared engine (or shards, AU-KG.sharding.tenant-partitioned-sharding-hrw) lives. |
| `AGENT_URL` / gateway URL | `http://agent-utilities:8000` | The shared gateway the UI calls over HTTP. |
| `STATE_DB_URI` | shared Postgres DSN | So client instances share sessions/queues/checkpoints (AU-OS.state.unified-durable-state-externalization–18). |

One node runs the **host** (`KG_DAEMON_ROLE=host` or unset + the engine), everything
else runs **client**. Pair with the [Enterprise Enablement Runbook](enterprise-enablement-runbook.md)
when turning the shared backend's scale-out flags on.

## Per-surface posture

### agent-terminal-ui — the reference (fully decoupled)
The gold standard: **no `agent_utilities` dependency at all** (deps are
textual/httpx/rich/agent-client-protocol/pyyaml). It speaks only HTTP to `AGENT_URL`;
a dependency-free vendored `GoalSpec` parser (`agent_terminal_ui/goal.py`) exists
specifically to avoid importing the backend. A `--headless` mode (`headless.py`,
~30 MB, no Textual widget tree) runs many non-interactive instances per node, and a
runtime-only `Dockerfile` ships the frontend without the backend. ~30–50 MB/instance.

### agent-webui — thin via client role
The React SPA scales infinitely in the browser; the **Python API server** is what you
scale horizontally. It needs `agent-utilities[agent,graph]` (the engine *client* + the
canonical gateway routes), so it is not as small as terminal-ui — but it is thin when
run as **`KG_DAEMON_ROLE=client`**: the server skips the in-process host daemon
(`server.py` `_start_kg_host_daemon`) and reaches a shared host over the engine socket.
Run **one** host instance (or a sharded backend) and **many** client-role API instances
behind a load balancer. The heavy KG imports in `api_extensions.py` are used only inside
their handlers (no module-level engine instantiation), so importing the server is light.

### geniusbot — desktop cockpit (not horizontally scaled)
A PySide6 desktop app is one-per-user-desktop, not an instance you fan out in
containers (it needs a display). Its scaling concern is therefore *startup weight*, and
it already follows the standard: the `BackendAdapter`
(`geniusbot/services/backend_adapter.py`) is **gateway-first** — it routes through the
HTTP `GatewayClient` and only lazy-imports the `agent_utilities` engine inside the one
method that needs a local fallback, so the UI never eagerly loads the backend. The
`gateway_client` SDK it consumes (AU-ECO.interop.gateway-client-sdk) imports only `httpx` + `agent_utilities.http`,
not the engine.

## Why webui can't be as small as terminal-ui (and that's fine)

terminal-ui is a pure protocol client — it never needs backend code, so it drops the
`agent_utilities` dependency entirely. webui mounts the *canonical* gateway routes and
uses the engine *client*, so it must ship `agent_utilities`. The win there is not a
tiny image; it is **runtime memory + host topology**: a client-role webui does not load
the embedding model, does not host the engine, and does not run the daemon — so N of
them share one host instead of standing up N engines. Same principle (one shared
backend, many thin frontends), different floor.
