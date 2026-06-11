# Scaling the Gateway

**CONCEPT:OS-5.23 — Gateway Middle-Tier Hardening** (Python-tier Prometheus
metrics, per-tenant rate limiting, engine circuit breaker, multi-worker
readiness).

The API gateway (`agent_utilities.server` + `gateway/graph_api.py`)
historically ran as exactly one process with one event loop. This page
documents how to run more than one worker/replica, what is safe about that
today, and which state remains per-process.

## Worker model

```
GATEWAY_WORKERS=1   (default)  one process, one event loop, in-process KG daemon
GATEWAY_WORKERS=N   (N>1)      pre-forked worker pool on ONE shared listen socket
```

With `GATEWAY_WORKERS>1`, `_run_agent_server` binds the listen socket once,
then forks **before building the app**, so every worker constructs its own
FastAPI app, engine client connections and daemon role. The parent is worker 0
and reaps the children when its server exits. (uvicorn's own `workers=` flag
requires an import-string app, which the dynamically-built gateway app cannot
provide — hence the explicit pre-fork.) The flag is ignored under pytest and
with the terminal UI.

You can equally scale with **N container replicas** (each `GATEWAY_WORKERS=1`)
behind a load balancer — every statement below about "per-process" state
applies the same way.

## KG host daemon: exactly one, by construction

The consolidated KG host daemon (queue drain, graph writer, task workers,
maintenance/golden-loop ticks) is serialized by the **advisory `flock`
host-lock** (`knowledge_graph/core/host_lock.py`): each worker resolves its
role independently after the fork, the first to acquire the lock becomes
`host`, and every other worker self-heals to `client` (no daemon threads, lets
the host drain the durable queue). The lock auto-releases when the holder
dies, so a crashed host worker never blocks a restart. This was verified
against the fork model: role resolution happens **per-child, after fork**, so
the inherited-lock-fd hazard does not arise.

Consequence: daemon ticks (enrichment, hygiene, golden loop, …) run in ONE
worker only — that is the intended topology, identical to running the
gateway next to MCP servers on one machine.

## What is per-process (deliberate, documented)

| State | Behaviour across workers/replicas |
| --- | --- |
| Prometheus metrics registry | Per-process. A scrape of `/metrics` through the shared socket samples ONE worker; aggregate in Prometheus (scrape each replica) or run 1 worker/container. |
| Rate-limit token buckets | Per-process: `GATEWAY_RATE_LIMIT` is effectively multiplied by the worker count. Precise distributed limiting belongs to the state-externalization track. |
| Engine circuit breaker | Per-process per endpoint. Each worker discovers a dead engine independently (≤ threshold extra probes per worker). |
| Dashboard `Aggregator` cache | 10s TTL read cache + thread pool — bounded divergence, safe to duplicate. |
| Dashboard layout (`ConfigManager`) | NOT divergent: reads/writes go to the shared YAML file (XDG config dir) on every request. |
| Durable execution / sessions / engine task queue | Externalized already (SQLite/engine-side) — workers coordinate through the KG host. |

Nothing in `gateway/api.py` mutates module state after startup except
`save_layout`, which persists straight to disk.

## Python-tier metrics

Mounted by `register_graph_routes` (so the gateway **and** the agent-webui
backend both get it). Naming mirrors the Rust engine's `epistemic_graph_*`
series:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `agent_utilities_gateway_requests_total` | `route`, `method`, `status` | Request count. `route` is always a route TEMPLATE (`/api/things/{id}`) — unmatched requests collapse into `unmatched`. |
| `agent_utilities_gateway_request_duration_seconds` | `route` | Latency histogram. |
| `agent_utilities_gateway_in_flight_requests` | — | Gauge of in-flight requests. |
| `agent_utilities_gateway_rate_limited_total` | `tenant` | 429s from the token-bucket limiter. |
| `agent_utilities_gateway_engine_requests_total` | `op`, `outcome` | Engine client calls (`ok` / `connection_error` / `error` / `short_circuited`). |
| `agent_utilities_gateway_engine_breaker_state` | `endpoint` | 0=closed, 1=half-open, 2=open. |

`GET /metrics` is exempt from the identity middleware (scrapers cannot mint
JWTs) and from rate limiting. `prometheus_client` is the optional `metrics`
extra; absent, everything degrades to a no-op and `/metrics` returns a
placeholder. Toggle with `GATEWAY_METRICS` (default on).

## Per-tenant rate limiting

`GATEWAY_RATE_LIMIT` (req/s sustained, default 0 = off) +
`GATEWAY_RATE_BURST` (default 2× rate). The ASGI limiter sits **inside** the
OS-5.14 identity middleware, so the bucket key uses the server-minted
`ActorContext`: tenant → authenticated actor id → client IP. Rejections are
`429` with `Retry-After` and a JSON body. Health routes and `/metrics` are
exempt.

## Engine circuit breaker

Every `GraphComputeEngine` call is guarded by a shared per-endpoint breaker
(`knowledge_graph/core/engine_breaker.py`): `ENGINE_BREAKER_THRESHOLD`
(default 5) consecutive connect/timeout failures open the circuit;
`ENGINE_BREAKER_COOLDOWN` (default 15s) later a single half-open probe heals
or re-opens it. While open, callers get the fast, typed
`EngineCircuitOpenError` (a `ConnectionError` subclass) instead of hammering a
dead socket. Application-level errors (bad Cypher, missing node) never trip
the breaker. `ENGINE_BREAKER_THRESHOLD=0` disables tripping.

## Flags

| Flag | Default | What it sets |
| --- | --- | --- |
| `GATEWAY_METRICS` | `true` | Python-tier Prometheus middleware + `GET /metrics` |
| `GATEWAY_RATE_LIMIT` | `0` (off) | Per-tenant sustained req/s |
| `GATEWAY_RATE_BURST` | `0` (→ 2× rate) | Token-bucket capacity |
| `GATEWAY_WORKERS` | `1` | Pre-forked gateway worker processes |
| `ENGINE_BREAKER_THRESHOLD` | `5` | Failures before the engine circuit opens (0 = off) |
| `ENGINE_BREAKER_COOLDOWN` | `15` | Seconds before the half-open probe |

All are typed fields on `AgentConfig` (`core/config.py`) per the
configuration-discipline rule.
