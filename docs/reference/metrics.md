# Metrics Reference

Consolidated catalogue of every Prometheus metric series the agent-utilities
Python tier emits. All series below are registered in ONE module —
[`agent_utilities/observability/gateway_metrics.py`](https://github.com/Knuckles-Team/agent-utilities/blob/main/agent_utilities/observability/gateway_metrics.py)
(CONCEPT:OS-5.23) — and incremented by the subsystem modules listed per table.
Names, types, labels, and meanings are verified against that module's
registrations and docstrings.

## How to scrape

- The gateway (and the agent-webui backend — both mount `register_graph_routes`)
  exposes `GET /metrics` in Prometheus exposition format. The endpoint is exempt
  from the identity middleware and rate limiting (scrapers cannot mint JWTs).
- `prometheus_client` is the optional `metrics` extra. Without it every series
  degrades to a shared no-op (recording costs ~nothing) and `/metrics` returns a
  self-describing placeholder with HTTP 200.
- The whole middleware + endpoint is toggled by `GATEWAY_METRICS` (default on);
  see [../architecture/configuration.md](../architecture/configuration.md).
- Metrics live in the default per-process registry. With `GATEWAY_WORKERS>1`
  (pre-forked workers on one listen socket) a scrape samples ONE worker —
  aggregate across replicas in Prometheus or run one worker per container
  (see [../architecture/gateway_scaling.md](../architecture/gateway_scaling.md)).
- Request-duration histogram buckets: 5ms to 60s
  (`0.005 ... 30, 60`).
- Worked scrape/dashboard setup: [../examples/observability.md](../examples/observability.md).

Cardinality discipline: the `route` label is always a route TEMPLATE
(`/api/graph/{name}`), never a raw path — unmatched requests (404/405) collapse
into the single `unmatched` bucket so internet scanners cannot mint series.
`endpoint` cardinality is bounded by the configured `GRAPH_SERVICE_ENDPOINTS`
list; `server` by the children declared in `mcp_config.json`.

## Gateway HTTP (CONCEPT:OS-5.23)

Recorded by `GatewayMetricsMiddleware` (pure ASGI, mounted outermost by
`gateway/graph_api.py::register_graph_routes` so 401/429 rejections are counted
too; `/metrics` itself is not instrumented).

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_gateway_requests_total` | Counter | `route`, `method`, `status` | Gateway HTTP requests by route template, method, and status code | `observability/gateway_metrics.py` (middleware) | OS-5.23 |
| `agent_utilities_gateway_request_duration_seconds` | Histogram | `route` | Request duration by route template | `observability/gateway_metrics.py` (middleware) | OS-5.23 |
| `agent_utilities_gateway_in_flight_requests` | Gauge | — | Requests currently being handled by this process | `observability/gateway_metrics.py` (middleware) | OS-5.23 |
| `agent_utilities_gateway_rate_limited_total` | Counter | `tenant` | Requests rejected (429) by the per-tenant token-bucket rate limiter | `gateway/rate_limit.py` | OS-5.23 |

## Engine client & circuit breaker (CONCEPT:OS-5.23)

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_gateway_engine_requests_total` | Counter | `op`, `outcome` | epistemic-graph engine client calls by operation and outcome (`ok` \| `connection_error` \| `error` \| `short_circuited`) | `knowledge_graph/core/engine_breaker.py` | OS-5.23 |
| `agent_utilities_gateway_engine_breaker_state` | Gauge | `endpoint` | Engine circuit-breaker state per endpoint (0=closed, 1=half-open, 2=open) | `knowledge_graph/core/engine_breaker.py` | OS-5.23 |

## Engine shard topology (CONCEPT:KG-2.58 / OS-5.28)

One series per configured `GRAPH_SERVICE_ENDPOINTS` entry. The reachability
gauge is refreshed on every real client connect attempt and by the daemon's
`shard_topology_status` probe (surfaced through `gateway/api.py`); the
per-shard counter splits the engine-call outcomes so a hot or failing shard is
visible at a glance. The existing breaker gauge above is already per-endpoint,
so each shard gets its own circuit breaker.

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_engine_shard_up` | Gauge | `endpoint` | Per-shard engine reachability (1=reachable, 0=unreachable) | `knowledge_graph/core/shard_topology.py` | KG-2.58 / OS-5.28 |
| `agent_utilities_engine_shard_requests_total` | Counter | `endpoint`, `outcome` | Engine client calls per shard endpoint and outcome (`ok` \| `connection_error` \| `error` \| `short_circuited`) | `knowledge_graph/core/engine_breaker.py` | KG-2.58 / OS-5.28 |

## KG ingest queue backpressure (CONCEPT:KG-2.57)

Sampled by the KG maintenance scheduler on the leader host. Depth is uniform
across queue backends (sqlite/postgres = row count, kafka = kg-ingest
consumer-group lag); the lag series exists separately so Kafka dashboards and
alerts read naturally. These two series also feed the autoscaler's zero-infra
`LocalMetricsProvider` signals (`queue_depth`, `consumer_lag`) when
`SCALING_PROMETHEUS_URL` is unset (CONCEPT:OS-5.29,
`orchestration/scaling_signals.py`).

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_kg_ingest_queue_depth` | Gauge | `backend` | Pending KG ingest tasks in the selected durable task queue | `knowledge_graph/core/engine_tasks.py` | KG-2.57 |
| `agent_utilities_kg_ingest_consumer_lag` | Gauge | `topic`, `group` | Total kg-ingest consumer-group lag (unconsumed messages) per topic | `knowledge_graph/core/engine_tasks.py` | KG-2.57 |

## MCP multiplexer child resilience (CONCEPT:ECO-4.34)

One series per aggregated child server (~50, bounded by `mcp_config.json`).
The multiplexer runs standalone; like every series here these degrade to
no-ops when the `metrics` extra is absent.

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_mcp_child_calls_total` | Counter | `server`, `outcome` | Multiplexer tool calls per child and outcome (`ok` \| `error` \| `transport_error` \| `timeout` \| `busy` \| `unavailable` \| `short_circuited`) | `mcp/child_resilience.py` | ECO-4.34 |
| `agent_utilities_mcp_child_breaker_state` | Gauge | `server` | Per-child circuit-breaker state (0=closed, 1=half-open, 2=open) | `mcp/child_resilience.py` | ECO-4.34 |
| `agent_utilities_mcp_child_restarts_total` | Counter | `server` | Automatic restarts of crashed child servers | `mcp/child_resilience.py` | ECO-4.34 |
| `agent_utilities_mcp_child_queue_depth` | Gauge | `server` | Tool calls queued behind a child's concurrency limit right now | `mcp/child_resilience.py` | ECO-4.34 |

## Queue-driven agent dispatch (CONCEPT:ORCH-1.45)

Sampled by the dispatch workers on their fleet-registry heartbeat tick. Depth
is uniform across queue transports (kafka = agent-dispatch consumer-group lag,
postgres/sqlite = row count).

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_dispatch_queue_depth` | Gauge | `backend` | Unclaimed dispatched agent turns in the `agent_turns` queue | `orchestration/agent_dispatch_worker.py` | ORCH-1.45 |
| `agent_utilities_dispatch_turns_total` | Counter | `outcome` | Dispatched agent turns processed by this worker process (`completed` \| `failed` \| `skipped` \| `expired`) | `orchestration/agent_dispatch_worker.py` | ORCH-1.45 |
| `agent_utilities_dispatch_workers` | Gauge | — | Live agent-dispatch workers (fresh heartbeats in the fleet registry) | `orchestration/agent_dispatch_worker.py` | ORCH-1.45 |

## Agent communication bus (CONCEPT:ECO-4.87)

The agent-to-agent bus (`AgentBus`, ECO-4.84). Participant gauges are sampled on the
`status`/health read; message and dispatch counters and the send-duration histogram are
emitted on the `send`/`dispatch` path. Labels are small enums, so cardinality stays flat.

| Name | Type | Labels | Meaning | Emitted by (module) | Since (concept id) |
|---|---|---|---|---|---|
| `agent_utilities_bus_participants` | Gauge | `status` | Registered participants by computed presence (`online` \| `offline`) | `messaging/bus.py` | ECO-4.87 |
| `agent_utilities_bus_messages_total` | Counter | `kind`, `outcome` | Bus messages by kind (`direct` \| `topic`) and outcome (`delivered` \| `denied` \| `no_recipient`) | `messaging/bus.py` | ECO-4.87 |
| `agent_utilities_bus_send_seconds` | Histogram | — | Server-side latency of one `AgentBus.send` (gate + per-recipient durable write) | `messaging/bus.py` | ECO-4.87 |
| `agent_utilities_bus_dispatch_total` | Counter | `outcome` | Message→fleet-work dispatches (`submitted` \| `denied` \| `failed`) | `messaging/bus.py` | ECO-4.87 |

Load-test these with `scripts/bench_bus.py` against a live hub; the modeled expectation
(participants/hub, msgs/s/connection) is in `docs/scaling/capacity_model.py`
(`bus_plan_for`). The Grafana dashboard is `agent-bus.json` (generated by
`scripts/gen_grafana_dashboards.py`).

## Fleet autoscaler (CONCEPT:OS-5.29)

The autoscaler registers no metric series of its own. It CONSUMES signals:
either this process's own gauges above via the zero-infra
`LocalMetricsProvider` (`queue_depth` →
`agent_utilities_kg_ingest_queue_depth`, `consumer_lag` →
`agent_utilities_kg_ingest_consumer_lag`), or instant Prometheus HTTP queries
(`sum(...)` over the same series) when `SCALING_PROMETHEUS_URL` is set —
see `orchestration/scaling_signals.py` and `orchestration/fleet_autoscaler.py`.

## Rust engine series (`epistemic_graph_*`)

The epistemic-graph engine exposes its own native `epistemic_graph_*`
Prometheus series — these live in the engine repository, not in
agent-utilities, and their exact names are defined by the engine's own
documentation. What this repo wires and documents:

- Each engine process takes a `--metrics-addr` flag and serves its own
  `/metrics` listener — one scrape target per shard
  ([../architecture/engine_sharding.md](../architecture/engine_sharding.md)).
- The runnable 3-shard example
  ([`docker/engine-shards.compose.yml`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docker/engine-shards.compose.yml))
  publishes RPC on `9101`-`9103` and the Prometheus listeners on
  `9111`-`9113` (`--metrics-addr 0.0.0.0:911N`).
- The Python-tier `agent_utilities_*` prefix deliberately mirrors the engine's
  naming style so dashboards read coherently
  ([../architecture/gateway_scaling.md](../architecture/gateway_scaling.md)).

## Series count

21 Python-tier series total: 4 gateway HTTP, 2 engine client/breaker, 2 shard
topology, 2 ingest queue, 4 MCP child, 3 dispatch, 4 agent bus — all registered in
`observability/gateway_metrics.py` and verified against code in this pass.
