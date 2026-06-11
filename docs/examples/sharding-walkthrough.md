# Worked Example: 3-Shard Engine Walkthrough

## What this demonstrates

Running three epistemic-graph engine shards from the shipped compose file,
pointing every client at them with `GRAPH_SERVICE_ENDPOINTS`, and watching
tenant-partitioned routing work: `tenant → named graph → HRW → shard`
(CONCEPT:KG-2.58), with topology visibility (CONCEPT:OS-5.28) and the
fail-loud failure semantics.

Deep dive: [engine_sharding.md](../architecture/engine_sharding.md).

## Prerequisites (ladder rung)

The "sharded engine tier" rung of the
[deployment ladder](../guides/deployment-configurations.md): Docker (or three
hand-started `epistemic-graph-server` processes), and one shared
`GRAPH_SERVICE_AUTH_SECRET` across all shards and all clients.

## 1. Start three shards

The repo ships a runnable example at `docker/engine-shards.compose.yml` —
three independent engine processes, each with its own TCP port, persist dir
and Prometheus metrics listener, all sharing ONE HMAC secret:

```yaml
# docker/engine-shards.compose.yml (fragment)
services:
  engine-shard-1:
    command:
      [
        "epistemic-graph-server",
        "--tcp-addr", "0.0.0.0:9101",
        "--persist-dir", "/data/shard-1",
        "--metrics-addr", "0.0.0.0:9111",
      ]
    volumes:
      - engine-shard-1:/data/shard-1
    ports:
      - "9101:9101" # RPC (MessagePack)
      - "9111:9111" # Prometheus /metrics
  # engine-shard-2: ports 9102 (RPC) / 9112 (metrics), volume engine-shard-2
  # engine-shard-3: ports 9103 (RPC) / 9113 (metrics), volume engine-shard-3
```

```bash
export GRAPH_SERVICE_AUTH_SECRET="$(openssl rand -hex 32)"   # same everywhere
docker compose -f docker/engine-shards.compose.yml up -d
```

The secret is deliberately not defaulted in the compose file — leave it unset
and the engine binary refuses to start (fail-loud, CONCEPT:OS-5.14).

## 2. Point every client at the full list

`GRAPH_SERVICE_ENDPOINTS` accepts a comma-separated string **or** a JSON
list (a `before`-validator on `AgentConfig` coerces both through the
canonical `to_list`):

```bash
export GRAPH_SERVICE_ENDPOINTS="tcp://localhost:9101,tcp://localhost:9102,tcp://localhost:9103"
# equivalently:
export GRAPH_SERVICE_ENDPOINTS='["tcp://localhost:9101","tcp://localhost:9102","tcp://localhost:9103"]'
```

Rules (verified in `core/config.py` + `knowledge_graph/core/shard_topology.py`):

- One entry behaves exactly like the single socket/tcp_addr path (zero-infra
  default preserved); **2+ entries enable sharding**. When set, the list
  overrides `GRAPH_SERVICE_SOCKET` / `GRAPH_SERVICE_TCP_ADDR`.
- Endpoint strings are hashed **verbatim** as both the HRW input and the
  connect target — configure every client with the *identical* strings
  (order does not matter; HRW is order-independent), with explicit
  `unix://` / `tcp://` schemes.

## 3. Tenant → graph → shard

Two pieces compose the routing:

1. **Tenant → named graph** (`shard_topology.tenant_graph_name`, also
   exposed as `KnowledgeGraph.tenant_graph()`): when a caller does not target
   an explicit graph and the ambient `ActorContext` carries a tenant, the
   default graph (`KG_DEFAULT_GRAPH`, default `__bus__`) maps to
   `tenant__<slugified-tenant>__<base>`. No tenant ⇒ the base graph,
   unchanged — single-tenant deployments are byte-for-byte unaffected.
2. **Graph → shard** (HRW rendezvous hashing): the sync client path
   delegates to the *same* `epistemic_graph.pool.ShardRouter` implementation
   async pool users call (`_get_shard_endpoint`: per endpoint, score =
   MD5(`"{endpoint}-{graph_name}"`), highest score wins), so sync and async
   callers can never disagree on placement.

Observed routing in this tree (real `shard_endpoint_for` output):

```text
'acme'    -> graph 'tenant__acme____bus__'    -> tcp://localhost:9103
'globex'  -> graph 'tenant__globex____bus__'  -> tcp://localhost:9101
'initech' -> graph 'tenant__initech____bus__' -> tcp://localhost:9102
None      -> graph '__bus__'                  -> tcp://localhost:9103
```

Reproduce:

```python
from agent_utilities.knowledge_graph.core.shard_topology import (
    shard_endpoint_for, tenant_graph_name,
)
eps = ["tcp://localhost:9101", "tcp://localhost:9102", "tcp://localhost:9103"]
g = tenant_graph_name("acme", "__bus__")     # 'tenant__acme____bus__'
print(shard_endpoint_for(g, eps))            # 'tcp://localhost:9103'
```

Resolution order for the effective routing graph
(`resolve_routing_graph`): explicit non-default graph name → ambient tenant's
graph → `KG_DEFAULT_GRAPH`.

## 4. Inspect the topology

```bash
curl -sS http://localhost:8000/api/dashboard/daemon/shards
```

(The handler is `GET /daemon/shards` on the gateway dashboard router, which
`agent_utilities/server/app.py` mounts under `/api/dashboard`.) Response shape
(verified in `shard_topology_status()`):

```json
{
  "mode": "sharded",
  "default_graph": "__bus__",
  "endpoints": [
    {"endpoint": "tcp://localhost:9101", "local": false, "reachable": true, "breaker": "closed"},
    {"endpoint": "tcp://localhost:9102", "local": false, "reachable": true, "breaker": "closed"},
    {"endpoint": "tcp://localhost:9103", "local": false, "reachable": false, "breaker": "open"}
  ]
}
```

The probe is a transport-level connect (no authenticated RPC per scrape) and
refreshes the `agent_utilities_engine_shard_up{endpoint}` gauge as a side
effect. `breaker` is the per-endpoint circuit-breaker state
(`closed` / `half_open` / `open`).

Prometheus series for the shard tier (registered in
`agent_utilities/observability/gateway_metrics.py`):

- `agent_utilities_engine_shard_up{endpoint}` — 1/0, refreshed on every real
  client connect attempt and by the status probe;
- `agent_utilities_engine_shard_requests_total{endpoint,outcome}` — engine
  calls per shard (`ok | connection_error | error | short_circuited`);
- `agent_utilities_gateway_engine_breaker_state{endpoint}` — already
  per-endpoint, so each shard gets its own breaker series for free;
- each shard's own native `epistemic_graph_*` series on its `--metrics-addr`
  listener (9111/9112/9113 here) — see the
  [observability example](observability.md).

## 5. Failure semantics (what breaks, and how loudly)

Stop one shard and touch a graph it owns:

```bash
docker stop engine-shard-3
```

The client raises a `ConnectionError` naming the shard, the graph, and the
remediation (from `knowledge_graph/core/graph_compute.py`):

```text
ConnectionError: Configured engine shard 'tcp://localhost:9103' (owner of
graph 'tenant__acme____bus__' by HRW over GRAPH_SERVICE_ENDPOINTS) is
unreachable: [Errno 111] Connection refused. Start that shard's
epistemic-graph-server (or remove it from GRAPH_SERVICE_ENDPOINTS — moving a
graph between shards requires a manual snapshot export/import). Autostart
applies only to the local unix:// endpoint, never to remote shards.
```

Why fail-loud, not fail-over:

- **`EPISTEMIC_GRAPH_AUTOSTART=1` only ever spawns a LOCAL (`unix://`)
  engine** (it is read from the environment in `graph_compute.py`; in sharded
  mode `is_local_endpoint()` gates it). Auto-starting a local stand-in for a
  remote shard would silently split that shard's graphs into invisible
  islands.
- **The circuit breaker** (OS-5.23, shared per endpoint): consecutive
  connect/timeout failures open the circuit and subsequent callers fail fast
  with the typed `EngineCircuitOpenError` (a `ConnectionError`) instead of
  hammering a dead socket; a half-open probe after the cooldown heals it.
  Graphs on the two healthy shards are completely unaffected throughout.

```bash
docker start engine-shard-3   # breaker half-opens, probe succeeds, closes
```

## 6. The honest re-sharding caveat

HRW minimizes movement when the endpoint list changes — adding/removing one
of N shards reassigns ~1/N of graphs — **but no data moves automatically**. A
graph whose HRW winner changed re-creates *empty* on its new shard until you
migrate it manually:

1. quiesce writers for that graph;
2. export from the old shard (`lifecycle.to_msgpack` /
   `GraphComputeEngine.to_msgpack()`, or copy its `--persist-dir`
   checkpoint);
3. roll out the new `GRAPH_SERVICE_ENDPOINTS` everywhere (identical strings);
4. import on the new shard (`from_msgpack`), then delete the stale copy.

See [engine_sharding.md](../architecture/engine_sharding.md#rebalancing-out-of-scope--the-honest-caveat)
for the full procedure. Durable tiers (pggraph L3) are not partitioned by
this mechanism and are unaffected.

## Verification

```bash
python3 -m pytest tests/unit/knowledge_graph/test_engine_sharding.py -q
```

---

*Smoke-run against this tree (2026-06-11): the tenant→graph→shard routing
table in section 3 was produced by executing `tenant_graph_name` +
`shard_endpoint_for` against this tree, and `python3 -m pytest
tests/unit/knowledge_graph/test_engine_sharding.py -q` passed (22 passed,
1 skipped). The compose bring-up, `/api/dashboard/daemon/shards` curl, and
shard-stop failure flow were reviewed against code only (no containers were
started for this doc).*
