# Tenant-Partitioned Engine Sharding

> CONCEPT:KG-2.58 (sharding) · CONCEPT:OS-5.28 (topology visibility)

Stage-2 scaling for the epistemic-graph compute tier: run **N independent
engine processes ("shards")** and let every client route to the right one —
no proxy hop, no coordinator, no engine changes.

## The partition model in one line

```
tenant  →  named graph  →  HRW (rendezvous hash)  →  shard endpoint
```

- **The named graph is the partition unit.** Each engine process keeps its own
  string-keyed named-graph registry; a graph lives wholly on exactly one
  shard, so single-graph operations never need cross-shard coordination.
- **Tenancy enters only by choosing the graph name.** When a caller does not
  target an explicit graph and the ambient `ActorContext` (CONCEPT:OS-5.14)
  carries a tenant, the default graph is mapped to
  `tenant__<tenant>__<base>` by `tenant_graph_name()`
  (`knowledge_graph/core/shard_topology.py`, also exported from
  `agent_utilities.knowledge_graph` and as `KnowledgeGraph.tenant_graph()`).
- **Shard choice is a pure function of the graph name.** The sync client path
  (`GraphComputeEngine`) delegates to the exact HRW implementation in
  `epistemic_graph.pool.ShardRouter`, so sync and async callers can never
  disagree on placement.

## Configuration

| Flag (on `AgentConfig`) | Default | Meaning |
|---|---|---|
| `GRAPH_SERVICE_ENDPOINTS` | unset | Comma-separated or JSON list of shard endpoints (`unix://` / `tcp://`). Unset or one entry = today's single-engine behaviour (zero-infra preserved); 2+ entries enable sharding. |
| `KG_DEFAULT_GRAPH` | `__bus__` | The default named graph; the ambient tenant maps onto `tenant__<t>__<default>` in sharded mode only. |

Routing-key resolution (`resolve_routing_graph`):

1. explicit, non-default graph name → used verbatim;
2. ambient `ActorContext` tenant → `tenant_graph_name(tenant, default)`;
3. otherwise → the configured default graph.

Endpoint strings are hashed **verbatim** — configure every client with the
*identical* list (order does not matter; HRW is order-independent) and with
explicit schemes.

## Operational semantics

- **Autostart is local-only.** `EPISTEMIC_GRAPH_AUTOSTART=1` may spawn an
  engine only for a local (`unix://`) endpoint. In sharded mode an
  unreachable remote (`tcp://`) shard is a **fail-loud `ConnectionError`**
  naming the shard, the graph it owns, and the remediation — the same
  hard-contract convention as the CONCEPT:KG-2.55 task queue. Auto-starting a
  local stand-in would silently split that shard's graphs into invisible
  islands.
- **The flock host role is per-host.** `host_lock.py` elects ONE daemon owner
  per host for the *local* engine; remote shards are reported by the status
  surfaces, never managed.
- **Auth is fleet-wide.** All shards and all clients must share ONE
  `GRAPH_SERVICE_AUTH_SECRET` (CONCEPT:OS-5.14). Set it explicitly in
  multi-host deployments — the auto-generated per-install secret only covers
  one host.

## Rebalancing (out of scope — the honest caveat)

HRW keeps key movement minimal when a shard is added or removed (~1/N of
graphs change owner), but **no data moves automatically**. A graph whose HRW
winner changed re-creates **empty** on its new shard until you migrate it
manually with the existing snapshot tooling:

1. quiesce writers for that graph;
2. export from the old shard (`lifecycle.to_msgpack` /
   `GraphComputeEngine.to_msgpack()`, or copy its `--persist-dir` checkpoint);
3. import on the new shard (`from_msgpack`) **after** the endpoint list
   changed everywhere;
4. delete the stale copy from the old shard.

Durable tiers (pggraph L3) are unaffected — they are not partitioned by this
mechanism.

## Topology visibility (CONCEPT:OS-5.28)

- `shard_topology_status()` → shard mode, per-endpoint transport-level
  reachability probe, locality, and circuit-breaker state. Surfaced on:
  - the unified daemon status (`unified_daemon_status()["shards"]`, i.e.
    `GET /daemon/status` and `python -m agent_utilities.gateway.daemon --status`);
  - the gateway dashboard route `GET /daemon/shards`;
  - graph-os `GET /health` (cheap config-only summary: `shard_mode`,
    `shard_count` — no probe on the liveness path).
- Prometheus (OS-5.23 registry, `agent_utilities/observability/gateway_metrics.py`):
  - `agent_utilities_engine_shard_up{endpoint}` — 1/0, refreshed on every real
    client connect and by the status probe;
  - `agent_utilities_engine_shard_requests_total{endpoint,outcome}` — the
    engine-call outcomes (`ok | connection_error | error | short_circuited`)
    split per shard;
  - the existing `agent_utilities_gateway_engine_breaker_state{endpoint}` is
    already per-endpoint, so each shard gets its own circuit breaker for free.
- Each engine process can additionally expose its own native metrics with
  `--metrics-addr` (`epistemic_graph_*` series, one scrape target per shard).

## Worked example — 3 shards on one host

See [`docker/engine-shards.compose.yml`](https://github.com/knuckles-team/agent-utilities/blob/main/docker/engine-shards.compose.yml)
for the runnable compose file (3 engines, distinct ports + persist dirs +
metrics listeners, one shared secret), or by hand:

```bash
export GRAPH_SERVICE_AUTH_SECRET="$(openssl rand -hex 32)"
for i in 1 2 3; do
  epistemic-graph-server \
    --tcp-addr "127.0.0.1:910${i}" \
    --persist-dir "/var/lib/epistemic-graph/shard-${i}" \
    --metrics-addr "127.0.0.1:911${i}" &
done

# Every agent-utilities client / gateway / ingest worker:
export GRAPH_SERVICE_ENDPOINTS="tcp://127.0.0.1:9101,tcp://127.0.0.1:9102,tcp://127.0.0.1:9103"
```

Multi-host is the same picture with one engine (or a few, on big hosts) per
machine and hostnames in the endpoint list. Capacity planning for shard
counts lives in [`docs/scaling/capacity_model.md`](../scaling/capacity_model.md)
(`RESIDENTS_PER_L0_SHARD`).
