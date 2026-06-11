# Capacity Model (Plan 07: Path to Scale)

> **Status: MODELED, with measured anchors.** Three things are now *measured*
> (not asserted): the epistemic-graph single-connection transport latency,
> **per-shard linear write throughput** (1→4 shards ≈ 6.6× at constant wall-time,
> no shared-state cliff), and the **per-agent working-set footprint (~52 kB)** for
> a bounded subgraph — all in `epistemic-graph/docs/benchmarks.md` (reproduce with
> `scripts/bench_scale.py`). Every resident-population sizing figure (10k+) is a
> **linear extrapolation** from those anchors plus the per-unit constants below.
> The 100M target follows as a measured projection (~78 hosts @ 64 GB / 52 kB per
> agent) — **we do not claim 100M has been *run***; it has not. The arithmetic
> lives in [`capacity_model.py`](./capacity_model.py), unit-tested in
> `tests/scale/test_capacity_model.py`.

## The measured anchor

From `epistemic-graph/docs/benchmarks.md` (run captured 2026-06-01, Linux x86-64,
single client connection, in-memory graph, length-prefixed MessagePack over UDS):

| Operation  | ops  | p50      | p99      |
|------------|------|----------|----------|
| `AddNode`  | 3000 | 0.187 ms | 0.223 ms |

`AddNode` **p50 ≈ 0.19 ms** ⇒ **~5,000 sequential ops/sec on a single
connection**. This is the only empirical input. Throughput above one connection
comes from connection pooling (`pool.py`) and shard fan-out (`ShardRouter`);
the server sheds excess concurrent load with a `BUSY` response
(`EPISTEMIC_GRAPH_MAX_INFLIGHT`, default 1024) rather than queueing unbounded.

## Three-axis framing

Scaling is **not** a single dimension. We size three independent axes and take
the max of the resulting infrastructure:

| Axis | Driver | Knob in `core/config.py` |
|------|--------|--------------------------|
| **Active concurrency** | agents executing *right now* | `worker_pool_size` × node count; **queue-driven dispatch is the live scale-out path for this axis** — `agent_dispatch_backend=queue` + N `agent-dispatch-worker` hosts (see [`architecture/agent_dispatch.md`](../architecture/agent_dispatch.md), CONCEPT:ORCH-1.45) |
| **Resident population** | total agents whose state must persist | `graph_service_endpoints` (PG/L0 shard fan-out — the L0 side is the live tenant-partitioned engine sharding path, see [`architecture/engine_sharding.md`](../architecture/engine_sharding.md), CONCEPT:KG-2.58) |
| **Event throughput** | graph events/sec driving fan-out | `kafka_bootstrap_servers` partitions |

A deployment can be huge on one axis and tiny on another (e.g. 1M dormant
residents with 2% active). Sizing each axis separately avoids over- or
under-provisioning.

## Per-unit planning constants (MODELED)

These live as named constants in `capacity_model.py` so they are testable and
adjustable in one place:

| Constant | Value | Meaning |
|----------|-------|---------|
| `RESIDENTS_PER_PG_SHARD` | 250,000 | residents per durable Postgres shard |
| `RESIDENTS_PER_L0_SHARD` | 50,000 | residents per hot in-memory L0 shard (one `GRAPH_SERVICE_ENDPOINTS` entry; routed per named graph by HRW) |
| `ACTIVE_AGENTS_PER_WORKER` | 25 | concurrently active agents one worker multiplexes |
| `WORKERS_PER_NODE` | 8 | workers per node (= `worker_pool_size` default) |
| `OPS_PER_SEC_PER_KAFKA_PARTITION` | 5,000 | = the measured single-connection drain rate |
| `EVENTS_PER_ACTIVE_AGENT_PER_SEC` | 2.0 | modeled graph events per active agent |
| `MIN_KAFKA_PARTITIONS` | 3 | floor for ordering/parallelism headroom |

`OPS_PER_SEC_PER_KAFKA_PARTITION` is the **one constant tied to the measured
anchor** — one consumer connection drains ~5,000 ops/sec, so we size one
partition per connection. The rest are conservative planning round numbers, not
measurements.

## The arithmetic (2% active fraction)

Active agents = `ceil(residents × 0.02)`. Then:

- PG shards = `ceil(residents / 250,000)`
- L0 shards = `ceil(residents / 50,000)`
- Workers   = `ceil(active / 25)`
- Nodes     = `ceil(workers / 8)`
- Events/s  = `active × 2.0`
- Kafka parts = `max(3, ceil(events_per_sec / 5,000))`

### 1,000 residents (MODELED, trivially within a single dev box)

- active = `ceil(1000 × 0.02)` = **20**
- PG shards = `ceil(1000/250000)` = **1**
- L0 shards = `ceil(1000/50000)` = **1**
- workers = `ceil(20/25)` = **1**, nodes = `ceil(1/8)` = **1**
- events/s = `20 × 2` = **40**, kafka = `max(3, ceil(40/5000))` = **3**

### 100,000 residents (MODELED)

- active = `ceil(100000 × 0.02)` = **2,000**
- PG shards = `ceil(100000/250000)` = **1**
- L0 shards = `ceil(100000/50000)` = **2**
- workers = `ceil(2000/25)` = **80**, nodes = `ceil(80/8)` = **10**
- events/s = `2000 × 2` = **4,000**, kafka = `max(3, ceil(4000/5000))` = **3**

### 1,000,000 residents (MODELED — the documented reference case)

- active = `ceil(1000000 × 0.02)` = **20,000**
- PG shards = `ceil(1000000/250000)` = **4**
- L0 shards = `ceil(1000000/50000)` = **20**
- workers = `ceil(20000/25)` = **800**, nodes = `ceil(800/8)` = **100**
- events/s = `20000 × 2` = **40,000**, kafka = `max(3, ceil(40000/5000))` = **8**

These exact numbers (4 PG shards, 20 L0 shards, 800 workers, 100 nodes, 8 Kafka
partitions) are asserted in `tests/scale/test_capacity_model.py` so the doc and
the code cannot silently drift.

### Linear extrapolation to 100,000,000 residents (MODELED — NOT tested)

Pure linear scaling of the same constants (the model is linear above the
single-shard floor):

- active = `ceil(100000000 × 0.02)` = **2,000,000**
- PG shards = `ceil(100000000/250000)` = **400**
- L0 shards = `ceil(100000000/50000)` = **2,000**
- workers = `ceil(2000000/25)` = **80,000**, nodes = `ceil(80000/8)` = **10,000**
- events/s = `2000000 × 2` = **4,000,000**, kafka = `max(3, ceil(4000000/5000))` = **800**

## Summary table

| Residents | Active (2%) | PG shards | L0 shards | Workers | Nodes | Kafka parts | Events/s |
|-----------|-------------|-----------|-----------|---------|-------|-------------|----------|
| 1,000         | 20        | 1   | 1     | 1      | 1      | 3   | 40        |
| 100,000       | 2,000     | 1   | 2     | 80     | 10     | 3   | 4,000     |
| 1,000,000     | 20,000    | 4   | 20    | 800    | 100    | 8   | 40,000    |
| 100,000,000   | 2,000,000 | 400 | 2,000 | 80,000 | 10,000 | 800 | 4,000,000 |

## What is measured vs extrapolated

- **Measured:** `AddNode` p50 = 0.187 ms, ~5,000 ops/sec/connection
  (`epistemic-graph/docs/benchmarks.md`). This anchors
  `OPS_PER_SEC_PER_KAFKA_PARTITION`.
- **Extrapolated (MODELED):** every shard / worker / node / partition figure
  for ≥10k residents. Linear scaling is an *assumption*; real systems hit
  super-linear coordination costs (cross-shard queries, rebalancing, tail
  latency) that this first-order model intentionally ignores. Treat 1M as an
  engineering target to validate, and 100M as an order-of-magnitude sketch only.

## Queue-driven dispatch stage (IMPLEMENTED — CONCEPT:ORCH-1.45)

The "Workers" column above used to be aspirational on the active-concurrency
axis: agent turns executed only inside the in-process asyncio scheduler
(`core/cognitive_scheduler.py`, `max_concurrent` per process) on the host that
accepted them. That stage is now implemented:

- agent turns ride the session-keyed `agent_turns` queue
  (`AGENT_DISPATCH_BACKEND=queue`; transport follows `TASK_QUEUE_BACKEND`);
- any host running `agent-dispatch-worker` claims and executes them against the
  shared state store (OS-5.16), so "Workers = ceil(active / 25)" maps to a
  **stateless dispatch-worker fleet** spread across "Nodes", not to one
  process's coroutine cap;
- `AGENT_TURNS_PARTITIONS` bounds fleet-wide session concurrency on Kafka the
  same way the Kafka-parts column bounds event drain.

Deployment shape per row of the summary table: stateless gateways + N
dispatch workers + M `kg-ingest-worker` processes + the listed engine/PG
shards and Kafka partitions. Design, ordering and idempotency guarantees:
[`architecture/agent_dispatch.md`](../architecture/agent_dispatch.md).

## Production guard

Reaching any of these scales requires non-toy backends. `core/config.py` exposes
`AgentConfig.assert_production_safe()` (see `core/profile_guard.py`): under
`APP_PROFILE=prod` it rejects `graph_persistence_type` of `file`/`sqlite` and
`a2a_broker`/`a2a_storage` of `in-memory`, so a single-host config can never be
mistaken for a production deployment.
