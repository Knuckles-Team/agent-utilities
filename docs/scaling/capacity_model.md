# Capacity Model (Plan 07: Path to Scale)

> **Status (SCALE-P2-1): 1M is now a DEFINED workload contract, measured by a
> harness — not a linear-arithmetic claim.** This page used to say "the 100M
> target follows as a measured projection" and stop there; Codex's SCALE-P2-1
> review correctly called that out — a modeled shard/worker/node COUNT is not
> a demonstrated CAPACITY. The linear arithmetic below (`capacity_model.py`)
> is still here and still useful as a **first-order infrastructure-sizing
> tool** (how many PG shards/L0 shards/nodes to provision), but it is no
> longer what "1M" MEANS. What "1M residents, sustained" means now is defined
> precisely by [`workload_contract.yml`](./workload_contract.yml) +
> [`workload_contract.py`](./workload_contract.py) — registered agents,
> concurrent sessions/turns, turns/tool-calls/graph-mutations/messages/tokens
> per second, tenant count + skew (incl. one elephant tenant), per-agent
> working-set/history/media footprint, interactive/background mix,
> availability + RPO/RTO, and p50/p95/p99/p99.9 SLO targets for
> queue/query/write/end-to-end latency — and it is GENERATED and MEASURED
> against those SLOs by [`scripts/scale/loadgen.py`](../../scripts/scale/loadgen.py)
> (a real driver, not more arithmetic), asserted by the soak/chaos harness in
> [`tests/scale/soak/`](../../tests/scale/soak/). See **"What is CI-measured vs
> hardware-pending"** below for the honest current status: the CI-runnable
> subset actually runs and asserts invariants today; the real-scale/real-duration/
> real-multi-node scenarios are documented and skip-marked, not faked.
>
> The measured anchors below (transport latency, per-shard write throughput,
> per-agent working-set) still stand and now ALSO anchor several
> `workload_contract.yml` fields directly (see that file's `# anchor:` comments)
> so the contract and this model cannot silently drift apart —
> `tests/scale/test_workload_contract.py` cross-checks them.

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
| **Active concurrency** | agents executing *right now* | `worker_pool_size` × node count; **queue-driven dispatch is the live scale-out path for this axis** — `agent_dispatch_backend=queue` + N `agent-dispatch-worker` hosts (see [`architecture/agent_dispatch.md`](../architecture/agent_dispatch.md), CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch) |
| **Resident population** | total agents whose state must persist | `graph_service_endpoints` (PG/L0 shard fan-out — the L0 side is the live tenant-partitioned engine sharding path, see [`architecture/engine_sharding.md`](../architecture/engine_sharding.md), CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw) |
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

## Queue-driven dispatch stage (IMPLEMENTED — CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch)

The "Workers" column above used to be aspirational on the active-concurrency
axis: agent turns executed only inside the in-process asyncio scheduler
(`core/cognitive_scheduler.py`, `max_concurrent` per process) on the host that
accepted them. That stage is now implemented:

- agent turns ride the session-keyed `agent_turns` queue
  (`AGENT_DISPATCH_BACKEND=queue`; transport follows `TASK_QUEUE_BACKEND`);
- any host running `agent-dispatch-worker` claims and executes them against the
  shared state store (AU-OS.state.unified-durable-state-externalization), so "Workers = ceil(active / 25)" maps to a
  **stateless dispatch-worker fleet** spread across "Nodes", not to one
  process's coroutine cap;
- `AGENT_TURNS_PARTITIONS` bounds fleet-wide session concurrency on Kafka the
  same way the Kafka-parts column bounds event drain.

Deployment shape per row of the summary table: stateless gateways + N
dispatch workers + M `kg-ingest-worker` processes + the listed engine/PG
shards and Kafka partitions. Design, ordering and idempotency guarantees:
[`architecture/agent_dispatch.md`](../architecture/agent_dispatch.md).

## The workload contract + load generator + soak/chaos harness (SCALE-P2-1)

The acceptance criterion for "1,000,000 residents" is now: **sustained SLOs
with bounded resource use, and no lost/duplicate/cross-tenant/falsely-completed
side effects** — not an infrastructure-count formula. Three artifacts implement
that:

1. **The contract** ([`workload_contract.yml`](./workload_contract.yml) +
   [`workload_contract.py`](./workload_contract.py)) — a machine-readable,
   validated (`WorkloadContractError` on any malformed/inconsistent field)
   definition of the workload: population, concurrency, five independent rate
   axes (turns/tool-calls/graph-mutations/messages/tokens per second), tenant
   count + Zipf skew + one deliberately oversized elephant tenant, per-agent
   working-set/history/media footprint, interactive/background mix,
   availability target + RPO/RTO, and the four SLO axes (queue/query/write/
   end-to-end latency) each with p50/p95/p99/p99.9 targets. `ScaledWorkload`
   scales the population/rate axes by a `scale` factor for a small CI run or
   the full `scale=1.0` — the SLO targets and per-unit sizes never scale.
2. **The load generator** ([`scripts/scale/loadgen.py`](../../scripts/scale/loadgen.py))
   — actually GENERATES that workload: submits `WorkItem`s (turns) through the
   real engine-native CAS/lease/fencing state machine
   (`orchestration/work_item.py`), publishes `AgentBus` messages with tenant
   namespacing, drives independent mutation/tool-call producers, and measures
   the four SLO axes' percentiles for real. Two engine modes:
   `--engine mock` (an in-memory `FakeScaleEngine`, driven by a deterministic
   single-threaded discrete-event simulation — immune to host CPU jitter and
   near-instant in real wall time, since nothing genuinely sleeps) and
   `--engine live` (the process-active epistemic-graph engine, genuine
   concurrent asyncio tasks against real wall-clock time — the real-hardware
   soak path).
3. **The soak/chaos harness** ([`tests/scale/soak/`](../../tests/scale/soak/)) —
   asserts the acceptance criterion's invariants against the CI-runnable
   subset (see the table below), and documents-but-skips the scenarios that
   need real multi-node hardware, with the harness call each would make
   written out (`test_hardware_pending.py`).

### What is CI-measured vs hardware-pending

| Scenario | CI-runnable today | Notes |
|---|---|---|
| Steady-state SLO + invariant check (scaled) | **Yes** — `test_steady_burst.py::test_steady_phase_slos_and_invariants_hold` | Scaled-down population/rates; SLO *targets* are the full-scale ones (per-operation, not population-dependent) |
| Burst-on-steady + bounded backlog | **Yes** — `test_steady_burst.py::test_burst_phase_bounded_backlog_and_slos_hold` | Asserts the worker pool drains the burst, not an ever-growing queue |
| Worker/host loss mid-lease + crash-recovery reclaim | **Yes** — `test_chaos_worker_and_delivery.py` | Exact CAS/lease state built directly against `work_item.py`, not rate-based |
| Duplicate/redelivered claim + idempotent re-ack | **Yes** — `test_chaos_worker_and_delivery.py` | At-least-once redelivery modeled at the claim/commit level |
| Timeout → retry (backoff) → DLQ | **Yes** — `test_chaos_lifecycle_and_dlq.py` | Exercises `max_attempts`/backoff/dead-letter exactly |
| Cancel mid-flight (never falsely-completes) | **Yes** — `test_chaos_lifecycle_and_dlq.py` | |
| Hot-tenant/noisy-neighbor quota isolation | **Yes** — `test_chaos_tenant_and_restart.py` | Elephant tenant hits `max_tenant_in_flight`; other tenants unaffected |
| Full restart / cold activation (state-machine semantics) | **Yes** — `test_chaos_tenant_and_restart.py` | Engine snapshot/restore models the durable-store guarantee; NOT a claim about real 1M-scale cold-hydrate latency (see below) |
| Rolling upgrade (worker-pool replacement) | **Yes** — `test_chaos_tenant_and_restart.py` | Old-generation leases expire/reclaim; fencing rejects stale acks |
| 24-72h steady + burst soak at the REAL 1M population | **Hardware-pending** — `test_hardware_pending.py` | Needs a deployed fleet + that much wall-clock time; the exact `loadgen.py --engine live --scale 1.0` invocation is written in the test |
| Broker rebalance / live Kafka partition expansion | **Hardware-pending** | Needs a real Kafka cluster |
| Shard split/move under concurrent writes | **Hardware-pending** | Needs a real multi-shard engine deployment |
| Worker/gateway/broker/leader/node/zone loss (REAL infra) | **Hardware-pending** | Needs real multi-node/multi-zone infra + a kill-injection tool |
| Rolling upgrade + schema migration across REAL hosts | **Hardware-pending** | Needs a real multi-host rolling-deploy pipeline |
| Cold activation of the ACTUAL 1,000,000 residents | **Hardware-pending** | Needs the real population on real L0/PG shards — the CI test only proves the state-machine semantics, not real-scale hydrate latency |

Guardrail this table exists to enforce: a modeled capacity is never reported
as a demonstrated result. Every "Hardware-pending" row is a real, currently
`pytest.mark.skip`-marked test with the manual run recipe in its docstring —
not a silently-omitted scenario.

## Production guard

Reaching any of these scales requires non-toy backends. `core/config.py` exposes
`AgentConfig.assert_production_safe()` (see `core/profile_guard.py`): under
`APP_PROFILE=prod` it rejects `graph_persistence_type` of `file`/`sqlite` and
`a2a_broker`/`a2a_storage` of `in-memory`, so a single-host config can never be
mistaken for a production deployment.
