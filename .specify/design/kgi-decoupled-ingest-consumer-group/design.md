# Design Document: Ingest workers become standalone Kafka-consumer processes, not daemon threads inside the engine host

CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer

> `agent_utilities/knowledge_graph/ingest_worker.py` (primary — the `kg-ingest`
> consumer group), `agent_utilities/knowledge_graph/core/kafka_queue_backend.py`
> (the producer/topic/partition-key side), `agent_utilities/knowledge_graph/core/engine_tasks.py`
> (the shared worker body this reuses), `agent_utilities/observability/gateway_metrics.py:242-248`
> and `agent_utilities/knowledge_graph/ingestion/batch_orchestrator.py:88-99`
> (uniform queue-depth/lag observability), `docs/reference/metrics.md:68`.

## Decision — with the Kafka queue backend selected, any number of worker processes on any host join the `kg-ingest` consumer group as engine clients, reusing the exact in-process worker body

`ingest_worker.py:4-16` states the decision directly: **"ingest workers no
longer have to live as daemon threads inside the host engine process."**
With `TASK_QUEUE_BACKEND=kafka` selected, worker processes — on any host —
consume keyed task messages from `kg_tasks` and process them with the SAME
worker body the in-process workers use
(`TaskManagerMixin._execute_claimed_task` — **extracted, not duplicated**,
`ingest_worker.py:11-12`). Workers are engine *clients*: they talk to the
single Rust epistemic-graph daemon over UDS/TCP with the shared HMAC secret
and never take the KG host flock — `KG_DAEMON_ROLE=client` is forced so a
worker process spawns no daemon threads of its own (`ingest_worker.py:14-17`).

**The rejected alternative is the prior architecture**: ingestion workers as
daemon threads living inside the same process as the engine host. It loses
on the property this decision exists to gain — horizontal scale-out. A
daemon-thread worker pool is bounded by one host's CPU/memory and dies with
that host's process; it cannot be scaled by adding more worker processes
elsewhere, and every worker restart couples to the engine daemon's own
lifecycle. Making workers engine *clients* over the network decouples worker
scaling from engine-host scaling entirely.

**Delivery-semantics decisions bundled with this one** (`ingest_worker.py:19-30`):

- **At-least-once**: offsets commit only after a task finishes (or is
  durably marked failed); a worker crash redelivers to another group member.
- **Idempotent claims**: Kafka is notification transport only — every
  consumer atomically claims the deterministic `WorkItem`; a negative claim
  is final and never consults a second status/lock authority
  (`claim_task_envelope`, `ingest_worker.py:68-80`).
- **Per-key ordering, not global**: `kafka_queue_backend.py:14-24` documents
  the companion producer-side decision — every task is produced to
  `kg_tasks` with a partition key (`tenant:` → `corpus:` → `type:`, first
  match wins), so Kafka guarantees per-key ordering **without global
  serialization**. The consumer-side docstring is explicit about the cost:
  "There is no cross-partition priority lane (unlike the graph-polling
  mode's high-priority poll)" (`ingest_worker.py:29-30`) — a real, named
  tradeoff versus `AU-KG.ingest.hardened-priority-scheduled-task`'s bucket
  priority, accepted because per-key ordering is the property this backend
  is chosen for.
- **Fail-closed backend selection**: `kafka_queue_backend.py:5-10`
  (`CONCEPT:AU-KG.backend.selectable-queue-backend`, documented elsewhere) —
  an unreachable broker raises at startup rather than silently degrading to
  per-host SQLite, which would split the fleet's queue into invisible
  islands. This decoupled consumer only exists meaningfully because that
  fail-closed guarantee makes "the queue" a single fleet-wide truth workers
  anywhere can trust.

Observability follows the same decoupling: `gateway_metrics.py:242-248`
records `KG_INGEST_QUEUE_DEPTH` uniformly across backends (sqlite/postgres
row count, kafka = `kg-ingest` consumer-group lag), and
`batch_orchestrator.py:88-99`'s `_inflight_count` prefers the engine's
uniform `ingest_queue_depth()` for the same reason — callers reasoning about
backpressure don't need to know which backend (in-process or decoupled) is
actually draining the queue.

## Risk Assessment

- **Blast Radius**: `ingest_worker.py`, `kafka_queue_backend.py`,
  `core/engine_tasks.py` (the shared `_execute_claimed_task` body),
  `observability/gateway_metrics.py`.
- **Backward Compatible**: Yes — the in-process daemon-thread pool remains
  the default under the SQLite/Postgres backends; Kafka + decoupled workers
  is an opt-in topology.
- **Breaking Changes**: None.
- **Known weak point**: no cross-partition priority lane, named explicitly
  in the module docstring — a critical-priority task queued behind a large
  backlog on the same partition key has no fast-path under this backend, in
  contrast to the graph-polling mode's high-priority poll.
