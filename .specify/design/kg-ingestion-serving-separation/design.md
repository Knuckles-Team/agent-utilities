# Design Document: A serving process never embeds and writes memory inline — it enqueues and returns; only the host/ingest worker does the write

CONCEPT:AU-KG.memory.ingestion-serving-separation

> Realised by `agent_utilities/knowledge_graph/core/engine_memory.py:322-405`
> (`store_memory`), with the plane decision at `:339-344` and the enqueue path
> at `:374-405`. Introduced by commit `8d28c280` ("feat(kg): ingestion/serving
> plane separation — offload writes to the queue").

## Decision — split by process *role*, not by call size, and make the serving path's write path asynchronous by construction

`store_memory` now branches on `effective_daemon_role()`. In any process whose
role is not `host`, the embed-and-write is not performed at all: it is enqueued
as a `kg_memory` task on the durable queue and the call returns immediately.
Only the host / ingest-worker process performs the actual embed and write, and
its task handler runs with `_local=True` so it cannot re-enqueue and loop.

**The rejected alternative is doing the embed and write inline in whatever
process happened to call `store_memory`, and it was rejected because it
deadlocked.** The introducing commit states the failure directly:

> *"the synchronous psycopg pool + heavy in-process ingestion (embed+write)
> deadlocked any concurrent read/reply on the shared pool and stalled the event
> loop."*

This is worth stating precisely, because the obvious reading — "writes are
slow, so move them off the hot path" — understates it. The problem is not
latency, it is *resource inversion on a shared pool*. Ingestion holds
connections from the same synchronous pool that serving needs to answer reads.
Under concurrency the ingesting work can hold enough of the pool that the reads
which would let it finish cannot proceed, and the process stops making progress
at all rather than merely going slower.

Two weaker alternatives are ruled out by that framing. Making the write
asynchronous *in place* would not help — the contention is for pool
connections, not for the event loop, so an async write against the same pool
deadlocks the same way. Sizing the pool larger only moves the concurrency level
at which it happens. Separating the *planes* is what actually removes the
inversion: the serving process never acquires a connection for ingestion, so it
cannot be starved by one.

The durable queue is load-bearing in this choice rather than incidental. It is
what makes "return immediately" safe — the write is not dropped, it is owed —
and it is why the serving path can be made unconditional rather than
best-effort.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_memory.py`
  and every caller of `store_memory` in a non-host process.
- **Backward Compatible**: No, and deliberately: `store_memory` in a serving
  process is now **eventually** consistent. A caller that stores a memory and
  immediately reads it back may not see it. Callers that relied on
  read-after-write within one serving process are broken by design.
- **Known weak point**: correctness now depends on `effective_daemon_role()`
  being accurate. A process mis-reporting itself as `host` reintroduces the
  original deadlock; one mis-reporting as serving when no ingest worker is
  running enqueues writes that nothing drains, and the failure is silent in
  both directions — the queue simply grows.
