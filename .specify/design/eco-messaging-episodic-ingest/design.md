# Design Document: Chat auto-ingestion is a blocking KG write moved off the reply loop

CONCEPT:AU-ECO.messaging.blocking-store-memory ·
CONCEPT:AU-ECO.messaging.episodic-memory-recall

> `agent_utilities/messaging/kg_ingest.py`

## Decision — `store_memory` is blocking (graph write + embedding), so it runs via `asyncio.to_thread`

`CONCEPT:AU-ECO.messaging.blocking-store-memory` (`kg_ingest.py:98`)

Every inbound/outbound chat turn is auto-ingested into the KG as episodic
memory. `engine.store_memory` performs a graph write plus an embedding call,
both synchronous, blocking operations. Calling it directly from the async
message-handling coroutine would stall the event loop — and therefore every
other concurrent conversation's reply — for the duration of the write.

**The rejected alternative** is calling it inline (`await
engine.store_memory(...)` as if it were async, or a bare synchronous call).
The code instead wraps it in `asyncio.to_thread(engine.store_memory, ...)`
(`kg_ingest.py:96`) specifically so "ingest never stalls the messaging/reply
loop" (comment at the call site) — the cost of a KG write is paid on a worker
thread, not the loop driving replies.

### Pointer — `CONCEPT:AU-ECO.messaging.episodic-memory-recall`

The memory this ingests is deliberately **episodic**, read back by the
*universal* recall path (see
`.specify/design/eco-messaging-universal-graph-agent/design.md`), not by a
bespoke per-channel history query. Reply continuity within one conversation
comes from the per-session conversation *memento*
(`knowledge_graph/memory/memento_compressor.py`), so this ingestion path
does not additionally stamp a flat `channel_key`/`role`/`text` scaffold the
way a hand-rolled chat-log table would — that would be a second, redundant
continuity mechanism competing with the memento's. Episodic ingestion here
exists for cross-session/cross-surface recall ("what did we discuss last
week"), not for the *next-turn* continuity, which is a different mechanism
entirely.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/kg_ingest.py`.
- **Backward Compatible**: Yes — describes existing behavior.
- **Known weak point**: `asyncio.to_thread` bounds loop-blocking but not
  total worker-thread contention; a burst of conversations each ingesting
  concurrently competes for the thread pool, not the event loop — a
  different (currently unaddressed) resource.
