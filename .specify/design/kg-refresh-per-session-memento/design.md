# Design Document: The reply path reads mementos from a process-local cache refreshed by the existing background pass — it never fetches them inline

CONCEPT:AU-KG.memory.refresh-per-session-memento

> Realised by `agent_utilities/knowledge_graph/memory/session_memento_cache.py:1-16`
> (module docstring), `:32-90` (`SessionMementoCache`) and `:93-108`
> (`refresh_session_memento_cache`); consumed at
> `agent_utilities/messaging/router.py:792` and
> `agent_utilities/orchestration/agent_runner.py:880`, `:2475`. Introduced by
> commit `b794e6af` ("perf(orchestration): chat execution profile +
> non-blocking reply path").

## Decision — reuse the background pass that is already running instead of adding a refresh mechanism

The module docstring states the problem exactly:

> *"`get_recent_mementos(...)` is a synchronous backend round-trip. Running it
> inline on the async reply path (inside `_build_execution_config`) blocked the
> event loop on every chat turn."*

Priming a turn with recent mementos is worth doing, but it was being paid for
on the hot path, synchronously, on *every* turn — and because it blocked the
event loop, the cost was not confined to the turn that incurred it.

The fix is a process-local, thread-safe LRU keyed by session
(`SessionMementoCache`). The reply path reads from the cache and never fetches.
The substantive choice is *who fills it*: `refresh_session_memento_cache` is
called from the **already-existing** `_persist_and_enrich` background pass,
which runs after a turn completes. No new timer, no new task, no new background
worker.

**The rejected alternative is the prior behaviour — the inline synchronous
fetch** — rejected because a blocking call on an async reply path taxes every
concurrent turn, not just its own.

But the more interesting rejected alternative is the obvious fix that was *not*
taken: adding a dedicated refresh loop or a TTL-triggered async fetch. That
would work, and it would introduce a second thing that must be started,
supervised, shut down, and reasoned about when it fails. Attaching the refresh
to a pass that already runs at exactly the right moment — after a turn, when
the session's mementos have just changed — gets freshness with no new lifecycle.
The cache is warm for turn N+1 precisely because turn N finished.

The consequence accepted in exchange is that the cache is refreshed on turn
boundaries, not on demand. A session's first turn reads a cold cache, and a
memento written by something *other* than this session's own turns is not
picked up until the next turn completes.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/memory/session_memento_cache.py`,
  `agent_utilities/messaging/router.py`,
  `agent_utilities/orchestration/agent_runner.py`.
- **Backward Compatible**: Yes — same data, one turn staler in the worst case.
- **Known weak point**: the cache is process-local, so in a multi-replica
  deployment each replica maintains its own view of a session's mementos and a
  session whose turns land on different replicas sees inconsistent priming.
  There is no invalidation path either: a memento deleted or superseded
  out-of-band stays in the cache until the session's next turn refreshes it or
  the LRU evicts it.
