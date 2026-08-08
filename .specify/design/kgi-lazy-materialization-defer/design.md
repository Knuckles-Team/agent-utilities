# Design Document: A typed `PARTIAL_MATERIALIZATION` claim signal is deferred, never treated as an ingestion failure

CONCEPT:AU-KG.ingest.partial-materialization

> `agent_utilities/knowledge_graph/core/engine_tasks.py:464-482` (the typed
> signal parser, `_retryable_partial_materialization`), `agent_utilities/knowledge_graph/core/engine_tasks.py:3825-3852,5776-5798`
> (the two call sites that defer on it), `tests/unit/knowledge_graph/test_ingest_task_workitem_lifecycle.py:145-198`
> (the behavior spec).

**Governance note**: the `CONCEPT:AU-KG.ingest.partial-materialization`
marker literal appears only in the test file, not in `engine_tasks.py`
itself, which is what the machine-triage `[review]` flag caught. The
implementation it documents is real and substantial
(`_retryable_partial_materialization`, `_defer_task_for_materialization`,
used at two call sites) — this is a real decision missing its own marker in
source, not an unrealized one. Recorded here rather than retired.

## Decision — a catalog-known graph that rejects operations mid lazy-open rebuild is availability state, not a task failure; the claim is deferred (lease released, no attempt consumed) rather than failed or dead-lettered

`_retryable_partial_materialization` (`engine_tasks.py:464-482`) parses the
engine's error payload and states the rule in its own docstring: **"Return
the engine's typed hydration signal, never a text lookalike. A catalog-known
graph deliberately rejects every graph operation while its bounded lazy-open
rebuild is incomplete. That is availability state, not an ingestion
failure. Only the exact retryable wire payload is safe to defer; malformed,
stale, and terminal materialization errors continue through ordinary
failure handling."** It matches ONLY a JSON payload with
`code == "PARTIAL_MATERIALIZATION"` and `retryable is True` — anything else
(a malformed payload, a non-JSON error, a retryable flag absent or false)
returns `None` and falls through to normal failure handling.

**The rejected alternative is treating a mid-rebuild rejection as an
ordinary task failure** — retryable via the standard `_TASK_MAX_ATTEMPTS`
counter, or worse, dead-lettered after repeated failures against a graph
that was never actually broken, just still opening. It loses because it
conflates two different kinds of "the operation didn't succeed":
infrastructure availability state (the graph will accept operations again
once its lazy-open rebuild finishes) versus a genuine task-level failure
(bad input, a bug, a permanently unreachable dependency). Counting the
former against `_TASK_MAX_ATTEMPTS` would exhaust a task's retry budget on a
condition that resolves itself with time, potentially dead-lettering
perfectly valid work.

**The defer mechanism is deliberately minimal**: `_defer_task_for_materialization`
(`engine_tasks.py:5776-5798`) releases the lease immediately, "without
consuming an attempt," and re-queues via `defer_work_item(...,
next_retry_at=time.time() + 1.0, reason_ref="partial_materialization")` — a
short, fixed backoff, not an escalating one, because the underlying
condition (a bounded rebuild) is expected to resolve quickly. The claim
handling comment at the first call site (`engine_tasks.py:3833-3841`)
underscores the recovery model: "the native WorkItem itself self-heals via
its own lease TTL + the pre-existing expired-lease reaper — never converted
into an application failed/dead-letter path."

**This composes with, and must not be silently dropped by, the reserved-
hydration claim floor** — the test file's own framing
(`test_ingest_task_workitem_lifecycle.py:159-166`): "Wave-0's materialization
deferral and the reserved-hydration claim floor must BOTH survive in the
same `_claim_next_task`... They touch adjacent regions of one function, so a
merge could silently drop either." The two mechanisms are independently
real decisions occupying the same function, which is exactly the kind of
adjacency this test exists to guard.

## Risk Assessment

- **Blast Radius**: `core/engine_tasks.py`'s claim path
  (`_claim_next_task`, `_defer_task_for_materialization`).
- **Backward Compatible**: Yes.
- **Breaking Changes**: None.
- **Known weak point**: the source implementation carries no
  `CONCEPT:AU-KG.ingest.partial-materialization` marker of its own — only
  the test does. A future refactor of `engine_tasks.py` that doesn't consult
  the test suite has no in-source signal that this deferral logic is a
  deliberate decision rather than incidental error-handling, and could
  regress it silently.
