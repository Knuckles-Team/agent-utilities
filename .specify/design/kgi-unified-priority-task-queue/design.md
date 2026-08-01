# Design Document: One discrete integer priority bucket, enforced by native WorkItem claim ordering, shared by every scheduler/dispatcher

CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task

> `agent_utilities/knowledge_graph/core/engine_tasks.py:412-430,2991-3005`
> (primary — `_PRIORITY_BUCKETS`, `submit_task`), `agent_utilities/knowledge_graph/research/loops.py:527-538`
> (`_prio_bucket` — the shared lazy-import normalizer wrapper),
> `agent_utilities/orchestration/agent_dispatch.py:96-100` (`DispatchEnvelope.prio_bucket`
> reusing the identical bucket), `docs/recipes/unified-scheduling.md`.

## Decision — priority is ONE discrete integer bucket (0=critical .. 3=background), fenced natively by the WorkItem claim; no string priority or client-side selector exists anywhere

`engine_tasks.py:418-424` states the constraint as a fact about the system,
not an aspiration: **"Priority is one discrete integer bucket (0=critical ..
3=background). The native WorkItem claim orders and fences these buckets; no
string priority or client-side claim selector exists."** The four buckets
(`_PRIO_CRITICAL, _PRIO_HIGH, _PRIO_NORMAL, _PRIO_BACKGROUND = 0, 1, 2, 3`)
are the entire vocabulary.

**The rejected alternative, named by its explicit absence, is a
richer/string-typed priority model** — named priority levels
(`"critical"`/`"high"`/…), or a client-side selector that picks which
claim to attempt next based on local heuristics. Either would let priority
resolution live partly outside the native WorkItem claim, which is exactly
what this decision forecloses: the ordering and fencing guarantee comes from
the *native claim itself*, not from application code cooperating around it.
A string/label scheme would also need a mapping table to whatever the native
claim actually understands — this decision skips that translation layer by
making the bucket integer the one and only representation, end to end.

**The decision is also "ONE normalizer, reused everywhere," not per-call-site
coercion.** `loops.py:527-538`'s `_prio_bucket` is explicitly a "thin lazy-import
wrapper over `engine_tasks._coerce_prio_bucket` — the single priority
normalizer shared by tasks / dispatch / schedules / loops," lazy specifically
to avoid an import cycle (mirroring how `bus.py`/`state_tools.py`/
`schedule_engine.py` reach the same normalizer). `agent_dispatch.py:96-100`
confirms the same integer contract on the dispatch side: `DispatchEnvelope.prio_bucket`
is "the ONE discrete integer bucket... identical to the WorkItem `prio_bucket`,"
validated through the same `_coerce_prio_bucket` function
(`agent_dispatch.py:104-108`). Four independent subsystems (task submission,
research loops, agent dispatch, the unified scheduler) share one bucket
vocabulary and one coercion function rather than each defining or converting
its own.

**Bundled into the same submission path**: `submit_task`
(`engine_tasks.py:2991-3005`) also owns `scheduled_for`/`depends_on`,
evaluated atomically by native WorkItem selection, and a deterministic
`job_id` option so a double-fire from the unified Scheduler
(`sched:<name>:<minute>`) becomes an idempotent upsert rather than a
duplicate task. `_TASK_MAX_ATTEMPTS = 3` (`engine_tasks.py:426`) is the
retry/dead-letter bound this priority queue enforces uniformly across
buckets.

## Risk Assessment

- **Blast Radius**: `core/engine_tasks.py` (`_coerce_prio_bucket`,
  `submit_task`), `research/loops.py`, `orchestration/agent_dispatch.py`,
  the unified scheduler (`docs/recipes/unified-scheduling.md`).
- **Backward Compatible**: Yes — this documents the existing, already-shipped
  contract.
- **Breaking Changes**: None.
- **Known weak point**: because there is no cross-partition/cross-lane
  priority escalation described here beyond the four buckets, a critical
  task queued behind a large background backlog on the SAME claim source
  still waits for bucket-ordered fairness rather than preempting — this
  composes with (and is bounded by) whatever lane/fairness scheme the
  claiming worker pool runs, which this document does not itself resolve.
