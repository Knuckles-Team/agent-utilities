# Design Document: Ingestion stages are separate worker pools coupled ONLY by bounded queues — never a shared lock, never one blocking coroutine

CONCEPT:AU-KG.ingest.staged

> `agent_utilities/knowledge_graph/ingestion/staged_pipeline.py`, pinned by
> `tests/unit/knowledge_graph/test_staged_pipeline.py`.

> Note on the marker id: the tool triage flagged this as `[retire]` — "id is
> a single generic noun — names a subject area, not a choice" — but asked
> for a read-first check because "staged" is plausibly a real two-phase
> pipeline decision. Having read `staged_pipeline.py`, it is exactly that: a
> concrete, non-obvious architectural choice (bounded-queue-coupled worker
> pools vs. one inline blocking coroutine) with a real rejected alternative.
> It is a document candidate, not a retire.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.generalized-cross-lane-parallelization` | fans independent ITEMS out concurrently within one stage; this document fans different STAGES of the same item's processing out concurrently — orthogonal axes that compose | 0.35 | KG |
| `AU-KG.ingest.deterministic-classifier` / `AU-KG.ingest.over-same-tree-fan` | the classify step that would sit upstream of a `parse`/`write` stage boundary in this model | 0.20 | KG |

### Extension Analysis

- **Primary Extension Point**: `compute_stage_workers(kind)` — sizing a new
  stage kind (`fetch`/`parse`/`enrich`/`write`/`reason`) is a new branch
  there, not a change to `Stage`/`StagedPipeline` themselves.
- **New Concept Required?**: No.

## Decision — a linear chain of independent async worker pools connected by bounded queues, not one inline `fetch → parse → enrich → write` coroutine

`CONCEPT:AU-KG.ingest.staged`

The block this removes, named directly in the module docstring
(`staged_pipeline.py:13-20`): `IngestionEngine.ingest` today runs the
structural write and the LLM-bound enrichment inline, in the SAME coroutine,
before it returns. The durable WRITE side sits idle while a job enriches,
and the worker that claimed the job cannot pick up the next one until
enrichment finishes. This module is roadmap item **C** of the north-star
compute architecture (`staged_pipeline.py:3`), implementing "the operator
principle (the law)": *"Separated, interconnected queues, never
dependent-locked; non-blocking everywhere; blocked time with nothing to do
is wasted compute. A slow stage must NOT stall an unrelated stage"*
(`staged_pipeline.py:7-11`).

The model: each `Stage` is its own fixed-size async worker pool consuming a
bounded `asyncio.Queue` and producing into the next stage's bounded queue
(`staged_pipeline.py:22-27`, `Stage.__init__`/`_worker`,
`staged_pipeline.py:114-183`). Three properties make this a genuine
decision, not just "use a queue":

1. **Stages run concurrently** — while repo A enriches, repo B parses and
   repo C writes; the pools are independent (`staged_pipeline.py:29-30`).
2. **Backpressure, not locks** — a full downstream queue makes the
   *producer* stage `await queue.put(...)`; it never takes a lock another
   stage needs and never stalls an unrelated stage
   (`staged_pipeline.py:31-33`, enforced in `_worker`'s bounded
   `await nxt.in_q.put(out)`, `staged_pipeline.py:176-181`).
3. **No dependent locks** — coupling is ONLY via the bounded queues; a
   handler exception is counted and swallowed so one poison item can never
   stall its pool (`staged_pipeline.py:170-173`, `Stage`'s own docstring
   `staged_pipeline.py:117-121`).

Shutdown is chained `Queue.join()`: drain stage *k* fully (every
`task_done` accounted, which guarantees all of its puts into stage *k+1*
landed) before joining stage *k+1* — lock-free and deadlock-free for a
linear DAG, since downstream pools keep draining while an upstream `join`
waits, so a backpressured `put` always unblocks (`staged_pipeline.py:41-45`,
`StagedPipeline.drain`, `staged_pipeline.py:252-265`). Pool sizes are
auto-derived per bottleneck via `compute_stage_workers` — fetch is
net-bound (a small multiple of the cpu anchor), parse/write are cpu-bound
(the shared ingest-worker anchor), enrich is LLM/GPU-slot-bound (capped by
`KG_LLM_CONCURRENCY` minus the reserved interactive slot so background
enrichment can never starve the messaging responder), reason is a small
serial-mirror fan-out (`staged_pipeline.py:290-328`) — all sharing
`compute_ingest_worker_count` as the one cpu/mem sizing anchor so the
Pi-OOM cap applies uniformly.

**The rejected alternative is the status quo it explicitly names**: one
inline coroutine running `fetch → parse → enrich → write` for each item to
completion before starting the next. That alternative is simpler but
violates the stated law directly — a slow LLM-bound enrich call blocks the
cheap, fast durable write of the SAME item, and blocks the worker from
starting the NEXT item's fetch/parse entirely, even though those stages have
nothing to do with the slow one. `test_staged_pipeline.py`'s own framing
calls out its load-bearing test: `test_slow_enrich_does_not_block_write`
"proves the operator principle: a slow ENRICH stage does NOT stall the
WRITE of other items" (`test_staged_pipeline.py:3-7`).

**A second, implicit rejected alternative is coordinating stages via a
shared lock or a single shared queue with priority tags** instead of one
bounded queue per stage boundary. The docstring is explicit that "a stage
never holds a lock that a different stage must acquire" and that reads/other
stages never block on a writer (`staged_pipeline.py:34-36`) — a shared-lock
design would reintroduce exactly the coupling the staged model exists to
remove, just with a different name.

## Risk Assessment

- **Blast Radius**: currently `agent_utilities/knowledge_graph/ingestion/
  staged_pipeline.py` only — the module is a self-contained primitive
  (`Stage`/`StagedPipeline`/`compute_stage_workers`) not yet wired into
  `IngestionEngine.ingest`'s call path (the block it targets, per the
  docstring, is `IngestionEngine.ingest` itself running inline today).
- **Backward Compatible**: N/A pending integration — as a standalone
  primitive it changes nothing about existing ingest behavior until a
  caller adopts it.
- **Known weak point**: the module is the mechanism, not yet the migration.
  Until `IngestionEngine.ingest`'s inline `fetch → parse → enrich → write`
  is actually rewired onto a `StagedPipeline`, the tail-latency problem this
  document describes (`staged_pipeline.py:13-20`) persists in production;
  the primitive existing and being tested does not by itself fix the
  blocking behavior it was built to remove.
