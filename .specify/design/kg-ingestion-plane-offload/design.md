# Design Document: The ingestion plane is separate from the serving plane

> Two decisions authored across `knowledge_graph/core/engine_tasks.py`,
> `knowledge_graph/core/task_lanes.py`, `knowledge_graph/core/worker_scheduler.py`
> and `core/resource_priority.py`. Backfilled under the concept-lineage rule
> (CONCEPT:AU-OS.governance.concept-lineage-parent-doc): twelve `AU-KG.compute`
> markers realise one of these two and point here.

CONCEPT:AU-KG.compute.offloaded-memory-write ·
CONCEPT:AU-KG.compute.interactive-lane-floor

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ORCH.execution.two-level-fair-rotation` | the lane partition + fair round-robin claim these decisions sit on top of | 0.70 | ORCH |
| `AU-ORCH.scheduling.resource-priority-edict` | the `PriorityClass` vocabulary the floor keys off; documented in prose at `docs/architecture/resource-priority-edict.md` | 0.65 | ORCH |
| `AU-KG.ingest.floor-codebase-admission-cap` | the admission cap floored at the engine's shard-writer width — the sibling decision in the same module | 0.50 | KG |

### Extension Analysis

- **Primary Extension Point**: `engine_tasks.py`'s task dispatch and
  `worker_scheduler.py`'s admission control.
- **Extension Strategy**: augment.
- **New Concept Required?**: No new ones. This names the two that already had
  markers.

## Decision 1 — a memory write is offloaded from the serving process

`CONCEPT:AU-KG.compute.offloaded-memory-write`

A `store_memory` call arriving on a serving/read path does not embed and write
inline. It is enqueued as a `kg_memory` task; the host worker performs the
embed + write (`_local=True`, so it never re-enqueues itself) on the ingestion
plane.

**The rejected alternative is the obvious one: write inline.** It is simpler and
it is what the code did first. It loses because embedding is the expensive,
GPU-contended step in the system, and doing it on the serving path puts a
live query behind a model call whose latency is set by whatever background
ingestion is currently loading the same GPU. Separating the planes means a read
never waits on an embed.

The cost is honest and accepted: a memory write is **not** durable-visible the
instant the call returns. Callers that need read-your-write get it via the task
handle, not by assuming synchrony.

### What the pointers to this decision are

`persistent-task-tracking` (the durable task record that makes the offload
observable), `reactive-push` and `p99-latency-metric` (how a caller learns the
offloaded work finished, and the tail-latency signal that says whether the
separation is working), `registered-edge-type` and
`per-channel-embedding-backfill` (background work that only exists because
there is a plane to run it on — the backfill's round-robin over channels is the
fairness rule *within* that plane, chosen over a single
`WHERE embedding IS NULL LIMIT n` FIFO that would let one channel monopolise it),
and `lane-bound-task` (a task carries the lane it belongs to).

## Decision 2 — interactive work gets a reserved worker floor

`CONCEPT:AU-KG.compute.interactive-lane-floor`

Lanes partition the task queue by functional domain and are claimed round-robin,
so no lane starves another. That alone is not enough: fair rotation still lets
background ingestion occupy every worker at the moment an interactive request
arrives.

So `INTERACTIVE` lanes hold a **reserved floor** of workers that non-interactive
lanes can never occupy. **The rejected alternative was strict priority** — the
scheme the queue actually used before lanes existed, where one ordering covered
all task types and a backed-up type head-of-line-blocked the rest (codebase
ingestion sat at 75-pending / 0-processed while `loop_cycle`/`research`
churned, visible only via a manual metrics dump).

Strict priority also has the opposite failure: it starves background work to
zero. The floor is deliberately a *floor*, not a monopoly — background work uses
spare capacity when nothing higher contends. The full law, its exception
(initial skill + MCP hydration is `HYDRATION`/rank 1, **not** deprioritised) and
the class table live in `docs/architecture/resource-priority-edict.md`; this
document records the decision, that file records the mechanism.

### What the pointers to this decision are

`lane-soft-timeout` (the per-lane soft execution bound that keeps a stuck task
from holding a floor slot), `task-priority-tag` (the tag a task carries into the
lane), and `priority-class-propagation` (carrying the class across a spawned
child agent and across the observability correlation context, so the edict does
not stop at a process boundary).

`resolve` was a third marker in this cluster: the scheduler asks the **live
engine** for its durable shard-writer width `K` rather than estimating from local
CPU count, because in split storage the engine may expose `K=4` from eight CPUs
while the scheduling host sees sixteen and over-admits. That is a real decision;
it belongs to `AU-KG.ingest.floor-codebase-admission-cap`, which already states
it, and the bare id `resolve` named nothing — so the marker is retired rather
than pointed.

## Data Flow

1. **ORCH**: lanes are claimed round-robin; the floor is enforced at admission.
2. **KG**: offloaded writes embed and persist on the ingestion plane.
3. **AHE**: none directly.
4. **ECO**: MCP write tools enqueue rather than write.
5. **OS**: `PriorityClass` propagates through the observability correlation
   context.

## Risk Assessment

- **Blast Radius**: `engine_tasks.py`, `task_lanes.py`, `worker_scheduler.py`,
  `core/resource_priority.py`, `observability/correlation.py`,
  `mcp/tools/write_ingest_tools.py`.
- **Backward Compatible**: Yes at the API level; **not** at the timing level —
  a caller that assumed a synchronous memory write sees an enqueue.
- **Breaking Changes**: none in signature; the read-your-write assumption is the
  one behavioural change and is called out above.
