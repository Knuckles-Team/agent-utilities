# Design Document: Dead-letter must be loud and drainable; a superseded fact must stay inspectable — both preserve history instead of silently dropping it

> `agent_utilities/knowledge_graph/ingestion/dead_letter.py`,
> `agent_utilities/knowledge_graph/ingestion/supersession.py` — Track B of
> the "universal-ingestion program" (same introducing commit `bd8391f6`
> "feat(mcp): add graph_claims(action=evaluate) — a live entrypoint for
> governed promotion").

CONCEPT:AU-KG.ingest.dead-letter-drain ·
CONCEPT:AU-KG.ingest.fact-supersession

## Decision 1 — dead-lettered items get a real list/drain API, never a silent drop or a parallel queue

`dead_letter.py:1-20`.

**What already existed, and the gap**: a `WorkItem` already reaches a
durable `dead_letter` terminal status once its native retry/backoff budget
is exhausted (`commit_result`), and `ops_context.diagnose_ops` already
surfaced a dead-letter COUNT for operational health answers. What did NOT
exist: an actual `list`/`drain` API — a dead-lettered item was queryable
only as an aggregate count, with no way to see WHICH items or requeue one.

**The rejected alternative, named directly**: building a second queue or a
parallel DLQ store. Instead this module is built DIRECTLY over the existing
`WorkItem` label/read pattern (`work_item.get_work_item`) — the gap is an
API gap, not a missing storage layer.

**A load-bearing sub-decision**: draining is explicitly a DELIBERATE,
operator-initiated action, never automatic — an item does not silently
retry itself out of dead-letter status; a human/agent must actively decide
to requeue it, preserving the "loud, never silent" invariant this whole
track exists for.

## Decision 2 — a superseded fact is archived and interval-closed, never deleted

`CONCEPT:AU-KG.ingest.fact-supersession` — `supersession.py:1-15`.

**The rejected alternative, implicit in the requirement itself**: deleting a
superseded fact outright — the simplest implementation, and the one that
loses the evidence that retired it.

**The design chosen, assembled from existing pieces, not rebuilt**:
tombstoning via a `ChangeEnvelope(operation="delete")` through
`ingest_envelope` (see `.specify/design/kgi-change-envelope-atomic/design.md`)
already archives a node (`archived=True`, `archivedReason`) and closes its
bitemporal validity interval (`_stamp_ambient_valid_until`) WITHOUT deleting
it. Supersession reuses this exact primitive rather than introducing a
second retraction mechanism — a superseded fact remains a real, inspectable
node with its full history and the evidence that superseded it, just marked
archived and validity-closed.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/dead_letter.py`,
  `agent_utilities/knowledge_graph/ingestion/supersession.py`,
  `agent_utilities/mcp/tools/job_tools.py`,
  `agent_utilities/orchestration/work_item.py`.
- **Backward Compatible**: Yes — both build on existing primitives
  (`WorkItem` terminal status, `ChangeEnvelope` delete/archive) rather than
  introducing new storage.
- **Breaking Changes**: None.
- **Known weak point**: dead-letter drain being operator-initiated-only
  means a large backlog of dead-lettered items accumulates indefinitely
  without an automatic sweep — nothing forces attention to a growing
  backlog beyond whatever alerts on the aggregate count `diagnose_ops`
  already surfaced.
