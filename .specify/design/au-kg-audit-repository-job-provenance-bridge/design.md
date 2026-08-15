# Design Document: Repository job provenance bridge (RMDD-19)

CONCEPT:AU-KG.audit.repository-job-provenance-bridge

> `agent_utilities/observability/repository_provenance.py` (the writer/query
> layer), `agent_utilities/observability/repository_metrics.py`,
> `agent_utilities/observability/gateway_metrics.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| KG-2.296 (`RunTrace`/`ToolCall` provenance ontology) | the canonical agent-run provenance graph this bridge projects onto | high | KG |

### Extension Analysis

- **Primary Extension Point**: the existing `RunTrace -[:USED_TOOL]-> ToolCall`
  ontology `orchestration/agent_runner.py` already writes for agent runs.
- **Extension Strategy**: augment — reuse the same node shapes for a second
  domain (repository-manager job lifecycle) instead of a parallel store.
- **New Concept Required?**: Yes — a generic, repository-domain-agnostic
  writer/query layer for this specific lifecycle vocabulary didn't exist yet.

## Problem

Repository Manager's job lifecycle (submit, lease/claim, admission,
start/heartbeat/checkpoint, cancel/retry/dead-letter, command result,
artifact publication, validation/certificate, candidate/generation/bisection,
concept, landing/push, GC/reconcile) had no audit trail in the knowledge
graph. RMDD-19's brief was explicit: "emit through existing graph authority;
do not introduce another store."

## Decision

`agent_utilities/observability/repository_provenance.py` is a thin additional
layer over the EXISTING `RunTrace`/`ToolCall` provenance ontology — no second
graph authority, no second audit store. One repository-job attempt projects
onto one `RunTrace` node (keyed deterministically from `work_item_id`/
`attempt`); every lifecycle event of that attempt projects onto a `ToolCall`
node linked to it, the same shape `orchestration/agent_runner.py` already
writes for agent runs.

- **One chokepoint**: every event kind funnels through
  `write_repository_event` — callers (repository-manager's domain emitters)
  never call `engine.add_node` directly.
- **Idempotency**: node ids are deterministic functions of
  `(work_item_id, attempt, kind, occurrence)` — never a random uuid4 or a
  wall-clock value — so replaying the same logical event upserts the SAME
  node rather than duplicating it. Callers derive `occurrence` from the
  event's own immutable identity, never a runtime/process-local counter.
- **Fail loud, never fabricate**: a caller that cannot reach the graph gets an
  explicit `RepositoryProvenanceUnavailable` rather than a silently-dropped
  write.

## Wire-First

`reconciliation_report`/`write_repository_event` are the module's two public
entrypoints; `repository_metrics.py` and `gateway_metrics.py` consume the
same provenance for the observability surface.
