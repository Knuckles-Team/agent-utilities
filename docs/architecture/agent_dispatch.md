# Queue-Driven Agent Dispatch

**Concept:** ORCH-1.45 — session-partitioned agent-turn queue consumed by a
stateless dispatch-worker fleet. Builds directly on OS-5.16 (state
externalization), KG-2.54–2.57 (the durable task-queue stack), OS-5.14
(worker auth) and OS-5.18 (the fleet supervisory plane).

## The problem

The cognitive scheduler (`core/cognitive_scheduler.py`) is an in-process
asyncio priority queue: `max_concurrent` agent turns per **process**, a
per-process `_processes` table, and no cross-host dispatch. Even with
sessions/goals externalized to Postgres (OS-5.16), *execution* stayed pinned
to the host that accepted the goal — a busy gateway could not hand a turn to
an idle peer, and the scheduler tier could not scale horizontally.

## The model

```
caller ──► enqueue seam ──► agent_turns queue ──► dispatch worker (any host)
            (job handle)     key = session_id        │
                                                     ├─ session_execution_guard
                                                     ├─ claim (idempotent)
                                                     ├─ EXISTING execution body
                                                     └─ durable writeback + ack
```

- **Envelope, not payload.** `AgentTurnEnvelope`
  (`orchestration/agent_dispatch.py`) carries `job_id` (the idempotency key),
  `session_id`, `kind` (`goal_loop` | `orchestrator_task`), `payload_ref`,
  tenant, priority, deadline. Bodies live in the durable stores the envelope
  references — the `goals`/`sessions` rows (the full goal spec is persisted
  in the session's `metadata_json`) or the `:Task` graph node.
- **Transport = the existing KG-2.55 stack.** `TASK_QUEUE_BACKEND`/auto picks
  Kafka (`agent_turns` topic, consumer group `agent-dispatch`), Postgres
  (SKIP LOCKED claims on `agent_dispatch_queue` in the shared state store), or
  the zero-infra per-host SQLite file. The same fail-loud contract applies:
  an explicitly selected kafka/postgres transport that is unreachable raises
  `TaskQueueUnavailable` instead of silently islanding the queue.
- **Enqueue seams.** `graph_orchestrate action=dispatch` and the goal
  machinery (`core/sessions.create_goal`) check one flag,
  `AGENT_DISPATCH_BACKEND` (`inline` | `queue`, default `inline`). Inline is
  the previous in-process behavior byte-for-byte. In queue mode the caller
  gets a **job handle** — poll `graph_orchestrate action=status` /
  `/api/graph/orchestrate/job/{job_id}` (orchestrator jobs) or the goals API
  (goal runs).
- **Worker** (`orchestration/agent_dispatch_worker.py`, console script
  `agent-dispatch-worker`): claims the referenced record, rehydrates state
  from the shared store, and executes through the **existing** bodies —
  `run_goal_loop` for goals, the orchestration manager's agent execution for
  orchestrator jobs (the same extraction discipline as the KG-2.57
  `kg-ingest` worker: relocate, never duplicate). Those bodies already write
  turns, iterations and final status back into the durable stores.

## Ordering: session beats tenant

`partition_key_for` (KG-2.56) gains `session:<id>` at the **top** of the key
hierarchy. Turn N+1 of a session reads the state turn N wrote — interleaving
two turns of one session corrupts the conversation, so per-session serial
execution is a *correctness* requirement, while tenant keying is only an
ordering/fairness grouping for ingest work. A session never spans tenants, so
session-keying cannot weaken tenant isolation. Distinct sessions spread
across partitions and execute in parallel; `AGENT_TURNS_PARTITIONS`
(default 6, grow-only) bounds fleet-wide session concurrency on Kafka.

## Delivery and idempotency guarantees

- **At-least-once.** The ack/offset-commit happens strictly AFTER a turn is
  processed or durably marked failed. A worker crash redelivers the envelope:
  Kafka group rebalance, Postgres visibility timeout (600 s), SQLite
  head-until-ack.
- **Idempotent claims.** `job_id`/`payload_ref` is the idempotency key. The
  claim check skips terminal jobs (duplicate delivery) and jobs whose
  `running` claim is fresh (a live worker owns them); a claim older than
  `CLAIM_TTL_S` (1 h) is presumed dead and **re-claimed** — crash → claim
  expiry → requeue, the ingest reaper pattern folded into the claim itself.
- **Per-session mutual exclusion.** `session_execution_guard` holds a
  process-local per-session lock plus the OS-5.16 Postgres advisory lock
  (`agent-session:<id>`) for the claim+execute+writeback cycle. At-least-once
  delivery can hand one session's turn to two workers; the guard guarantees
  exactly one executes, the other claims-and-skips. A crashed holder releases
  the advisory lock server-side, so recovery is redelivery + re-claim, never
  a stuck session.
- **Deadlines.** A turn consumed after `deadline_unix` is durably failed
  ("expired") without execution.

## Placement: queue-pull, no central placer

Workers **pull** turns when they have capacity; nothing pushes work at them.
At this stage that is the correct design, not a shortcut: the partitioned
queue already provides per-session serialization (the only hard placement
constraint), uniform load spreading across consumers, and automatic
rebalancing when workers join/leave — a central placer would add a
coordination point, a failure mode, and rebalance churn while enforcing
nothing the partition key doesn't already enforce. Affinity-aware placement
(HRW routing toward workers with warm session caches) is future work layered
on the same envelope, worth its complexity only once checkpoint-rehydration
cost dominates turn latency.

## Fleet visibility

- Workers heartbeat into the `dispatch_workers` table of the **sessions
  store** (the same registry surface OS-5.18 reads): worker id, host,
  capacity, active sessions, transport, liveness. Stale heartbeats
  (> 90 s) drop out.
- `/api/fleet/topology` returns `dispatch_workers` (+ a totals count).
- `graph_orchestrate job/{id}` shows **which worker/host executed**: the
  claim stamps `claimed_by`/`dispatch_host` on the `:Task` node and
  completion stamps `executed_by`; goal rows carry the worker token in
  `owner_host`.
- Metrics on the OS-5.23 registry:
  `agent_utilities_dispatch_queue_depth{backend}`,
  `agent_utilities_dispatch_turns_total{outcome}`,
  `agent_utilities_dispatch_workers`.

## Deployment shape

A horizontally scaled deployment is now four independent tiers:

| Tier | Scales by | State |
|---|---|---|
| Stateless gateways (`GATEWAY_WORKERS`, N hosts) | request volume | none (OS-5.16) |
| **N dispatch workers** (`agent-dispatch-worker`) | active agent turns | none — claims + shared store |
| M ingest workers (`kg-ingest-worker`, KG-2.57) | ingest backlog | none |
| Engine shards (`GRAPH_SERVICE_ENDPOINTS`, KG-2.58) + state Postgres + Kafka | resident population / events | durable |

Flags: `STATE_DB_URI` (required for multi-host dispatch — sessions must be
shared), `TASK_QUEUE_BACKEND=kafka` (or postgres), `AGENT_DISPATCH_BACKEND=queue`,
`AGENT_TURNS_PARTITIONS`. Single-host/dev needs none of them: inline dispatch
and the SQLite transport remain the zero-infra default.

## Testing

`tests/unit/test_agent_dispatch.py` — envelope round-trip, session-key
precedence, inline-mode live-path (unchanged behavior), queue-mode job
handles on both seams, worker claim/execute/writeback against a fake queue +
real SQLite sessions store, two-workers-one-session mutual exclusion,
stale-claim crash-requeue, deadline expiry, poison-envelope tolerance,
heartbeat/topology/metrics surfaces. No broker, Postgres or engine daemon
required.
