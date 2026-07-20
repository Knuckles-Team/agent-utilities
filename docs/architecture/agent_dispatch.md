# Queue-Driven Agent Dispatch

Agent execution is queue-only. A gateway never executes a goal or orchestrator
turn inline and there is no dispatch-backend selector.

## Authority model

```text
request
  -> fail-closed queue-depth admission check
  -> submit WorkItem
  -> atomically admit AgentTurnEnvelope under the depth bound
  -> worker claims WorkItem with lease epoch + fencing token
  -> execute with periodic lease heartbeat
  -> renew immediately before each side effect
  -> fenced WorkItem commit
  -> acknowledge queue delivery
```

`WorkItem` is the only writable lifecycle authority. Goal definitions, loop
definitions, session rows, outcomes, and provenance may describe work, but none
of them owns a second writable status or lease. The current state machine is:

```text
submitted -> ready -> leased -> running
    -> succeeded | failed | cancelled | dead_letter
```

Claims, renewals, dependency release, retries, cancellation, and terminal
commits use the engine-native WorkItem transaction surface. A commit is rejected
unless tenant, owner, lease epoch, and fencing token still match. Outcome and
provenance nodes are append-only evidence written around the authoritative
commit.

## Queue contract

`AgentTurnEnvelope` carries only `job_id`, `session_id`, `kind`, `payload_ref`,
tenant, priority, deadline, and attempt metadata. The referenced WorkItem or
session store owns the durable body.

The transport follows `TASK_QUEUE_BACKEND`:

- Kafka: `agent_turns`, consumer group `agent-dispatch`;
- PostgreSQL: `agent_dispatch_queue` with skip-locked claims;
- local development: an XDG-scoped SQLite queue.

An explicitly selected unavailable transport fails loudly. Kafka partitioning
uses `session_id`, so one session is serial while different sessions can run in
parallel. `AGENT_TURNS_PARTITIONS` is grow-only and bounds Kafka concurrency.
`AGENT_DISPATCH_MAX_DEPTH` bounds accepted durable turns. SQLite and PostgreSQL
perform compare-and-admit atomically; Kafka rejects from authoritative consumer
group lag and relies on broker quota/retention as its cross-producer hard bound.
An unavailable depth probe is never interpreted as an empty queue.

Dispatch lease recovery is typed and aligned with the workload contract:
`AGENT_DISPATCH_CLAIM_TTL_S` defaults to 120 seconds and is capped at the
300-second RTO; `AGENT_DISPATCH_RENEW_INTERVAL_S` defaults to 30 seconds. The
doctor reports this as `dispatch_lease_recovery`, and the lease guard renews both
periodically and immediately before every authoritative side effect.

Kafka producers coalesce concurrent submissions (or an explicit `put_many`
batch) behind one delivery-confirmed flush barrier. Consumers are thread-local:
each worker owns its consumer and the opaque acknowledgement receipt carries
that owner, removing the former process-wide poll/commit lock.

PostgreSQL stores only opaque tenant and fairness references in scheduling
columns. Claims order expired deadlines first for prompt cancellation, then
priority, least-recently-served tenant, least-recently-served fairness group,
deadline, enqueue time, and id. The schema is current-only; there is no FIFO
fallback or alternate legacy claim query. Its acknowledgement receipt includes
the opaque claim owner and timestamp, so a worker whose visibility claim was
reassigned cannot delete the newer worker's row. Fairness history is pruned when
its tenant/group has no remaining queued work.

## Delivery invariants

- Queue acknowledgement occurs only after a durable fenced WorkItem commit.
- Redelivery is expected; deterministic WorkItem/idempotency identifiers make it
  safe.
- Workers renew leases periodically while executing and synchronously before
  each bounded side effect. Executors receive a `WorkItemLeaseGuard` and must
  route mutations through `lease_guard.side_effect(...)`. A stale worker cannot
  commit after a newer lease epoch has been issued, and a fenced queue delivery
  is left unacknowledged for redelivery.
- `session_execution_guard` adds local locking and a fleet-wide PostgreSQL
  advisory lock when shared state is configured.
- Expired envelopes cancel their referenced WorkItem without writing status to a
  Goal, Loop, Task, or Concept projection.
- An orchestrator envelope must reference an existing WorkItem. Workers never
  infer, adopt, or manufacture authority from another node type.

## Operational surfaces

- Worker: `agent-dispatch-worker`
- Enqueue implementation: `agent_utilities/orchestration/agent_dispatch.py`
- Claim/execute/commit implementation:
  `agent_utilities/orchestration/agent_dispatch_worker.py`
- Work authority: `agent_utilities/orchestration/work_item.py`
- Queue controls: `TASK_QUEUE_BACKEND`, `AGENT_TURNS_PARTITIONS`,
  `AGENT_DISPATCH_MAX_DEPTH`, `AGENT_DISPATCH_CLAIM_TTL_S`,
  `AGENT_DISPATCH_RENEW_INTERVAL_S`
- Multi-host session state: `STATE_DB_URI`

Worker heartbeat, queue depth, success/failure, and lease failures are exposed
through fleet topology and gateway metrics. The doctor reports configuration and
transport failures without persisting hostnames, usernames, or filesystem paths.
