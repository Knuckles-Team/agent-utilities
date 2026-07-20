# Worked Example: Queue-Driven Agent Dispatch (enqueue → worker → writeback)

**What this demonstrates.** CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — dispatching an orchestrator job
onto the durable, session-keyed `agent_turns` queue instead of executing it
in-process: the caller gets a job handle, any host running an
`agent-dispatch-worker` claims and executes the turn through the existing
execution paths, writes the result back durably, and the whole fleet is visible
through worker heartbeats, `/api/fleet/topology`, and Prometheus metrics. Deep
dive: [Agent dispatch architecture](../architecture/agent_dispatch.md).

**Prerequisites (ladder rung).** Works on the single-host rung of
[Deployment configurations](../guides/deployment-configurations.md) with zero
extra infrastructure (per-host SQLite queue). The multi-host story shown in the
AgentConfig projection below additionally needs the shared-state rung (`STATE_DB_URI` Postgres,
CONCEPT:AU-OS.state.unified-durable-state-externalization) or Kafka (`TASK_QUEUE_BACKEND=kafka`).

---

## 1. Configuration

One typed flag flips dispatch from in-process to queue-backed
(`AgentConfig.agent_dispatch_backend` in `agent_utilities/core/config.py`;
values `inline` | `queue`):

The `KEY=value` notation below names aliases to project into the gateway and every
worker process. It is not a repository dotenv file.

```text

# ORCH-1.45: dispatch returns a job handle; a worker fleet executes.
# Dispatch is queue-only; select its transport with TASK_QUEUE_BACKEND.

# Queue transport (KG-2.55 resolution, shared with the ingest plane):
#   unset      -> auto (Postgres with STATE_DB_URI, otherwise per-host SQLite)
#   sqlite     -> per-host file data_dir()/agent_dispatch_queue.db (zero-infra)
#   postgres   -> SKIP LOCKED claims on the agent_dispatch_queue table of the state store
#   kafka      -> keyed `agent_turns` topic, consumer group `agent-dispatch`
TASK_QUEUE_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=kafka.example.test:9092
# Partitions ensured on `agent_turns` (grow-only). Bounds how many sessions
# can execute concurrently across the whole worker fleet.
AGENT_TURNS_PARTITIONS=6
# Fail-closed admission bound. A failed depth probe is never treated as zero.
AGENT_DISPATCH_MAX_DEPTH=100000
# Crash-recovery lease: the TTL is capped at the published 300-second RTO.
AGENT_DISPATCH_CLAIM_TTL_S=120
AGENT_DISPATCH_RENEW_INTERVAL_S=30

# AU-OS.state.unified-durable-state-externalization shared state store: goals/sessions + the dispatch_workers fleet
# registry. With Postgres here, every gateway sees every host's workers and
# the per-session advisory lock is fleet-wide.
STATE_DB_URI=${STATE_DB_URI_FROM_SECRET}

# Workers run as engine CLIENTS (the worker forces KG_DAEMON_ROLE=client
# itself). Point them at the engine host:
GRAPH_SERVICE_ENDPOINTS=tls://engine-host.example.test:9474
ENGINE_TLS_PROFILE_REF=vault://platform/engine/tls-profile
GRAPH_SERVICE_AUTH_SECRET=${GRAPH_SERVICE_AUTH_SECRET_FROM_SUPERVISOR} # OS-5.14
```

An explicitly selected `kafka`/`postgres` transport that is unreachable raises
`TaskQueueUnavailable` at startup — never a silent SQLite degrade.

## 2. Enqueue: `graph_jobs action=dispatch`

```json
{
  "tool": "graph_orchestrate",
  "arguments": {
    "action": "dispatch",
    "task": "Summarize yesterday's ingest failures and file remediation topics",
    "dependencies": "[]"
  }
}
```

What happens: a deterministic orchestrator WorkItem is created, then a small
`AgentTurnEnvelope` goes onto the queue and the caller gets a **job handle**
instead of an execution promise. The envelope cannot create or adopt work.

**Expected output** (queue mode; in `inline` mode the same call returns the
plain string `"Task dispatched. Job ID: orch-1a2b3c4d"`):

```json
{
  "job_id": "orch-1a2b3c4d",
  "session_id": "orch-1a2b3c4d",
  "kind": "orchestrator_task",
  "dispatch": "queued",
  "status": "pending",
  "status_url": "/api/graph/jobs",
  "status_request": {"action": "status", "job_id": "orch-1a2b3c4d"}
}
```

A bare dispatch has no session, so the job id is its own session scope — serial
with itself, parallel with everything else.

### The envelope on the wire

`AgentTurnEnvelope` (`agent_utilities/orchestration/agent_dispatch.py`) carries
references, never bodies:

```json
{
  "job_id": "orch-1a2b3c4d",
  "session_id": "orch-1a2b3c4d",
  "kind": "orchestrator_task",
  "payload_ref": "orch-1a2b3c4d",
  "agent_name": "",
  "tenant": "",
  "priority": "normal",
  "deadline_unix": null,
  "attempt": 0,
  "enqueued_at": 1781234567.89
}
```

`kind` is `goal_loop` or `orchestrator_task`; `job_id` doubles as the
idempotency key; `payload_ref` addresses the sole WorkItem.

### Session-keyed ordering

`partition_key_for`
(`agent_utilities/knowledge_graph/core/kafka_queue_backend.py`) keys the
message with an opaque `session:<reference>` — a session key outranks everything, including
the ambient tenant key, so all turns of one session land on one partition and
execute serially (turn N+1 reads the state turn N wrote), while distinct
sessions parallelize across `AGENT_TURNS_PARTITIONS`.

## 3. Run the worker fleet

The console script (registered in `pyproject.toml` `[project.scripts]`) is
`agent-dispatch-worker`; its only CLI argument is `--workers`:

```bash
# On any host that can reach the queue + engine:
agent-dispatch-worker --workers 2
# equivalently:
python -m agent_utilities.orchestration.agent_dispatch_worker --workers 2
```

Startup behavior (`agent_utilities/orchestration/agent_dispatch_worker.py`):

- Forces `KG_DAEMON_ROLE=client` — workers never contend for the KG host flock.
- Verifies native `ClaimWorkItem` availability and preflights WorkItem reads;
  failure
  exits with code 2 and a message naming `GRAPH_SERVICE_ENDPOINTS` (external)
  or the packaged-local transport and the OS-5.14 shared HMAC
  secret — a worker that cannot reach the engine fails loud instead of claiming
  turns and dropping them.
- Each consumer uses an opaque worker token; host names and local identities are
  not persisted.

Native WorkItem claim, renewable lease, epoch, and fencing token prevent a
redelivery from committing over the current holder. Expired lease recovery is
part of native claim. Ack/offset-commit happens strictly after the turn
finishes or is durably marked failed.

## 4. Heartbeats and fleet visibility

Every 30s (and at claim time) each worker upserts a liveness row into the
`dispatch_workers` table of the sessions store
(`record_dispatch_worker_heartbeat`); rows with heartbeats older than 90s are
excluded. `GET /api/fleet/topology` (`agent_utilities/gateway/fleet.py`)
surfaces them:

```json
{
  "domains": [],
  "goals": [],
  "dispatch_workers": [
    {
      "worker_id": "worker-a:41233:agent-dispatch:0",
      "host": "worker-a",
      "capacity": 1,
      "active_sessions": ["orch-1a2b3c4d"],
      "queue_backend": "KafkaQueueBackend",
      "started_at": 1781234500.12,
      "last_heartbeat": 1781234590.44
    },
    {
      "worker_id": "worker-a:41233:agent-dispatch:1",
      "host": "worker-a",
      "capacity": 1,
      "active_sessions": [],
      "queue_backend": "KafkaQueueBackend",
      "started_at": 1781234500.15,
      "last_heartbeat": 1781234590.47
    }
  ],
  "totals": {"domains": 0, "sessions": 0, "dispatch_workers": 2},
  "page": {"limit": 200, "offset": 0, "returned": 0}
}
```

`active_sessions` is the placement answer: which host is running which session
right now. Placement is queue-pull (workers claim when they have capacity) —
there is no central placer to fail or rebalance.

## 5. Poll the job

```json
{
  "tool": "graph_orchestrate",
  "arguments": {"action": "status", "job_id": "orch-1a2b3c4d"}
}
```

REST twin: `POST /api/graph/jobs` with `{"action":"status","job_id":"orch-1a2b3c4d"}`.

**Expected output.** The `status` action renders the WorkItem. Lease capability
material is not copied into another node or exposed as job metadata:

```python
# while executing
{'kind': 'orchestrator_task', 'status': 'running',
 'payload_ref': 'orch-1a2b3c4d'}

# after native commit
{'kind': 'orchestrator_task', 'status': 'succeeded',
 'payload_ref': 'orch-1a2b3c4d', 'result_ref': 'outcome:...'}
```

An unknown job returns `{'status': 'not_found', 'error': 'Job ... not found'}`.

## 6. Metrics

Defined in `agent_utilities/observability/gateway_metrics.py`, refreshed on the
worker heartbeat tick (no-ops without the `metrics` extra):

| Series | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `agent_utilities_dispatch_queue_depth` | gauge | `backend` | Unclaimed turns in `agent_turns` (Kafka = consumer-group lag; Postgres/SQLite = row count) |
| `agent_utilities_dispatch_turns_total` | counter | `outcome` | Processed turns by outcome: `completed` / `failed` / `skipped` (duplicate delivery) / `expired` (deadline passed) |
| `agent_utilities_dispatch_workers` | gauge | — | Live workers (fresh heartbeats in the fleet registry) |

## What landed in the KG / state store

```text
(:WorkItem {id: "workitem:orchestrator:orch-1a2b3c4d", status, payload_ref,
            lease_epoch, result_ref})        # sole durable work authority
dispatch_workers table rows                  # fleet registry (sessions store)
agent_turns queue                            # transient envelopes (refs only)
```

---

*Verification: smoke-run against this tree (2026-06-11). Executed:
`python3 -m pytest tests/unit/test_agent_dispatch.py
tests/unit/test_workflow_lineage_closeout.py
tests/unit/knowledge_graph/test_workflow_gate.py -q` — 50 passed (the dispatch
suite covers envelope round-trip, queue-mode handles, session partition keys,
claims/redelivery/stale re-claim, heartbeats and the topology surface). The
JSON handle, envelope and registry-row shapes above are taken from the code and
those tests; the topology/metrics values shown are illustrative (no live
multi-host fleet was run).*
