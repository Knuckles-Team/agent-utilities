# Durable-State Externalization & Multi-Host Operation

**Concepts:** OS-5.16 (unified state store), OS-5.17 (daemon leadership),
OS-5.18 (fleet supervisory plane at scale), KG-2.54 (cross-host task queue),
ORCH-1.44 (durable goal registry), ORCH-1.45 (queue-driven agent dispatch —
see [Queue-Driven Agent Dispatch](agent_dispatch.md))

## The problem

The platform's durable state historically lived in three per-host SQLite
files:

| Store | File | Consumer |
|---|---|---|
| Durable-execution checkpoints | `durable_execution.db` | `orchestration/durable_execution.py` |
| Sessions / turns / goals | `agent_terminal_ui.db` | `core/sessions.py`, `gateway/fleet.py` |
| KG task + staging queue | `kg_task_queue.db` | `knowledge_graph/core/engine_tasks.py` |

Per-host files mean a second host cannot safely participate (queue claims
double-fire, sessions are invisible across hosts, goals die with the gateway
process) and the gateway is stateful.

## One flag: `STATE_DB_URI`

`AgentConfig.state_db_uri` (alias `STATE_DB_URI`) selects the backend for ALL
three stores at once:

- **Unset (default)** — the zero-infra per-host SQLite files, byte-for-byte
  the previous behavior. Tests and dev environments need no infrastructure.
- **`postgresql://…`** — every store moves onto one shared Postgres through a
  single `psycopg_pool.ConnectionPool` (sized by `STATE_DB_POOL_SIZE`,
  default 8 — the same psycopg driver the KG `PostgreSQLBackend` uses).
  Schema is managed by lightweight idempotent `CREATE TABLE IF NOT EXISTS`
  migrations on first connect, the same convention as the Postgres checkpoint
  backend.

The seam is `agent_utilities/core/state_store.py`:

- `open_state_connection(store, sqlite_path, postgres_ddl)` — DB-API-ish
  connection that translates `?` placeholders to `%s` and yields rows
  addressable by index *and* name, so the existing SQLite SQL runs unchanged
  on both backends.
- `state_claim_guard(name)` — cross-host critical section (Postgres session
  advisory lock; no-op under SQLite).
- `ensure_state_schema(store, ddl)` — once-per-process idempotent migrations.

## What changes per store

- **Durable execution** (`DurableExecutionManager`) — backend-selectable
  `CheckpointStore` (SQLite or Postgres). The SQLite path no longer opens a
  connection per operation: one pooled connection per db file, lock-guarded.
  Idempotency-key exactly-once and resilience-policy at-least-once semantics
  are identical on both backends.
- **Sessions / goals** — `sessions`/`turns`/`goals` tables on the selected
  backend. `active_goals` / `background_goal_runs` are now an in-memory cache
  over the durable `goals` table (ORCH-1.44): every status change persists,
  and on restart this host's non-terminal goals are rehydrated as
  **`orphaned`** — visible and explicitly resumable, never silently lost.
- **KG task queue** — `PostgresTaskQueue` (KG-2.54) claims with
  `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED) RETURNING …`, so N
  hosts drain one queue without double-claims. A claimed-but-unacked item
  becomes claimable again after the visibility timeout (600 s) — the same
  at-least-once recovery the SQLite head-until-ack behavior provided. The
  Task-node claim in the worker loop is additionally serialized fleet-wide by
  `state_claim_guard("kg-task-claim")`.

## Daemon leadership (OS-5.17)

With N hosts, each host's flock only de-duplicates daemons *per host*.
`core/leadership.py` adds fleet-wide election: `DaemonLeadership(role)` holds
a Postgres session advisory lock (`pg_try_advisory_lock`, stable per-role
key). A crashed leader's lock is released server-side with its connection;
followers re-try on every poll, so fail-over happens within one tick.
Under the SQLite default `is_leader()` is always true — single-host behavior
unchanged.

### Tick classification

- **Leader-only** — everything in the consolidated maintenance scheduler
  (analysis, golden loop, failure ingest, anomaly consumer, fuseki publish,
  compaction, evolution, durable reconcile, enrichment, SDD/file watch,
  hygiene, task reaper) plus the embedding-backfill drain. These are
  whole-graph/singleton passes: N copies = duplicated LLM spend or double
  writes.
- **Per-host (capacity scaling)** — ingestion task workers, the
  submission-queue drain, and the graph-writer drain. Safe to scale out
  because their claims are cross-host atomic (KG-2.54).

The task reaper also degrades to conservative age-based reaping under
multi-host state (a foreign claim token no longer proves a dead worker —
another live host may own it).

## Queue-driven agent dispatch (ORCH-1.45)

State externalization made sessions/goals *visible* on every host; queue-driven
dispatch makes them *executable* on every host. With
`AGENT_DISPATCH_BACKEND=queue` an agent turn (goal run / orchestrator job)
rides the session-keyed `agent_turns` queue and any host's
`agent-dispatch-worker` claims it, rehydrates from this shared state store,
executes the existing goal/agent bodies, and writes back — sessions are no
longer pinned to their birth host. The workers' liveness registry is one more
table in this store (`dispatch_workers`, surfaced by `/api/fleet/topology`),
and per-session mutual exclusion reuses `state_claim_guard`
(`agent-session:<id>` advisory locks). Full design:
[Queue-Driven Agent Dispatch](agent_dispatch.md).

## Fleet supervisory plane at scale (OS-5.18)

`gateway/fleet.py` no longer scans every session row in Python:

- `/api/fleet/health` aggregates with `COUNT`/`GROUP BY` (status and
  JSON-derived domain) in SQL on both backends.
- `/api/fleet/topology` is paginated (`limit`/`offset`) and filterable
  (`status`), with totals from SQL aggregates.
- **pause/kill are desired-state writes**: sessions whose goal loop runs in
  the local process are cancelled in-process and finalized (fast path);
  under externalized state, remote sessions get
  `pause_requested`/`kill_requested`, which the owning host's goal loop
  reconciles on its next tick (`core.sessions._desired_session_action`) into
  `paused`/`cancelled`.

## Testing

No test requires a live Postgres. Unit suites exercise the Postgres logic
against in-memory emulations of exactly the SQL each backend issues
(`tests/unit/test_state_store.py`, `tests/unit/test_durable_state_postgres.py`,
`tests/unit/test_goal_durability.py`, `tests/unit/test_fleet_supervisory.py`).
A live end-to-end pass (`tests/integration/test_state_postgres_live.py`) runs
only when `STATE_DB_URI` is set and reachable — e.g. against the deployed
`kg-backbone_pggraph` service.
