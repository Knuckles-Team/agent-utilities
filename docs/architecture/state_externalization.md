# Durable-State Externalization & Multi-Host Operation

**Concepts:** AU-OS.state.unified-durable-state-externalization (unified state store), AU-OS.state.cross-host-daemon-leadership (daemon leadership),
AU-OS.state.fleet-supervisory-plane-at (fleet supervisory plane at scale), AU-KG.ingest.cross-host-safe-kg (cross-host task queue),
AU-ORCH.session.durable-goal-registry-goals (durable goal registry), ORCH-1.45 (queue-driven agent dispatch —
see [Queue-Driven Agent Dispatch](agent_dispatch.md))

## The problem

The platform's operational support state historically lived in per-host SQLite
files:

| Store | File | Consumer |
|---|---|---|
| Sessions / turns / fleet metadata | `agent_terminal_ui.db` | `core/sessions.py`, `gateway/fleet.py` |
| Queue delivery + staging rows | XDG-scoped queue stores | `knowledge_graph/core/engine_tasks.py`, agent dispatch |

Per-host files mean a second host cannot safely participate in those support
planes: delivery claims can double-fire, sessions are invisible across hosts,
and the gateway remains stateful. Durable execution itself is not in this
state-store layer. Its authoritative lifecycle and `checkpoint_id` live on the
engine-native `WorkItem`.

## One flag: `STATE_DB_URI`

`AgentConfig.state_db_uri` (alias `STATE_DB_URI`) selects the backend for the
operational support stores at once:

- **Unset (default)** — zero-infra per-host SQLite support stores. Tests and
  development environments need no external state service.
- **`postgresql://…`** — the support stores move onto one shared Postgres through a
  single `psycopg_pool.ConnectionPool` (sized by `STATE_DB_POOL_SIZE`,
  default 8).
  Schema is managed by lightweight idempotent `CREATE TABLE IF NOT EXISTS`
  migrations on first connect.

The seam is `agent_utilities/core/state_store.py`:

- `open_state_connection(store, sqlite_path, postgres_ddl)` — DB-API-ish
  connection that translates `?` placeholders to `%s` and yields rows
  addressable by index *and* name, so the existing SQLite SQL runs unchanged
  on both backends.
- `state_claim_guard(name)` — cross-host critical section (Postgres session
  advisory lock; no-op under SQLite).
- `ensure_state_schema(store, ddl)` — once-per-process idempotent migrations.

## What changes by plane

- **Durable execution** — unchanged by `STATE_DB_URI`. The native WorkItem owns
  claim, lease, fencing, `checkpoint_id`, idempotency, retry, and terminal
  result. Restart or redelivery resumes from that same engine record; no
  checkpoint sidecar exists.
- **Sessions / turns** — the selected support backend stores conversational
  session and turn records plus fleet visibility. Goal definitions and UX
  projections may be cached here, but their writable lifecycle remains the
  native WorkItem.
- **Queue delivery** — `PostgresTaskQueue` (AU-KG.ingest.cross-host-safe-kg) claims with
  `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED) RETURNING …`, so N
  hosts drain one delivery queue without double-claims. A claimed-but-unacked item
  becomes claimable again after the visibility timeout (600 s) — the same
  at-least-once recovery the SQLite head-until-ack behavior provided. The
  envelope references the sole WorkItem; queue state never becomes a second
  work lifecycle.

## Daemon leadership (AU-OS.state.cross-host-daemon-leadership)

With N hosts, each host's flock only de-duplicates daemons *per host*.
`core/leadership.py` adds fleet-wide election: `DaemonLeadership(role)` holds
a Postgres session advisory lock (`pg_try_advisory_lock`, stable per-role
key). A crashed leader's lock is released server-side with its connection;
followers re-try on every poll, so fail-over happens within one tick.
Under the SQLite default `is_leader()` is always true — single-host behavior
unchanged.

### Tick classification

- **Leader-only** — the scheduler tick that materializes due recurring work.
  N copies would enqueue duplicate work.
- **Per-host (capacity scaling)** — ingestion workers. Native WorkItem claims,
  leases, fencing, dependency release, and expired-lease recovery are atomic
  across hosts (AU-KG.ingest.cross-host-safe-kg).

## Queue-driven agent dispatch (ORCH-1.45)

State externalization made sessions and fleet metadata *visible* on every host; queue-driven
dispatch makes them *executable* on every host. An agent turn (goal run /
orchestrator job)
rides the session-keyed `agent_turns` queue and any host's
`agent-dispatch-worker` claims it, rehydrates from this shared state store,
executes the existing goal/agent bodies, and writes back — sessions are no
longer pinned to their birth host. The workers' liveness registry is one more
table in this store (`dispatch_workers`, surfaced by `/api/fleet/topology`),
and per-session mutual exclusion reuses `state_claim_guard`
(`agent-session:<id>` advisory locks). Full design:
[Queue-Driven Agent Dispatch](agent_dispatch.md).

## Fleet supervisory plane at scale (AU-OS.state.fleet-supervisory-plane-at)

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
against in-memory emulations of exactly the SQL each support backend issues
(`tests/unit/test_state_store.py`, `tests/unit/test_goal_durability.py`,
`tests/unit/test_fleet_supervisory.py`).
A live end-to-end pass (`tests/integration/test_state_postgres_live.py`) runs
only when `STATE_DB_URI` is set and reachable. WorkItem checkpoint behavior is
covered against the engine-native transaction surface, independently of the
support-state backend.
