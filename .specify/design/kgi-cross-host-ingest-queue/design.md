# Design Document: The task queue's Postgres backend claims atomically so N hosts can drain it without double-claiming

CONCEPT:AU-KG.ingest.cross-host-safe-kg

> `agent_utilities/knowledge_graph/core/postgres_queue_backend.py` (primary),
> `agent_utilities/core/leadership.py` (the tick-classification contract this
> serves), `agent_utilities/knowledge_graph/core/engine_tasks.py:1739-1780`
> (the caller), `tests/unit/test_durable_state_postgres.py` (SKIP LOCKED
> semantics verified against an in-memory SQL emulation).

## Decision — replace the SQLite queue's per-process lock with an atomic `SELECT ... FOR UPDATE SKIP LOCKED` claim on the shared Postgres state store

`postgres_queue_backend.py:1-19` states the problem directly: the SQLite
queue's `get()` returns the head row *without removing it* and relies on a
per-process `threading.Lock` for claim atomicity — safe on one host, but two
hosts each see and process the same head row (a double-claim). The fix is
one atomic statement:

```sql
UPDATE ... WHERE id = (SELECT id ... FOR UPDATE SKIP LOCKED) RETURNING ...
```

so N hosts can drain the same queue concurrently and never double-claim.
Visibility-timeout semantics are preserved on top: a claimed-but-unacked item
(the claimer crashed before `ack`) becomes claimable again once its claim
ages past `_VISIBILITY_TIMEOUT_S` — the same at-least-once recovery contract
the SQLite backend had, now safe across hosts (`postgres_queue_backend.py:18-20`).

**The rejected alternative is the SQLite backend's own scheme** — a
per-process `threading.Lock` guarding a peek-then-update — kept as the
default for the single-host case (`state_db_uri` unset) precisely because it
is cheaper and correct there. It loses the moment a second host runs the
same daemon: both hosts' locks are independent, so both can observe and
process the same head row, producing duplicate ingestion work. This is not a
hypothetical the code guards defensively against — it is the literal
motivating bug the module docstring exists to fix, backed by
`leadership.py`'s explicit split of daemon jobs into **leader-only**
(whole-graph/singleton passes where N copies means duplicated LLM spend or
double writes) versus **per-host** (ingestion capacity: task workers,
submission-queue drain, graph-writer drain) — the per-host category is safe
to run on every host *specifically because* its queue claims are atomic
(`leadership.py:20-33`).

`engine_tasks.py:1774-1780` ties the two mechanisms together at the call
site: with `state_db_uri` set, a Postgres advisory lock elects exactly one
leader fleet-wide for the leader-only jobs; followers idle on those jobs but
still contribute per-host capacity through this cross-host-safe queue. Under
the SQLite default, `is_leader()` is always true because flock already
enforces a single per-host daemon — so the single-host case never pays for
cross-host coordination it doesn't need.

## Risk Assessment

- **Blast Radius**: `postgres_queue_backend.py`, `core/leadership.py`,
  `core/engine_tasks.py` (the maintenance-scheduler loop and per-host worker
  pool).
- **Backward Compatible**: Yes — SQLite remains the default single-host
  backend; Postgres activates only when `state_db_uri` is configured.
- **Breaking Changes**: None.
- **Known weak point**: correctness depends entirely on every claimer using
  the same atomic `FOR UPDATE SKIP LOCKED` statement; a future queue backend
  (or a caller that bypasses this module to touch the table directly) would
  silently reintroduce the double-claim bug this module exists to fix. The
  test suite (`test_durable_state_postgres.py`) only verifies the SQL shape
  against an in-memory emulation, not a live Postgres server, in CI.
