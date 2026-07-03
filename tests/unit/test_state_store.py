"""Tests for the unified state-store layer (CONCEPT:OS-5.16 / OS-5.17).

Covers backend selection (SQLite default vs Postgres via ``state_db_uri``),
the placeholder/row adaptation seam, idempotent schema migrations, the
cross-host claim guard, and daemon leadership election — all WITHOUT a live
Postgres (fake pool/connection stubs). Live-Postgres coverage is in
``tests/integration/test_state_postgres_live.py`` (gated on ``STATE_DB_URI``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from agent_utilities.core import leadership as leadership_mod
from agent_utilities.core import state_store
from agent_utilities.core.leadership import DaemonLeadership

# ── Fakes ────────────────────────────────────────────────────────────────


class FakeCursor:
    def __init__(self, conn: FakeRawConn):
        self._conn = conn
        self._rows: list[tuple] = []
        self.description: Any = None

    def execute(self, sql: str, params: Any = ()) -> FakeCursor:
        self._conn.calls.append((" ".join(sql.split()), tuple(params or ())))
        rows, description = self._conn.next_result()
        self._rows = rows
        self.description = description
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    @property
    def rowcount(self) -> int:
        return len(self._rows)


class FakeRawConn:
    """Records executed SQL; serves scripted results."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.results: list[tuple[list[tuple], Any]] = []
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def script(self, rows: list[tuple], columns: list[str] | None = None) -> None:
        desc = [(c,) for c in columns] if columns else None
        self.results.append((rows, desc))

    def next_result(self):
        if self.results:
            return self.results.pop(0)
        return [], None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def execute(self, sql: str, params: Any = ()) -> FakeCursor:
        return self.cursor().execute(sql, params)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self):
        self.conn = FakeRawConn()
        self.put_back: list[Any] = []

    @contextmanager
    def connection(self):
        yield self.conn

    def getconn(self):
        return self.conn

    def putconn(self, conn):
        self.put_back.append(conn)


@pytest.fixture(autouse=True)
def _clean_state_store():
    state_store._migrated_stores.clear()
    yield
    state_store._migrated_stores.clear()


@pytest.fixture
def pg_mode(monkeypatch):
    """Force the Postgres code paths with a fake pool (no network)."""
    pool = FakePool()
    monkeypatch.setattr(state_store, "postgres_state_enabled", lambda: True)
    monkeypatch.setattr(state_store, "state_pool", lambda: pool)
    return pool


# ── Backend selection ─────────────────────────────────────────────────────


def test_sqlite_default_when_uri_unset():
    assert state_store.state_db_uri() is None
    assert state_store.postgres_state_enabled() is False


def test_non_postgres_uri_is_ignored(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "state_db_uri", "mysql://nope", raising=False)
    assert state_store.state_db_uri() is None
    monkeypatch.setattr(
        config, "state_db_uri", "postgresql://u:p@h:5432/db", raising=False
    )
    assert state_store.state_db_uri() == "postgresql://u:p@h:5432/db"
    assert state_store.postgres_state_enabled() is True


def test_open_state_connection_sqlite(tmp_path):
    db = tmp_path / "s.db"
    import sqlite3

    sqlite3.connect(str(db)).executescript(
        "CREATE TABLE t (id TEXT, n INTEGER); INSERT INTO t VALUES ('a', 1);"
    )
    conn = state_store.open_state_connection("test", lambda: db)
    assert conn.dialect == "sqlite"
    cur = conn.cursor()
    cur.execute("SELECT * FROM t WHERE id = ?", ("a",))
    row = cur.fetchone()
    assert row["n"] == 1  # name access
    assert row[1] == 1  # positional access
    assert dict(row) == {"id": "a", "n": 1}
    conn.close()


def test_open_state_connection_postgres_adapts_placeholders(pg_mode):
    pool = pg_mode
    conn = state_store.open_state_connection(
        "test",
        lambda: "/nonexistent",
        postgres_ddl="CREATE TABLE IF NOT EXISTS t (id TEXT)",
    )
    assert conn.dialect == "postgres"
    pool.conn.script([("a", 1)], columns=["id", "n"])
    cur = conn.cursor()
    cur.execute("SELECT id, n FROM t WHERE id = ?", ("a",))
    # Placeholder translated for psycopg.
    assert any("WHERE id = %s" in sql for sql, _ in pool.conn.calls)
    row = cur.fetchone()
    # Hybrid rows: positional, by-name, and dict() like sqlite3.Row.
    assert row[0] == "a"
    assert row["n"] == 1
    assert dict(row) == {"id": "a", "n": 1}
    conn.close()
    assert pool.put_back  # returned to the pool, not closed


def test_ensure_state_schema_runs_once(pg_mode):
    pool = pg_mode
    ddl = "CREATE TABLE IF NOT EXISTS x (id TEXT)"
    state_store.ensure_state_schema("x", ddl, pool=pool)
    state_store.ensure_state_schema("x", ddl, pool=pool)
    ddl_calls = [c for c, _ in pool.conn.calls if "CREATE TABLE" in c]
    assert len(ddl_calls) == 1


# ── Claim guard ───────────────────────────────────────────────────────────


def test_claim_guard_noop_under_sqlite():
    # Must not raise nor touch any pool when the default backend is active.
    with state_store.state_claim_guard("kg-task-claim"):
        pass


def test_claim_guard_advisory_lock_under_postgres(pg_mode):
    pool = pg_mode
    pool.conn.script([(True,)])  # lock acquired
    with state_store.state_claim_guard("kg-task-claim"):
        pass
    sqls = [sql for sql, _ in pool.conn.calls]
    assert any("pg_advisory_lock" in s for s in sqls)
    assert any("pg_advisory_unlock" in s for s in sqls)
    key = state_store.advisory_key("kg-task-claim")
    assert pool.conn.calls[0][1] == (key,)


def test_advisory_key_stable_and_signed_64bit():
    k1 = state_store.advisory_key("leader:kg-maintenance")
    k2 = state_store.advisory_key("leader:kg-maintenance")
    assert k1 == k2
    assert -(2**63) <= k1 < 2**63
    assert k1 != state_store.advisory_key("leader:other-role")


# ── Daemon leadership (CONCEPT:OS-5.17) ───────────────────────────────────


def test_leadership_always_leader_under_sqlite_default():
    # No state_db_uri → per-host flock already enforces a single daemon.
    assert DaemonLeadership("kg-maintenance").is_leader() is True


def test_leadership_acquires_and_caches(monkeypatch):
    lead = DaemonLeadership("kg-maintenance", dsn="postgresql://fake/db")
    conn = FakeRawConn()
    conn.script([(True,)])  # pg_try_advisory_lock → won
    monkeypatch.setattr(lead, "_connect", lambda: conn)
    assert lead.is_leader() is True
    # Second poll only health-checks (SELECT 1), does not re-acquire.
    conn.script([(1,)])
    assert lead.is_leader() is True
    sqls = [sql for sql, _ in conn.calls]
    assert sqls.count("SELECT pg_try_advisory_lock(%s)") == 1


def test_leadership_follower_when_lock_held_elsewhere(monkeypatch):
    lead = DaemonLeadership("kg-maintenance", dsn="postgresql://fake/db")
    conn = FakeRawConn()
    conn.script([(False,)])  # another host holds it
    monkeypatch.setattr(lead, "_connect", lambda: conn)
    assert lead.is_leader() is False
    # Fail-over: next poll re-tries and can win.
    conn.script([(True,)])
    assert lead.is_leader() is True


def test_leadership_lost_on_connection_death(monkeypatch):
    lead = DaemonLeadership("kg-maintenance", dsn="postgresql://fake/db")
    good = FakeRawConn()
    good.script([(True,)])
    conns = [good]
    monkeypatch.setattr(lead, "_connect", lambda: conns.pop(0))
    assert lead.is_leader() is True

    # The lock connection dies → leadership must drop, then re-elect on a
    # fresh connection.
    def _boom(sql, params=()):
        raise ConnectionError("server closed the connection")

    good.execute = _boom  # type: ignore[assignment]
    fresh = FakeRawConn()
    fresh.script([(True,)])
    conns.append(fresh)
    assert lead.is_leader() is True
    assert good.closed is True


def test_leadership_unreachable_postgres_means_no_leader(monkeypatch):
    lead = DaemonLeadership("kg-maintenance", dsn="postgresql://fake/db")

    def _no_connect():
        raise ConnectionError("refused")

    monkeypatch.setattr(lead, "_connect", _no_connect)
    assert lead.is_leader() is False


def test_get_leadership_is_per_role_singleton():
    a = leadership_mod.get_leadership("role-a")
    b = leadership_mod.get_leadership("role-a")
    c = leadership_mod.get_leadership("role-b")
    assert a is b
    assert a is not c
