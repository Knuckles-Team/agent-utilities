"""Bound the PostgreSQL authority-write tail (CONCEPT:KG-2.152).

Profiling showed the authority (L3, pg-age) write path is the ingestion ceiling:
``write[authority:PostgreSQLBackend]`` p50 ~3ms but MAX 11.7-16.4s under sustained
codebase ingest, while the mirrors stayed p50 ~3ms / max <9ms and mirror lag was 0.
The tail was NOT the SQL — it was connection-pool starvation: every lane worker +
the fan-out drainer threads + reads share ONE bounded pool, and a starved acquire
blocked the psycopg_pool default 30s. Plus the AGE backend re-ran ``LOAD 'age'`` +
``SET search_path`` on every ``execute``, lengthening connection hold time.

These tests pin the three fixes without needing a live Postgres:
  1. a pool-acquire timeout (``PoolTimeout``) is classified as retryable contention,
  2. the AGE per-connection session prep runs ONCE per physical connection, not per
     ``execute`` (N writes => 1 prep, not N), and
  3. the pool ceiling is env-tunable (and defaults higher than the old 10).
"""

from __future__ import annotations

import importlib

import pytest

from agent_utilities.knowledge_graph.backends.age_backend import AGEBackend
from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
)


# ── 1. PoolTimeout is retryable contention (caps the tail) ───────────────────


class _PoolTimeout(Exception):
    """Stand-in matching psycopg_pool.PoolTimeout by class NAME."""


_PoolTimeout.__name__ = "PoolTimeout"


@pytest.mark.concept("KG-2.152")
def test_pool_timeout_is_retryable_contention():
    # A starved-pool acquire must be treated like a lock wait so the resilience
    # policy backs off + retries (bounded) instead of dropping the write — and so
    # the unbounded 30s queue wait is replaced by a fast retryable failure.
    assert PostgreSQLBackend._is_lock_contention(_PoolTimeout("timed out")) is True


@pytest.mark.concept("KG-2.152")
def test_pool_timeout_phrase_is_retryable():
    exc = Exception("couldn't get a connection after 5.0 sec")
    assert PostgreSQLBackend._is_lock_contention(exc) is True


@pytest.mark.concept("KG-2.152")
def test_schema_error_still_not_contention():
    # Regression guard: the new branches must not reclassify a schema-drift error.
    exc = Exception('column "x" of relation "idea_block" does not exist')
    exc.sqlstate = "42703"  # type: ignore[attr-defined]
    assert PostgreSQLBackend._is_lock_contention(exc) is False


# ── 2. AGE session prep runs ONCE per physical connection, not per execute ───


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.connection = conn

    def execute(self, sql: str, params=None):  # noqa: ANN001
        self.connection.calls.append(sql)


class _FakeConn:
    def __init__(self) -> None:
        self.calls: list[str] = []


@pytest.mark.concept("KG-2.152")
def test_age_session_prepared_once_per_connection():
    be = AGEBackend.__new__(AGEBackend)  # no real DSN/pool needed
    conn = _FakeConn()

    # N "writes" on the SAME pooled connection.
    for _ in range(50):
        be._age_session(_FakeCursor(conn))

    prep = [c for c in conn.calls if c.startswith("LOAD") or c.startswith("SET ")]
    # The two prep statements run exactly ONCE (not 2 * 50) — the rest of the
    # writes reuse the already-prepared connection. This cuts 2 round-trips off
    # every authority write after the first, shortening connection hold time.
    assert len(prep) == 2, prep
    assert getattr(conn, "_age_prepared", False) is True


@pytest.mark.concept("KG-2.152")
def test_age_session_reprepares_a_fresh_connection():
    be = AGEBackend.__new__(AGEBackend)
    c1, c2 = _FakeConn(), _FakeConn()
    be._age_session(_FakeCursor(c1))
    be._age_session(_FakeCursor(c2))  # a different physical connection
    # Each distinct connection is prepared exactly once.
    assert len([c for c in c1.calls if c.startswith(("LOAD", "SET "))]) == 2
    assert len([c for c in c2.calls if c.startswith(("LOAD", "SET "))]) == 2


# ── 3. Pool ceiling is env-tunable and defaults higher than the old 10 ───────


@pytest.mark.concept("KG-2.152")
def test_pool_defaults_raised_and_env_tunable(monkeypatch):
    import agent_utilities.knowledge_graph.backends as backends_pkg

    # Default (no env): the ceiling must be well above the old starving 10.
    monkeypatch.delenv("GRAPH_DB_POOL_MAX", raising=False)
    monkeypatch.delenv("GRAPH_DB_POOL_MIN", raising=False)
    reloaded = importlib.reload(backends_pkg)
    try:
        assert reloaded._PG_POOL_MAX >= 32
        assert reloaded._PG_POOL_MAX > 10
        assert reloaded._PG_POOL_MIN >= 1

        # Env override is honored.
        monkeypatch.setenv("GRAPH_DB_POOL_MAX", "64")
        monkeypatch.setenv("GRAPH_DB_POOL_MIN", "8")
        reloaded = importlib.reload(backends_pkg)
        assert reloaded._PG_POOL_MAX == 64
        assert reloaded._PG_POOL_MIN == 8
    finally:
        # Restore module defaults for other tests in the session.
        monkeypatch.delenv("GRAPH_DB_POOL_MAX", raising=False)
        monkeypatch.delenv("GRAPH_DB_POOL_MIN", raising=False)
        importlib.reload(backends_pkg)


@pytest.mark.concept("KG-2.152")
def test_pool_acquire_timeout_default_is_bounded(monkeypatch):
    from agent_utilities.knowledge_graph.backends import postgresql_backend as pgb

    monkeypatch.delenv("GRAPH_DB_POOL_TIMEOUT", raising=False)
    # Far below psycopg_pool's 30s default — the whole point of the fix.
    assert 0.5 <= pgb._pool_acquire_timeout_s() <= 10.0
    monkeypatch.setenv("GRAPH_DB_POOL_TIMEOUT", "2.5")
    assert pgb._pool_acquire_timeout_s() == 2.5

    # An instance carries the bounded timeout (used by _ensure_pool + _conn).
    be = PostgreSQLBackend.__new__(PostgreSQLBackend)
    PostgreSQLBackend.__init__(be, dsn="postgresql://x/y")
    assert be._pool_timeout > 0
