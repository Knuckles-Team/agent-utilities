"""A failed statement on the SHARED single connection must roll back, not poison the txn.

Regression: ``_conn``'s ``_SingleConnPool`` branch ran commit/rollback AFTER the yield, so
an exception in the ``with`` body (raised AT the yield) skipped them — the connection was
left in an aborted transaction and EVERY subsequent write (incl. the auto-DDL self-heal's
CREATE TABLE + retry) cascaded "current transaction is aborted" (e.g. the EvictedBlock write
during a messaging turn poisoning all following Concept/edge/Session upserts).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
    _SingleConnPool,
)


class _RecordingConn:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def commit(self) -> None:
        self.calls.append("commit")

    def rollback(self) -> None:
        self.calls.append("rollback")


def _backend_with(conn: _RecordingConn) -> SimpleNamespace:
    pool = _SingleConnPool(conn)
    # AU-P0-5: `_conn` now scopes every checkout to the ambient tenant via
    # `self._scope_tenant(conn)` before yielding — fake it as a no-op so this
    # test stays focused on the commit/rollback contract it's regressing on.
    return SimpleNamespace(_ensure_pool=lambda: pool, _scope_tenant=lambda conn: None)


def test_single_conn_commits_on_success() -> None:
    conn = _RecordingConn()
    stub = _backend_with(conn)
    with PostgreSQLBackend._conn(stub) as c:
        assert c is conn
    assert conn.calls == ["commit"]


def test_single_conn_rolls_back_and_reraises_on_exception() -> None:
    conn = _RecordingConn()
    stub = _backend_with(conn)
    with pytest.raises(ValueError):
        with PostgreSQLBackend._conn(stub) as c:
            assert c is conn
            raise ValueError('relation "EvictedBlock" does not exist')
    # rolled back (so the NEXT write runs clean), and NOT committed
    assert conn.calls == ["rollback"]


def test_subsequent_write_is_clean_after_a_failure() -> None:
    # The cascade scenario: a failed write then a good write reuse the SAME connection.
    conn = _RecordingConn()
    stub = _backend_with(conn)
    with pytest.raises(RuntimeError):
        with PostgreSQLBackend._conn(stub):
            raise RuntimeError("boom")
    conn.calls.clear()
    with PostgreSQLBackend._conn(stub):
        pass  # the next statement succeeds
    assert conn.calls == ["commit"]  # not "transaction is aborted"
