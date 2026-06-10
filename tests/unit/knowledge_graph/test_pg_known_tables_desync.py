"""``ensure_label_table(force=True)`` must create the table even when the in-memory
``_known_tables`` cache wrongly claims it exists.

Regression: ``create_schema`` cached a table name even when its CREATE failed, so the
self-heal short-circuited (``if name in self._known_tables: return True``) and never
created e.g. ``idea_block`` — every document IdeaBlock write then spammed
``relation "idea_block" does not exist`` against pggraph. (CONCEPT:KG-2.8)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
)


class _FakeCursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def execute(self, sql: str, *_a: object) -> None:
        self._log.append(sql)

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_a: object) -> bool:
        return False


class _FakeConn:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._log)

    def commit(self) -> None:
        pass

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *_a: object) -> bool:
        return False


def _stub(known: set[str]) -> tuple[SimpleNamespace, list[str]]:
    log: list[str] = []
    return SimpleNamespace(_known_tables=known, _conn=lambda: _FakeConn(log)), log


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_force_creates_despite_stale_cache():
    # Cache wrongly claims idea_block exists; force must still run the DDL.
    stub, log = _stub({"idea_block"})
    assert PostgreSQLBackend.ensure_label_table(stub, "idea_block", force=True) is True
    assert any('CREATE TABLE IF NOT EXISTS "idea_block"' in s for s in log)


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_non_force_short_circuits_on_cache():
    stub, log = _stub({"idea_block"})
    assert PostgreSQLBackend.ensure_label_table(stub, "idea_block") is True
    assert log == []  # cached → no DDL


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_creates_and_caches_new_label():
    stub, log = _stub(set())
    assert PostgreSQLBackend.ensure_label_table(stub, "NewType") is True
    assert any('CREATE TABLE IF NOT EXISTS "NewType"' in s for s in log)
    assert "NewType" in stub._known_tables
