"""Server-enforced read boundary for the Apache AGE backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.backends.age_backend import AGEBackend


class _Cursor:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.description = [SimpleNamespace(name="n")]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, command: str) -> None:
        self.commands.append(command)
        if "MERGE" in command:
            raise RuntimeError("read-only transaction")

    def fetchall(self):
        return [('{"label":"Probe","properties":{"id":"node-1"}}::vertex',)]


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self) -> _Cursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


def _backend(cursor: _Cursor) -> AGEBackend:
    backend = object.__new__(AGEBackend)
    backend._graph_name = "agent_graph"
    connection = _Connection(cursor)
    backend._conn = lambda: connection
    return backend


def test_age_execute_read_uses_server_read_only_transaction() -> None:
    cursor = _Cursor()
    backend = _backend(cursor)

    assert backend.execute_read("MATCH (n) RETURN n") == [{"n": {"id": "node-1"}}]
    assert cursor.commands[0] == "SET TRANSACTION READ ONLY"
    assert "cypher(" in cursor.commands[-1]


def test_age_execute_read_propagates_server_write_rejection() -> None:
    cursor = _Cursor()
    backend = _backend(cursor)

    with pytest.raises(RuntimeError, match="read-only transaction"):
        backend.execute_read("MERGE (n:Probe {id: 'x'}) RETURN n")

    assert cursor.commands[0] == "SET TRANSACTION READ ONLY"
