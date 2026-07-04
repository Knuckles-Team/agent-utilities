"""Auto-DDL must heal EVERY column an INSERT references in one pass (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

Regression: PostgreSQL reports only the first missing column per attempt, so healing
one column per retry exhausted ``max_retries=3`` when a node carried several
undeclared props (e.g. an arbitrary-label ``idea_block`` with label/trusted_answer/
source) — the node was then dropped instead of mirrored. ``_ensure_insert_columns``
adds them all so the retry succeeds immediately.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
)


class _Stub(PostgreSQLBackend):
    def __init__(self) -> None:  # bypass live-PG __init__
        self.added: list[tuple[str, str]] = []

    def ensure_column(self, table: str, column: str) -> bool:
        self.added.append((table, column))
        return True


SQL = (
    'INSERT INTO "idea_block" ("id", "label", "type", "name", "trusted_answer", '
    '"source") VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET '
    '"label" = EXCLUDED."label"'
)


def test_heals_all_referenced_columns_skipping_id():
    s = _Stub()
    assert s._ensure_insert_columns("idea_block", SQL) is True
    cols = [c for _, c in s.added]
    # every non-id column is ensured, in one pass
    assert cols == ["label", "type", "name", "trusted_answer", "source"]
    assert ("idea_block", "id") not in s.added


def test_non_insert_sql_is_noop():
    s = _Stub()
    assert s._ensure_insert_columns("t", "SELECT 1") is False
    assert s.added == []
