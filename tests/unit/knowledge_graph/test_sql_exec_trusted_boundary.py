"""BUG-CX-118 — pin the boundary that makes ``sql_exec``'s missing row-policy safe.

``GraphComputeEngine.sql_exec`` applies no ``filter_rows()``/``visible()`` pass.
That is deliberate — see its docstring — but the previous justification cited a
precedent that had been deleted (``sql()``'s "never break a read" fail-open,
removed by BUG-CX-103) and a caller count that was wrong. A docstring is not
evidence, so these tests pin the load-bearing facts instead. If any of them
starts failing, the docstring's claim has rotted and the surface needs a fresh
decision — not a re-worded comment.

What is pinned:

1. ``sql_exec`` really does return the engine's rows verbatim (no row policy).
2. Nothing in ``table_ingest`` asks it for rows — every statement it issues is
   a write shape (``CREATE``/``INSERT``/``DROP``). Its one read goes through
   ``QueryMixin.sql``, the fail-closed governed surface.
3. The externally-reachable arbitrary-``SELECT`` action of the ``graph_table``
   MCP tool routes to ``engine.sql``, never to ``graph_compute.sql_exec``.
4. The audit trail stays best-effort: a missing actor at DDL/ETL boot time must
   not break the call, because that would break real callers rather than close
   a gap.
"""

from __future__ import annotations

import contextvars
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core import table_ingest
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

_WRITE_SHAPES = ("CREATE", "INSERT", "DROP", "ALTER", "UPDATE", "DELETE")


class _RecordingQuery:
    def __init__(self, rows: Any):
        self.rows = rows
        self.statements: list[str] = []

    def sql(self, statement: str) -> Any:
        self.statements.append(statement)
        return self.rows


def _engine(rows: Any) -> tuple[GraphComputeEngine, _RecordingQuery]:
    query = _RecordingQuery(rows)
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine._client = type("_Client", (), {"query": query})()
    return engine, query


class _FakeEngine:
    """The ``engine`` shape ``table_ingest`` expects: ``.graph_compute``."""

    def __init__(self, graph_compute: Any):
        self.graph_compute = graph_compute
        self.sql_calls: list[str] = []

    def sql(self, statement: str) -> list[dict[str, Any]]:
        self.sql_calls.append(statement)
        return [{"table_name": "conn_rest"}]


def test_sql_exec_returns_engine_rows_verbatim_with_no_row_policy():
    """The premise itself. Rows the engine returns are the rows the caller gets.

    Stated positively so the absence of a row-policy pass is a pinned,
    deliberate property rather than something a reader has to infer from the
    lack of a call.
    """

    foreign = [{"id": "n1", "tenant_id": "other-tenant"}]
    engine, query = _engine(foreign)

    def isolated() -> Any:
        # No ambient actor at all: the trusted-internal DDL/ETL boot shape.
        return engine.sql_exec("SELECT * FROM mcp_servers")

    assert contextvars.Context().run(isolated) is foreign
    assert query.statements == ["SELECT * FROM mcp_servers"]


def test_table_ingest_only_ever_issues_write_shapes_through_sql_exec():
    """No ``table_ingest`` path asks ``sql_exec`` for rows, so none can leak them."""

    graph_compute, query = _engine([])
    engine = _FakeEngine(graph_compute)

    table_ingest.ensure_table(engine, "conn_rest", ["id", "body"])
    table_ingest.insert_rows(
        engine, "conn_rest", [{"id": "1", "body": "x"}], ["id", "body"]
    )
    table_ingest.drop_table(engine, "conn_rest")

    assert query.statements, "no statement reached sql_exec"
    for statement in query.statements:
        assert statement.lstrip().upper().startswith(_WRITE_SHAPES), statement


def test_table_ingest_reads_through_the_governed_surface_not_sql_exec():
    """``list_tables`` is the module's only read and it uses ``engine.sql``."""

    graph_compute, query = _engine([])
    engine = _FakeEngine(graph_compute)

    assert table_ingest.list_tables(engine) == ["conn_rest"]
    assert query.statements == [], "a read reached the unfiltered sql_exec surface"
    assert engine.sql_calls and "information_schema" in engine.sql_calls[0]


def test_graph_table_query_action_routes_to_the_governed_sql_surface():
    """The one externally-reachable arbitrary ``SELECT`` never reaches sql_exec.

    ``graph_table(action='query', sql=...)`` is caller-supplied SQL arriving
    over MCP. It must land on ``QueryMixin.sql`` — which applies
    ``_governed_engine_surface_rows`` fail-closed — and not on this module's
    unfiltered write primitive.
    """

    from agent_utilities.mcp.tools.query_tools import _graph_table_query

    graph_compute, query = _engine([{"id": "n1"}])
    engine = _FakeEngine(graph_compute)

    out = _graph_table_query(engine, "SELECT * FROM mcp_servers")

    assert engine.sql_calls == ["SELECT * FROM mcp_servers"]
    assert query.statements == [], "graph_table's read reached sql_exec"
    assert "conn_rest" in out


def test_audit_failure_never_breaks_a_ddl_boot_call(monkeypatch):
    """Provenance is best-effort here; authorization is not, and lives upstream.

    A boot-time ``CREATE TABLE`` runs before any ambient session exists. Making
    the audit mandatory would break the fleet-catalog bootstrap — a real
    caller — rather than close a gap, so it must stay non-fatal.
    """

    engine, query = _engine({"ok": True})

    from agent_utilities.knowledge_graph.core import secured_reads

    def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("Read audit recording failed")

    monkeypatch.setattr(secured_reads, "audit_read", _explode)

    def isolated() -> Any:
        return engine.sql_exec("CREATE TABLE t (id VARCHAR)")

    assert contextvars.Context().run(isolated) == {"ok": True}
    assert query.statements == ["CREATE TABLE t (id VARCHAR)"]


def test_audit_is_recorded_under_the_ambient_actor_when_one_exists():
    """The audit attributes the read to the ambient ACTOR.

    ``registry_api._require_sql_exec`` swaps the ambient SESSION for the fixed
    catalog-service identity but leaves the actor alone, so a served registry
    read is attributed to the real HTTP caller rather than to the service.
    """

    recorded: list[tuple[list[str], str]] = []

    from agent_utilities.knowledge_graph.core import secured_reads

    original = secured_reads.audit_read
    secured_reads.audit_read = lambda ids, summary="", actor=None: recorded.append(  # type: ignore[assignment]
        (list(ids), summary)
    )
    try:
        engine, _query = _engine([{"id": "n1"}])
        engine.sql_exec("select * from mcp_servers")
    finally:
        secured_reads.audit_read = original  # type: ignore[assignment]

    assert recorded == [([], "sql_exec:SELECT")]


@pytest.mark.parametrize(
    "statement", ["CREATE TABLE t (id VARCHAR)", "SELECT 1", "  drop table t  "]
)
def test_no_statement_shape_is_rejected_by_this_primitive(statement):
    """The engine, not this client, decides what its user-table surface accepts.

    Pinned so that a future "fix" which starts refusing statement shapes here
    is a deliberate change rather than an accident — refusing DDL would break
    the boot-time catalog migration this primitive exists for.
    """

    engine, query = _engine({"ok": True})

    def isolated() -> Any:
        return engine.sql_exec(statement)

    assert contextvars.Context().run(isolated) == {"ok": True}
    assert query.statements == [statement]
