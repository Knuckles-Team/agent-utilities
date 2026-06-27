"""Connector / ETL → native engine SQL tables (CONCEPT:KG-2.266).

A fake engine records the SQL statements issued through ``graph_compute.sql_exec`` so
we can assert CREATE TABLE + bulk INSERT are emitted, and a fake connector verifies
the document-mirroring path.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import table_ingest

pytestmark = pytest.mark.concept("KG-2.266")


class _FakeGraphCompute:
    def __init__(self):
        self.statements = []

    def sql_exec(self, statement):
        self.statements.append(statement)
        return {"ok": True}


class _FakeEngine:
    def __init__(self, rows=None):
        self.graph_compute = _FakeGraphCompute()
        self._rows = rows or []

    def sql(self, query):
        return self._rows


def test_ensure_table_emits_create():
    eng = _FakeEngine()
    out = table_ingest.ensure_table(eng, "t1", ["id", "name"])
    assert out["created"] is True
    assert any(
        s.startswith("CREATE TABLE IF NOT EXISTS t1")
        for s in eng.graph_compute.statements
    )


def test_insert_rows_emits_batched_insert():
    eng = _FakeEngine()
    rows = [{"id": "a", "name": "x"}, {"id": "b", "name": "y'z"}]
    written = table_ingest.insert_rows(eng, "t1", rows, ["id", "name"])
    assert written == 2
    inserts = [
        s for s in eng.graph_compute.statements if s.startswith("INSERT INTO t1")
    ]
    assert len(inserts) == 1
    # single-quote in "y'z" is escaped
    assert "'y''z'" in inserts[0]


def test_invalid_identifier_rejected():
    eng = _FakeEngine()
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        table_ingest.ensure_table(eng, "bad name; DROP TABLE x", ["id"])


def test_ingest_connector_to_table_mirrors_documents(monkeypatch):
    class _Doc:
        def __init__(self, i):
            self.id = f"d{i}"
            self.source_uri = f"http://x/{i}"
            self.title = f"title {i}"
            self.text = f"body {i}"
            self.doc_type = "ticket"
            self.updated_at = "2026-01-01"
            self.metadata = {"k": i}

    class _Connector:
        def load(self):
            yield from (_Doc(0), _Doc(1), _Doc(2))

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        lambda source, config: _Connector(),
    )
    eng = _FakeEngine()
    out = table_ingest.ingest_connector_to_table(eng, "rest", table="my_tbl", limit=2)
    assert out["status"] == "ok"
    assert out["table"] == "my_tbl"
    assert out["rows_seen"] == 2  # limit honored
    assert out["rows_written"] == 2
    assert any(
        "CREATE TABLE IF NOT EXISTS my_tbl" in s for s in eng.graph_compute.statements
    )
    assert any("INSERT INTO my_tbl" in s for s in eng.graph_compute.statements)


def test_ingest_rows_infers_columns():
    eng = _FakeEngine()
    out = table_ingest.ingest_rows_to_table(
        eng, "etl_out", [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    )
    assert out["status"] == "ok"
    assert set(out["columns"]) == {"a", "b", "c"}
    assert out["rows_written"] == 2


def test_no_engine_surface_skips():
    out = table_ingest.ingest_connector_to_table(object(), "rest")
    assert out["status"] == "skipped"
