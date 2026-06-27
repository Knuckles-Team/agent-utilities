"""ETL sink='table' → native engine SQL table (CONCEPT:KG-2.266).

run_etl routes ``sink='table'`` to ``ingest_connector_to_table`` instead of the
graph-store / writeback dispatch, mirroring the inbound source's connector data into
an engine SQL table.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.etl import pipeline

pytestmark = pytest.mark.concept("KG-2.266")


def test_run_etl_sink_table_routes_to_table_ingest(monkeypatch):
    captured = {}

    def fake_ingest(
        engine, source, *, table=None, config=None, limit=1000, replace=False
    ):
        captured.update(
            source=source, table=table, limit=limit, replace=replace, config=config
        )
        return {"status": "ok", "table": table or f"conn_{source}", "rows_written": 5}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.table_ingest.ingest_connector_to_table",
        fake_ingest,
    )

    out = pipeline.run_etl(
        object(),
        source="rest",
        sink="table",
        ops={"table": "rest_mirror", "limit": 50},
        record_lineage=False,
    )
    assert out["status"] == "ok"
    assert out["outbound"]["rows_written"] == 5
    assert captured["source"] == "rest"
    assert captured["table"] == "rest_mirror"
    assert captured["limit"] == 50


def test_run_etl_sink_table_partial_on_skip(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.table_ingest.ingest_connector_to_table",
        lambda *a, **k: {"status": "skipped", "reason": "no engine SQL surface"},
    )
    out = pipeline.run_etl(object(), source="rest", sink="table", record_lineage=False)
    assert out["status"] == "partial"
    assert out["outbound"]["status"] == "skipped"
