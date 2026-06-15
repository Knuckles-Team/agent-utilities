"""Unit tests for the unified ETL pipeline + provenance contract (KG-2.98 / KG-2.9).

Fakes the engine/sinks so source→transform→sink dispatch, provenance stamping, and
the graph-store vs write-back routing are asserted without a live backend.
"""

from __future__ import annotations

from unittest.mock import patch

from agent_utilities.knowledge_graph.enrichment.provenance import stamp_source
from agent_utilities.knowledge_graph.etl.pipeline import run_etl


# ── provenance contract ──────────────────────────────────────────────────────
def test_stamp_source_sets_both_keys_and_respects_caller():
    assert stamp_source({"a": 1}, "Egeria") == {
        "a": 1,
        "source_system": "egeria",
        "domain": "egeria",
    }
    assert stamp_source({}, None) == {}
    assert stamp_source({}, "") == {}
    # caller-supplied values win
    assert stamp_source({"domain": "keep"}, "leanix")["domain"] == "keep"


# ── write_batch stamps the contract when given a source ──────────────────────
def test_write_batch_stamps_source_system():
    from agent_utilities.knowledge_graph.enrichment.models import (
        EnrichmentEdge,
        ExtractionBatch,
        GraphNode,
    )
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch

    captured = []

    class _BE:
        def add_node(self, node_id, type=None, **props):
            captured.append(("node", node_id, props))

        def add_edge(self, s, t, rel_type=None, **props):
            captured.append(("edge", s, t, props))

    batch = ExtractionBatch(
        nodes=[GraphNode(id="a:1", type="Asset", props={"name": "x"})],
        edges=[EnrichmentEdge(source="a:1", target="a:2", rel_type="LINKS", props={})],
    )
    write_batch(_BE(), batch, source="egeria")
    node = next(c for c in captured if c[0] == "node")
    assert node[2]["source_system"] == "egeria" and node[2]["domain"] == "egeria"
    edge = next(c for c in captured if c[0] == "edge")
    assert edge[3]["source_system"] == "egeria"

    # No source → untouched (internal-fact batches).
    captured.clear()
    write_batch(_BE(), batch)
    assert "source_system" not in next(c for c in captured if c[0] == "node")[2]


# ── run_etl dispatch ─────────────────────────────────────────────────────────
class _Engine:
    def __init__(self):
        self.backend = object()


def test_run_etl_inbound_only_calls_sync_source():
    eng = _Engine()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.source_sync.sync_source",
            return_value={"status": "ok", "nodes_hydrated": 7},
        ) as sync,
        patch(
            "agent_utilities.knowledge_graph.etl.lineage.record_etl_run",
            return_value="run-1",
        ),
    ):
        out = run_etl(eng, source="leanix", mode="delta")
    sync.assert_called_once()
    assert out["inbound"]["status"] == "ok"
    assert out["outbound"] is None
    assert out["lineage"]["direction"] == "inbound"


def test_run_etl_outbound_to_sparql_backend_pushes():
    eng = _Engine()

    class _SparqlBE:
        supports_sparql = True

    with (
        patch(
            "agent_utilities.knowledge_graph.integrations.stardog_sync.push_to_stardog",
            return_value={"status": "ok", "nodes": 3, "edges": 1},
        ) as push,
        patch(
            "agent_utilities.knowledge_graph.etl.lineage.record_etl_run",
            return_value="run-2",
        ),
        patch(
            "agent_utilities.knowledge_graph.enrichment.writeback.core.get_sink",
            return_value=None,
        ),
    ):
        out = run_etl(eng, sink="stardog", sink_backend=_SparqlBE(), sources=["leanix"])
    push.assert_called_once()
    assert out["outbound"]["nodes"] == 3
    assert out["lineage"]["direction"] == "outbound"


def test_run_etl_outbound_to_writeback_sink():
    eng = _Engine()
    sentinel = object()
    with (
        patch(
            "agent_utilities.knowledge_graph.enrichment.writeback.core.get_sink",
            return_value=sentinel,
        ),
        patch(
            "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
            return_value={"status": "completed", "created": 2},
        ) as wb,
        patch(
            "agent_utilities.knowledge_graph.etl.lineage.record_etl_run",
            return_value="run-3",
        ),
    ):
        out = run_etl(eng, sink="leanix", dry_run=True, ops={"creations": []})
    wb.assert_called_once()
    assert out["outbound"]["created"] == 2


def test_run_etl_through_records_through_lineage():
    eng = _Engine()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.source_sync.sync_source",
            return_value={"status": "ok", "nodes": 1},
        ),
        patch(
            "agent_utilities.knowledge_graph.enrichment.writeback.core.get_sink",
            return_value=object(),
        ),
        patch(
            "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
            return_value={"status": "completed"},
        ),
        patch(
            "agent_utilities.knowledge_graph.etl.lineage.record_etl_run",
            return_value="run-4",
        ) as rec,
    ):
        out = run_etl(eng, source="servicenow", sink="leanix")
    assert out["lineage"]["direction"] == "through"
    assert rec.call_args.kwargs["direction"] == "through"
