"""Self-registering source registry enables conflict-free parallel sources (KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import (
    EdgeRung,
    EnrichmentEdge,
    ExtractionBatch,
    GraphNode,
)
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    list_sources,
    register_extractor,
    write_batch,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def test_register_and_retrieve_source():
    def extract(cfg):
        return ExtractionBatch(category="demo")

    register_extractor("demo-itsm", extract, description="demo")
    src = get_source("demo-itsm")
    assert src is not None and src.description == "demo"
    assert any(s.category == "demo-itsm" for s in list_sources())


def test_write_batch_persists_nodes_and_edges():
    batch = ExtractionBatch(
        category="infra",
        nodes=[
            GraphNode(
                id="server:analysis-node-a",
                type="Server",
                props={"hostname": "analysis-node-a", "ip": "192.0.2.13"},
            ),
            GraphNode(
                id="service:pggraph",
                type="Service",
                props={"image": "pggraph", "replicas": 1},
            ),
        ],
        edges=[
            EnrichmentEdge(
                source="service:pggraph",
                target="server:analysis-node-a",
                rel_type="RUNS_ON",
            )
        ],
    )
    backend = FakeBackend()
    n, e = write_batch(backend, batch)
    assert n == 2 and e == 1
    assert backend.nodes["server:analysis-node-a"]["node_type"] == "Server"
    assert backend.nodes["server:analysis-node-a"]["hostname"] == "analysis-node-a"
    assert (
        "service:pggraph",
        "server:analysis-node-a",
        "RUNS_ON",
    ) in backend.edges


def test_write_batch_stamps_rung_and_confidence():
    """EH-274: write_batch promotes edge.rung/edge.confidence to first-class
    relationship properties, not left riding only in props."""
    batch = ExtractionBatch(
        category="infra",
        edges=[
            EnrichmentEdge(
                source="code:a",
                target="code:b",
                rel_type="CALLS",
                rung=EdgeRung.INFERRED,
                confidence=0.9,
            )
        ],
    )
    backend = FakeBackend()
    write_batch(backend, batch)
    assert backend.edge_props[0]["rung"] == "INFERRED"
    assert backend.edge_props[0]["confidence"] == 0.9


def test_write_batch_never_lets_a_higher_rung_overwrite_a_lower_one_in_batch():
    """EH-274 monotone safety, proven against a known-bad input: two edges for
    the SAME (source, target, rel_type) in one batch, the less-certain one
    listed SECOND. The write must keep the more-certain (lower-rung) fact,
    not last-writer-wins."""
    batch = ExtractionBatch(
        category="infra",
        edges=[
            EnrichmentEdge(
                source="code:a",
                target="code:b",
                rel_type="CALLS",
                rung=EdgeRung.EXTRACTED,
            ),
            EnrichmentEdge(
                source="code:a",
                target="code:b",
                rel_type="CALLS",
                rung=EdgeRung.ASSERTED,  # a less-certain, later write attempt
            ),
        ],
    )
    backend = FakeBackend()
    n, e = write_batch(backend, batch)
    assert e == 1  # collapsed to one edge, not two
    assert backend.edge_props[0]["rung"] == "EXTRACTED"


def test_discover_extractors_runs():
    # Auto-discovery must import the extractors package without error.
    from agent_utilities.knowledge_graph.enrichment.registry import discover_extractors

    assert discover_extractors() >= 1  # at least code_test / document modules
