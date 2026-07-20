"""Self-registering source registry enables conflict-free parallel sources (KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import (
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
    assert backend.nodes["server:analysis-node-a"]["type"] == "Server"
    assert backend.nodes["server:analysis-node-a"]["hostname"] == "analysis-node-a"
    assert (
        "service:pggraph",
        "server:analysis-node-a",
        "RUNS_ON",
    ) in backend.edges


def test_discover_extractors_runs():
    # Auto-discovery must import the extractors package without error.
    from agent_utilities.knowledge_graph.enrichment.registry import discover_extractors

    assert discover_extractors() >= 1  # at least code_test / document modules
