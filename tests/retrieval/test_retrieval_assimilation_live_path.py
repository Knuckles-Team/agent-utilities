"""Live-path (Wire-First) tests for the retrieval-assimilation cluster.

Exercises the new capabilities through the REAL ``IntelligenceGraphEngine`` /
``search_*`` entry points (not just the helper modules in isolation):

- CONCEPT:AU-KG.retrieval.unset-dependency-free ScoreGate — ``search_hybrid`` annotates ``_fused_score`` and
  adaptively trims via the dual-score gate.
- CONCEPT:AU-KG.query.chronoid-fits-residual-quantization ChronoID — ``search_hybrid`` annotates ``_time_bucket``; the
  ``temporal_semantic_ids`` entry point attaches ``_temporal_sid``.
- CONCEPT:AU-KG.query.adore-concept-expansion ADORE + CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval TASR — ``search_adore`` runs the
  iterative reformulate→retrieve→judge loop to a stopping decision.
"""

import time

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.get_active_backend",
        lambda: None,
    )
    g = GraphComputeEngine(backend_type="rust")
    for node in g.node_ids():
        g.remove_node(node)
    eng = IntelligenceGraphEngine(db_path=":memory:")
    now = time.time()
    eng.graph.add_node(
        "py1",
        name="Python Expert",
        description="Helps with python programming and packaging",
        embedding=[1.0, 0.1, 0.0, 0.0],
        event_time=now,
    )
    eng.graph.add_node(
        "py2",
        name="Python Tooling",
        description="python tooling, pytest and python virtualenv help",
        embedding=[0.9, 0.2, 0.0, 0.0],
        event_time=now - 200 * 86400,
    )
    eng.graph.add_node(
        "rs1",
        name="Rust Expert",
        description="Helps with rust borrow checker",
        embedding=[0.0, 0.0, 1.0, 0.1],
        event_time=now,
    )
    return eng


def test_score_gate_and_time_bucket_default_on(engine):
    """KG-2.85 + KG-2.86 are wired into the default search_hybrid flow."""
    results = engine.search_hybrid("python", top_k=10)
    assert results, "expected at least one python result"
    for r in results:
        # ScoreGate fused both encoder signals into every retained result.
        assert "_fused_score" in r
        # ChronoID recency token is attached by default.
        assert "_time_bucket" in r
        assert isinstance(r["_time_bucket"], int)
    ids = {r["id"] for r in results}
    assert "py1" in ids
    assert "rs1" not in ids  # weak/irrelevant tail trimmed


def test_temporal_semantic_ids_entry_point(engine):
    """KG-2.86 — the chrono_ids entry point attaches a temporal semantic ID."""
    results = engine.temporal_semantic_ids("python", top_k=10)
    assert results
    recent = next((r for r in results if r["id"] == "py1"), None)
    older = next((r for r in results if r["id"] == "py2"), None)
    assert recent is not None
    # Every result carries the explicit recency bucket.
    for r in results:
        assert "_time_bucket" in r
    # Embeddings present -> a residual-quantized semantic ID is produced.
    if recent.get("embedding"):
        assert "_temporal_sid" in recent
        assert isinstance(recent["_temporal_sid"], list)
    # The more-recent node lands in an earlier (smaller) bucket than the older one.
    if older is not None:
        assert recent["_time_bucket"] <= older["_time_bucket"]


def test_search_adore_iterative_loop(engine):
    """KG-2.88 + KG-2.87 — the ADORE entry point runs end-to-end and ranks."""
    results = engine.search_adore("python", top_k=5)
    assert isinstance(results, list)
    ids = {r["id"] for r in results}
    # The relevant python nodes surface through the graded-feedback loop.
    assert ids & {"py1", "py2"}
