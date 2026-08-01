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

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


class _FakeEmbed:
    """Deterministic embedding model matching this fixture's seeded node
    vectors (the same no-network pattern
    tests/unit/knowledge_graph/test_unified_plan_retrieval.py uses): "python"
    text embeds near py1/py2's stored vectors, "rust" near rs1's, so ANN
    ranking is fully determined without a real embedding provider.
    """

    def get_text_embedding(self, text: str) -> list[float]:
        lowered = text.lower()
        if "python" in lowered:
            return [1.0, 0.1, 0.0, 0.0]
        if "rust" in lowered:
            return [0.0, 0.0, 1.0, 0.1]
        return [0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def engine(engine_graph):
    # A bare GraphComputeEngine(backend_type="rust")/IntelligenceGraphEngine(
    # db_path=":memory:") pair each resolve their OWN routing graph
    # independently -- the former via the autouse isolate_graph_compute_engine
    # redirect, the latter via create_backend's bare EpistemicGraphBackend(),
    # which bypasses that same redirect (resolve_routing_graph(None) resolves
    # to the ambient tenant graph, not None/__commons__/__secrets__) -- so
    # `eng.graph` ends up bound to a DIFFERENT graph than the session the
    # test's actor is scoped to, and every write raises PermissionError: "A
    # graph-scoped view cannot retarget the verified GraphSession". Bind both
    # explicitly to the one isolated per-test engine_graph tenant instead.
    eng = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name)
    )
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
    # add_node(..., embedding=[...]) only stores it as a regular property --
    # it does NOT index it into the engine's ANN structure. search_hybrid's
    # vector arm (_engine_vector_search) queries that ANN index, not node
    # properties directly, so without this the vector arm always finds zero
    # candidates regardless of a working embed_model (same seeding pattern
    # test_unified_plan_retrieval.py uses: graph.add_embedding(...)).
    eng.graph.add_embedding("py1", [1.0, 0.1, 0.0, 0.0])
    eng.graph.add_embedding("py2", [0.9, 0.2, 0.0, 0.0])
    eng.graph.add_embedding("rs1", [0.0, 0.0, 1.0, 0.1])
    # No embedding provider is configured in this test environment
    # ("No embedding model is configured"), so search_hybrid's vector arm
    # never ran and fell to the scoreless keyword path, which the retrieval
    # quality gate always rejects (composite=0.0) -- these tests assert real
    # ScoreGate/ChronoID/ADORE annotations, which need the vector arm to
    # actually run. Inject the deterministic fake embedder directly (the
    # lazy embed_model property has a setter for exactly this).
    eng.hybrid_retriever.embed_model = _FakeEmbed()
    yield eng
    IntelligenceGraphEngine._ACTIVE_ENGINE = None


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
