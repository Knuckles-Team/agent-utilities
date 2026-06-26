"""`_engine_vector_search` ranks via the engine ANN — never an O(N) Python scan.

CONCEPT:KG-2.250 — the hand-orchestrated hybrid retriever's vector arm is
collapsed onto the engine. The vector neighbourhood comes from ONE cross-modal
unified plan (`graph.query_unified`, the engine sequencing filter + vector `Rank`
in one costed round-trip); on a lean engine built without the `query` feature it
falls to the engine's native `semantic_search` ANN primitive — still the engine's
vector index, still O(log N). There is NO O(N) Python `cosine_similarity` fallback
and NO `backend.execute` brute-force scan. (The real-engine end-to-end proof lives
in `test_unified_plan_retrieval.py`.)
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever


class _UnifiedGraph:
    """Engine graph that serves the unified plan (the `query`-feature path)."""

    def __init__(self) -> None:
        self.unified_calls = 0
        self.semantic_calls = 0
        self._props = {
            "n1": {"name": "Foo", "target_path": "/a/x.py"},
            "n2": {"name": "Bar", "target_path": "/b/y.py"},
        }

    def query_unified(
        self, _plan: list[dict[str, Any]], **_k: Any
    ) -> list[dict[str, Any]]:
        self.unified_calls += 1
        return [{"id": "n1", "score": 0.9}, {"id": "n2", "score": 0.7}]

    def semantic_search(self, _emb: list[float], _n: int = 5) -> list[Any]:
        self.semantic_calls += 1  # must NOT run when the unified plan works
        return []

    def _get_node_properties(self, nid: str) -> dict[str, Any]:
        return dict(self._props.get(nid, {}))


class _LeanGraph:
    """Lean engine (no `query` feature): unified plan errors, native ANN serves."""

    def __init__(self) -> None:
        self.semantic_calls = 0
        self._props = {"n1": {"name": "Foo"}, "n2": {"name": "Bar"}}

    def query_unified(self, _plan: list[dict[str, Any]], **_k: Any) -> Any:
        raise RuntimeError(
            "unknown variant `UnifiedQuery` (engine built without query)"
        )

    def semantic_search(self, _emb: list[float], _n: int = 5) -> list[Any]:
        self.semantic_calls += 1
        return [("n1", 0.9), ("n2", 0.7)]

    def _get_node_properties(self, nid: str) -> dict[str, Any]:
        return dict(self._props.get(nid, {}))


class _Engine:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.backend = graph


def _retriever(graph: Any) -> HybridRetriever:
    r = HybridRetriever.__new__(HybridRetriever)  # skip heavy __init__
    r.engine = _Engine(graph)  # type: ignore[assignment]
    return r


def test_labeled_search_uses_unified_plan_not_a_scan() -> None:
    """A label-scoped query composes Scan+Rank in ONE unified plan."""
    graph = _UnifiedGraph()
    r = _retriever(graph)

    out = r._engine_vector_search([0.1, 0.2, 0.3], top_k=5, threshold=0.0, label="Doc")

    assert [d["id"] for d in out] == ["n1", "n2"]  # engine-ranked order
    assert out[0]["_score"] == 0.9
    assert out[0]["name"] == "Foo"  # hydrated from the engine, in batch
    assert graph.unified_calls == 1
    # The unified plan served it — the native-ANN fallback was never needed.
    assert graph.semantic_calls == 0


def test_unseeded_search_uses_native_ann_not_a_scan() -> None:
    """Label-agnostic retrieval uses the engine's native ANN (unseeded kNN)."""
    graph = _UnifiedGraph()
    r = _retriever(graph)

    # No label → the unseeded kNN primitive, NOT an O(N) Python scan. (Our stub's
    # semantic_search returns [] so the arm is empty — the point is which engine
    # path is taken, asserted via the call counters.)
    r._engine_vector_search([0.1, 0.2, 0.3], top_k=5, threshold=0.0)

    assert graph.unified_calls == 0  # bare Rank has no seed — not used unseeded
    assert graph.semantic_calls == 1  # the engine ANN primitive was the vector path


def test_vector_search_respects_target_paths() -> None:
    graph = _UnifiedGraph()
    r = _retriever(graph)

    out = r._engine_vector_search(
        [0.1, 0.2, 0.3], top_k=5, threshold=0.0, target_paths=["/a/"], label="Doc"
    )

    assert [d["id"] for d in out] == ["n1"]  # only the /a/ path survives


def test_lean_engine_falls_to_native_ann_not_python_cosine() -> None:
    """No `query` feature ⇒ the engine's native ANN, NOT an O(N) Python scan."""
    graph = _LeanGraph()
    r = _retriever(graph)

    # Even with a label, a build without `query` errors on the unified plan and
    # degrades to the native ANN primitive — never a Python cosine scan.
    out = r._engine_vector_search([0.1, 0.2, 0.3], top_k=5, threshold=0.0, label="Doc")

    assert [d["id"] for d in out] == ["n1", "n2"]
    assert graph.semantic_calls == 1  # the engine ANN primitive served it


def test_old_on_python_scan_method_is_deleted() -> None:
    """The O(N) `_vector_search_native` brute-force entry point is gone."""
    assert not hasattr(HybridRetriever, "_vector_search_native")
