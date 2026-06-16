"""`_vector_search_native` prefers the backend's HNSW, not an O(N) full scan.

CONCEPT:KG-2.0 — the native vector fast path was Ladybug-only
(`CALL QUERY_VECTOR_INDEX`), so on the epistemic_graph/fanout backend it always
returned None and `retrieve_hybrid` degraded to an O(N) full-graph scan
(`_get_all_nodes_with_properties`, 8× per query). It must now use the backend's
own `semantic_search` (HNSW) and never touch `backend.execute` when that works.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever


class _SemBackend:
    """Backend exposing the engine's native vector search (HNSW)."""

    def __init__(self) -> None:
        self.semantic_calls = 0
        self.execute_calls = 0

    def semantic_search(self, _emb: list[float], _n: int = 5) -> list[dict[str, Any]]:
        self.semantic_calls += 1
        return [
            {"id": "n1", "_similarity": 0.9, "name": "Foo", "target_path": "/a/x.py"},
            {"id": "n2", "_similarity": 0.7, "name": "Bar", "target_path": "/b/y.py"},
        ]

    def execute(
        self, _q: str, _p: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.execute_calls += 1  # QUERY_VECTOR_INDEX / brute-force path — must NOT run
        return []


class _NoSemBackend:
    """Backend without a usable vector search (semantic_search returns nothing)."""

    def __init__(self) -> None:
        self.execute_calls = 0

    def semantic_search(self, _emb: list[float], _n: int = 5) -> list[dict[str, Any]]:
        return []

    def execute(
        self, _q: str, _p: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.execute_calls += 1
        return []  # no Ladybug index -> native path yields nothing


class _Engine:
    def __init__(self, backend: Any) -> None:
        self.backend = backend


def _retriever(backend: Any) -> HybridRetriever:
    r = HybridRetriever.__new__(HybridRetriever)  # skip heavy __init__
    r.engine = _Engine(backend)
    return r


def test_native_uses_backend_semantic_search_not_full_scan() -> None:
    backend = _SemBackend()
    r = _retriever(backend)

    out = r._vector_search_native([0.1, 0.2, 0.3], top_k=5)

    assert out is not None
    assert [d["id"] for d in out] == ["n1", "n2"]  # ranked by score
    assert out[0]["_score"] == 0.9
    assert backend.semantic_calls == 1
    # The whole point: the Ladybug/brute-force execute() path is never taken.
    assert backend.execute_calls == 0


def test_native_respects_target_paths() -> None:
    backend = _SemBackend()
    r = _retriever(backend)

    out = r._vector_search_native([0.1, 0.2, 0.3], top_k=5, target_paths=["/a/"])

    assert out is not None
    assert [d["id"] for d in out] == ["n1"]  # only the /a/ path survives


def test_native_falls_through_when_semantic_search_empty() -> None:
    backend = _NoSemBackend()
    r = _retriever(backend)

    # semantic_search yields nothing -> falls through to the Ladybug procedure,
    # which (no index here) yields nothing -> None so the caller brute-forces.
    out = r._vector_search_native([0.1, 0.2, 0.3], top_k=5)

    assert out is None
    assert backend.execute_calls >= 1  # the fallback path was attempted
