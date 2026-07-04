"""Vector search/index goes through the engine HNSW, not a per-process dict.

CONCEPT:AU-KG.query.object-graph-mapper — `add_embedding` must register vectors in the engine's HNSW (so
they survive restarts and `semantic_search` is O(log N)); `semantic_search` must
prefer the engine and only fall back to the local cosine cache. A one-time
`hydrate_engine_embeddings` indexes legacy `embedding` node properties.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


class _FakeGraph:
    def __init__(
        self,
        hits: list[Any] | None = None,
        nodes: dict[str, dict[str, Any]] | None = None,
        add_raises: bool = False,
    ) -> None:
        self.added: list[tuple[str, list[float]]] = []
        self._hits = hits or []
        self._nodes = nodes or {}
        self._add_raises = add_raises

    def add_embedding(self, nid: str, emb: list[float]) -> None:
        if self._add_raises:
            raise RuntimeError("engine down")
        self.added.append((nid, emb))

    def semantic_search(self, _q: list[float], n: int = 5) -> list[Any]:
        return self._hits[:n]

    def _get_node_properties(self, nid: str) -> dict[str, Any]:
        return dict(self._nodes.get(nid, {}))

    def has_node(self, nid: str) -> bool:
        return nid in self._nodes

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._nodes.items())


def _backend(graph: _FakeGraph) -> EpistemicGraphBackend:
    b = EpistemicGraphBackend.__new__(
        EpistemicGraphBackend
    )  # skip engine-connecting __init__
    b._graph = graph
    b._embeddings = {}
    return b


def test_add_embedding_writes_local_and_engine() -> None:
    g = _FakeGraph()
    b = _backend(g)
    b.add_embedding("n1", [0.1, 0.2])
    assert b._embeddings["n1"] == [0.1, 0.2]  # write-through cache
    assert g.added == [("n1", [0.1, 0.2])]  # indexed in the engine HNSW


def test_add_embedding_engine_failure_keeps_cache() -> None:
    g = _FakeGraph(add_raises=True)
    b = _backend(g)
    b.add_embedding("n1", [0.1, 0.2])  # must not raise
    assert b._embeddings["n1"] == [0.1, 0.2]


def test_semantic_search_prefers_engine() -> None:
    g = _FakeGraph(
        hits=[("n1", 0.9), ("n2", 0.7)],
        nodes={"n1": {"name": "A"}, "n2": {"name": "B"}},
    )
    b = _backend(g)  # local cache empty — proves results came from the engine
    out = b.semantic_search([0.1, 0.2], 5)
    assert [d["id"] for d in out] == ["n1", "n2"]
    assert out[0]["_similarity"] == 0.9
    assert out[0]["name"] == "A"


def test_semantic_search_falls_back_to_local_when_engine_empty() -> None:
    g = _FakeGraph(hits=[], nodes={"n1": {"name": "A"}})
    b = _backend(g)
    b._embeddings = {"n1": [1.0, 0.0]}
    out = b.semantic_search([1.0, 0.0], 5)  # engine empty -> local cosine
    assert [d["id"] for d in out] == ["n1"]
    assert out[0]["_similarity"] > 0.99


def test_hydrate_indexes_node_embedding_properties() -> None:
    g = _FakeGraph(
        nodes={
            "n1": {"embedding": [0.1, 0.2], "name": "A"},
            "n2": {"name": "B"},  # no embedding -> skipped
            "n3": {"embedding": [0.3, 0.4]},
        }
    )
    b = _backend(g)
    indexed = b.hydrate_engine_embeddings()
    assert indexed == 2
    assert sorted(nid for nid, _ in g.added) == ["n1", "n3"]


def test_graph_compute_wrappers_call_engine_client() -> None:
    class _NS:
        def __init__(self) -> None:
            self.added: tuple[str, list[float]] | None = None

        def add_embedding(self, nid: str, emb: list[float]) -> None:
            self.added = (nid, emb)

        def semantic_search(self, _q: list[float], n: int = 5) -> list[Any]:
            return [("n1", 0.5)]

    class _Client:
        def __init__(self) -> None:
            self.graph = _NS()

    g = GraphComputeEngine.__new__(GraphComputeEngine)
    g._client = _Client()
    g.add_embedding("n1", [0.1])
    assert g._client.graph.added == ("n1", [0.1])
    assert g.semantic_search([0.1], 3) == [("n1", 0.5)]
