"""Vector search/index goes through the engine HNSW, not a per-process dict.

CONCEPT:AU-KG.query.object-graph-mapper — `add_embedding` registers vectors in the engine's HNSW so
they survive restarts and `semantic_search` remains O(log N). There is no
per-process O(N) cosine authority. A one-time persisted-state migration indexes
pre-existing `embedding` node properties.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

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
        self.property_batch_calls = 0
        self.atomic_repaired: list[str] = []

    def add_embedding(self, nid: str, emb: list[float]) -> None:
        if self._add_raises:
            raise RuntimeError("engine down")
        self.added.append((nid, emb))

    def semantic_search(self, _q: list[float], n: int = 5) -> list[Any]:
        return self._hits[:n]

    def _get_node_properties(self, nid: str) -> dict[str, Any]:
        return dict(self._nodes.get(nid, {}))

    def _get_node_properties_batch(
        self, node_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        self.property_batch_calls += 1
        return {nid: dict(self._nodes.get(nid, {})) for nid in node_ids}

    def has_node(self, nid: str) -> bool:
        return nid in self._nodes

    def compare_and_set_node_embedding(
        self,
        nid: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        props = self._nodes.get(nid)
        if props is None or any(
            props.get(field) != expected for field, expected in conditions.items()
        ):
            return False
        props.update(updates)
        self.add_embedding(nid, embedding)
        props["_embedding_index_ready"] = True
        self.atomic_repaired.append(nid)
        return True

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._nodes.items())


def _backend(graph: _FakeGraph) -> EpistemicGraphBackend:
    b = EpistemicGraphBackend.__new__(
        EpistemicGraphBackend
    )  # skip engine-connecting __init__
    b._graph = graph
    return b


def test_add_embedding_writes_engine_index() -> None:
    g = _FakeGraph()
    b = _backend(g)
    b.add_embedding("n1", [0.1, 0.2])
    assert g.added == [("n1", [0.1, 0.2])]


def test_add_embedding_engine_failure_is_not_hidden() -> None:
    g = _FakeGraph(add_raises=True)
    b = _backend(g)

    with pytest.raises(RuntimeError, match="engine down"):
        b.add_embedding("n1", [0.1, 0.2])


def test_semantic_search_prefers_engine() -> None:
    g = _FakeGraph(
        hits=[("n1", 0.9), ("n2", 0.7)],
        nodes={
            "n1": {"embedding": [0.1, 0.2], "name": "A"},
            "n2": {"embedding": [0.3, 0.4], "name": "B"},
        },
    )
    b = _backend(g)  # local cache empty — proves results came from the engine
    out = b.semantic_search([0.1, 0.2], 5)
    assert [d["id"] for d in out] == ["n1", "n2"]
    assert out[0]["_similarity"] == 0.9
    assert out[0]["name"] == "A"
    assert g.property_batch_calls == 1


def test_semantic_search_excludes_invalidated_native_ann_candidate() -> None:
    g = _FakeGraph(
        hits=[("stale", 0.99), ("current", 0.8)],
        nodes={
            "stale": {"embedding": None, "name": "old text replaced"},
            "current": {"embedding": [0.1, 0.2], "name": "current"},
        },
    )
    b = _backend(g)

    assert b.semantic_search([0.1, 0.2], 5) == [
        {
            "embedding": [0.1, 0.2],
            "name": "current",
            "id": "current",
            "_similarity": 0.8,
        }
    ]


def test_backend_semantic_search_excludes_not_ready_vector_property() -> None:
    g = _FakeGraph(
        hits=[("projecting", 0.99), ("ready", 0.8)],
        nodes={
            "projecting": {
                "embedding": [0.9, 0.9],
                "_embedding_index_ready": False,
            },
            "ready": {
                "embedding": [0.1, 0.2],
                "_embedding_index_ready": True,
            },
        },
    )

    assert [row["id"] for row in _backend(g).semantic_search([0.1, 0.2], 5)] == [
        "ready"
    ]


def test_semantic_search_does_not_scan_a_local_cache_when_engine_is_empty() -> None:
    g = _FakeGraph(hits=[], nodes={"n1": {"name": "A"}})
    b = _backend(g)
    assert b.semantic_search([1.0, 0.0], 5) == []


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


def test_hydrate_repairs_committed_embedding_left_not_ready() -> None:
    g = _FakeGraph(
        nodes={
            "n1": {
                "embedding": [0.1, 0.2],
                "_embedding_index_ready": False,
            }
        }
    )
    b = _backend(g)

    assert b.hydrate_engine_embeddings() == 1
    assert g.atomic_repaired == ["n1"]
    assert g._nodes["n1"]["_embedding_index_ready"] is True


def test_graph_compute_wrappers_call_engine_client() -> None:
    class _NS:
        def __init__(self) -> None:
            self.added: tuple[str, list[float]] | None = None

        def add_embedding(self, nid: str, emb: list[float]) -> None:
            self.added = (nid, emb)

        def semantic_search(self, _q: list[float], n: int = 5) -> list[Any]:
            return [("stale", 0.9), ("n1", 0.5)][:n]

    class _Nodes:
        @staticmethod
        def properties_batch(_ids: list[str]) -> dict[str, dict[str, Any]]:
            return {
                "stale": {"embedding": None},
                "n1": {"embedding": [0.1]},
            }

    class _Client:
        def __init__(self) -> None:
            self.graph = _NS()
            self.nodes = _Nodes()

    g = GraphComputeEngine.__new__(GraphComputeEngine)
    g._client = _Client()
    g.add_embedding("n1", [0.1])
    assert g._client.graph.added == ("n1", [0.1])
    assert g.semantic_search([0.1], 3) == [("n1", 0.5)]


def test_atomic_embedding_cas_stages_guard_before_read_and_vector_commit() -> None:
    events: list[str] = []

    class _Txn:
        @staticmethod
        def begin() -> str:
            events.append("begin")
            return "txn-1"

        @staticmethod
        def cas(_txn, _node, _conditions, _updates) -> bool:
            events.append("cas")
            return True

        @staticmethod
        def add_embedding(_txn, _node, _embedding) -> bool:
            events.append("vector")
            return True

        @staticmethod
        def commit(_txn) -> bool:
            events.append("commit")
            return True

        @staticmethod
        def rollback(_txn) -> bool:
            events.append("rollback")
            return True

    class _Nodes:
        @staticmethod
        def properties(_node_id: str) -> dict[str, Any]:
            events.append("read")
            return {"name": "current", "embedding": None}

        @staticmethod
        def compare_and_set(_node_id, _conditions, _updates) -> bool:
            events.append("ready")
            return True

    class _Client:
        txn = _Txn()
        nodes = _Nodes()

    graph = GraphComputeEngine.__new__(GraphComputeEngine)
    graph._client = _Client()

    assert graph.compare_and_set_node_embedding(
        "n1",
        {"name": "current", "embedding": None},
        {"embedding": [0.1]},
        [0.1],
    )
    assert events == ["begin", "cas", "read", "vector", "commit", "ready"]


def test_atomic_embedding_cas_rolls_back_before_vector_when_snapshot_mismatches() -> (
    None
):
    events: list[str] = []

    class _Txn:
        @staticmethod
        def begin() -> str:
            events.append("begin")
            return "txn-1"

        @staticmethod
        def cas(*_args) -> bool:
            events.append("cas")
            return True

        @staticmethod
        def add_embedding(*_args) -> bool:
            events.append("vector")
            return True

        @staticmethod
        def commit(*_args) -> bool:
            events.append("commit")
            return True

        @staticmethod
        def rollback(*_args) -> bool:
            events.append("rollback")
            return True

    class _Nodes:
        @staticmethod
        def properties(_node_id: str) -> dict[str, Any]:
            events.append("read")
            return {"name": "changed", "embedding": None}

    class _Client:
        txn = _Txn()
        nodes = _Nodes()

    graph = GraphComputeEngine.__new__(GraphComputeEngine)
    graph._client = _Client()

    assert not graph.compare_and_set_node_embedding(
        "n1",
        {"name": "expected", "embedding": None},
        {"embedding": [0.1]},
        [0.1],
    )
    assert events == ["begin", "cas", "read", "rollback"]


def test_semantic_search_rejects_property_until_txn_ann_projection_is_ready() -> None:
    """Durable txn publishes graph before ANN; readiness keeps that window dark."""
    property_published = threading.Event()
    release_ann_projection = threading.Event()
    result: list[bool] = []

    class _Nodes:
        def __init__(self) -> None:
            self.props: dict[str, Any] = {
                "name": "new text",
                "embedding": None,
            }

        def properties(self, _node_id: str) -> dict[str, Any]:
            return dict(self.props)

        def properties_batch(self, _node_ids: list[str]) -> dict[str, Any]:
            return {"n1": dict(self.props)}

        def compare_and_set(self, _node_id, conditions, updates) -> bool:
            if any(self.props.get(key) != value for key, value in conditions.items()):
                return False
            self.props.update(updates)
            return True

    class _Graph:
        def __init__(self) -> None:
            self.score = 0.2  # old ANN vector against the new query/text

        def semantic_search(self, _query, _limit):
            return [("n1", self.score)]

    class _Query:
        def unified(self, _plan):
            return [{"id": "n1", "score": graph_ns.score}]

    nodes = _Nodes()
    graph_ns = _Graph()

    class _Txn:
        def __init__(self) -> None:
            self.updates: dict[str, Any] = {}

        @staticmethod
        def begin() -> str:
            return "txn-visibility"

        def cas(self, _txn, _node, _conditions, updates) -> bool:
            self.updates = dict(updates)
            return True

        @staticmethod
        def add_embedding(_txn, _node, _embedding) -> bool:
            return True

        def commit(self, _txn) -> bool:
            # Mirrors the engine's current served-publication order: graph
            # properties first, SemanticStore replacement second.
            nodes.props.update(self.updates)
            property_published.set()
            assert release_ann_projection.wait(timeout=5.0)
            graph_ns.score = 0.9
            return True

        @staticmethod
        def rollback(_txn) -> bool:
            return True

    class _Client:
        def __init__(self) -> None:
            self.nodes = nodes
            self.graph = graph_ns
            self.query = _Query()
            self.txn = _Txn()

    compute = GraphComputeEngine.__new__(GraphComputeEngine)
    compute._client = _Client()

    worker = threading.Thread(
        target=lambda: result.append(
            compute.compare_and_set_node_embedding(
                "n1",
                {"name": "new text", "embedding": None},
                {"embedding": [0.9]},
                [0.9],
            )
        )
    )
    worker.start()
    try:
        assert property_published.wait(timeout=5.0)
        assert nodes.props["embedding"] == [0.9]
        assert nodes.props["_embedding_index_ready"] is False
        assert graph_ns.score == 0.2
        assert compute.semantic_search([0.9], 1) == []
        assert compute.query_unified(
            [
                {"Scan": {"label": "Fixture"}},
                {"Rank": {"query": [0.9]}},
                {"Limit": {"k": 1}},
            ]
        ) == []
        assert worker.is_alive()

        release_ann_projection.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert result == [True]
        assert nodes.props["_embedding_index_ready"] is True
        assert compute.semantic_search([0.9], 1) == [("n1", 0.9)]
        ready_rows = compute.query_unified(
            [
                {"Scan": {"label": "Fixture"}},
                {"Rank": {"query": [0.9]}},
                {"Limit": {"k": 1}},
            ]
        )
        assert [(row["id"], row["score"]) for row in ready_rows] == [("n1", 0.9)]
    finally:
        release_ann_projection.set()
        worker.join(timeout=5.0)
