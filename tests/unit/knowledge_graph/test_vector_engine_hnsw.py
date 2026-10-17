"""Vector search/index goes through the engine HNSW, not a per-process dict.

CONCEPT:AU-KG.query.object-graph-mapper — `add_embedding` registers vectors in the engine's HNSW so
they survive restarts and `semantic_search` remains O(log N). There is no
per-process O(N) cosine authority. A one-time persisted-state migration indexes
pre-existing `embedding` node properties.
"""

from __future__ import annotations

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

    def compare_and_set_node_fields(
        self,
        nid: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        props = self._nodes.get(nid)
        if props is None or any(
            props.get(field) != expected for field, expected in conditions.items()
        ):
            return False
        props.update(updates)
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


def test_hydrate_marks_legacy_node_ready_so_a_converged_second_tick_writes_nothing() -> (
    None
):
    """D-BFR-5: hydration must be delta-driven, not a blind full-index replay.

    A legacy node with an ``embedding`` property but no readiness marker is
    genuinely unknown to the ANN, so the FIRST tick must index it — but it
    must then durably mark it ready so a SECOND tick (the steady state once
    the backfill/ingest chokepoint has populated the index) performs ZERO
    vector writes instead of replaying the whole index every 10 minutes.
    """
    g = _FakeGraph(
        nodes={
            "n1": {"embedding": [0.1, 0.2], "name": "A"},
            "n2": {"name": "B"},  # no embedding -> never scanned
            "n3": {"embedding": [0.3, 0.4]},
        }
    )
    b = _backend(g)

    first = b.hydrate_engine_embeddings()
    assert first == 2
    assert sorted(nid for nid, _ in g.added) == ["n1", "n3"]
    assert g._nodes["n1"]["_embedding_index_ready"] is True
    assert g._nodes["n3"]["_embedding_index_ready"] is True

    g.added.clear()
    second = b.hydrate_engine_embeddings()
    assert second == 0
    assert g.added == []


def test_hydrate_skips_already_ready_nodes_without_touching_the_ann() -> None:
    """A node whose readiness fence is already ``True`` must never be
    re-added — that IS the D-BFR-5 redundant-write bug this closes."""
    g = _FakeGraph(
        nodes={
            "n1": {"embedding": [0.1, 0.2], "_embedding_index_ready": True},
        }
    )
    b = _backend(g)

    assert b.hydrate_engine_embeddings() == 0
    assert g.added == []


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
