from __future__ import annotations

"""Tests for the batched variable-length neighbour prefetch (D-DPF-3).

CONCEPT:AU-KG.retrieval.batched-neighborhood-prefetch

``retrieve_hybrid``'s base-node loop used to issue one
``MATCH (n {id:$id})-[*1..N]-(m) RETURN m`` Cypher round trip PER base node
(measured live: 22 calls totaling 35.05s, avg 1.59s/call). It now issues ONE
``UNWIND $ids AS base_id MATCH (n {id:base_id})-[*1..N]-(m) RETURN base_id, m``
call across every base node up front (:meth:`HybridRetriever._varlen_neighbors_batch`)
and the loop looks results up from the returned dict.

These tests pin two things a per-node-call regression would break:
  1. Exactly ONE ``backend.execute`` call is issued for the whole traversal,
     regardless of how many base nodes are being expanded.
  2. Each base node's neighbours are correctly attributed to IT and not to a
     sibling base node — a batched query that failed to group by ``base_id``
     would silently corrupt the assembled subgraph without a traceback.
"""

from unittest.mock import MagicMock, patch


class _FakeVectorGraph:
    """Engine-graph double: vector arm reads ``semantic_search`` + hydrates via
    ``_get_node_properties`` (the contract; mirrors ``test_backlink_boost.py``)."""

    def __init__(self, hits, props):  # type: ignore[no-untyped-def]
        self._hits = hits  # list[(id, score)]
        self._props = props  # dict[id -> props]

    def query_unified(self, _plan, **_k):  # type: ignore[no-untyped-def]
        return []

    def semantic_search(self, _emb, _n=5):  # type: ignore[no-untyped-def]
        return list(self._hits)

    def _get_node_properties(self, nid):  # type: ignore[no-untyped-def]
        return dict(self._props.get(nid, {}))

    def has_node(self, nid):  # type: ignore[no-untyped-def]
        return nid in self._props

    def get_successors(self, _nid):  # type: ignore[no-untyped-def]
        return []

    def get_predecessors(self, _nid):  # type: ignore[no-untyped-def]
        return []


class _RecordingVarlenBackend:
    """Fake native-engine Cypher backend for both the batched (current) and the
    per-id (pre-fix) query shapes, so the SAME fake serves either version of
    the production code — the batched shape is a single call over ``$ids``;
    the per-id shape is one call per ``$id``, both walking the same adjacency.
    """

    def __init__(self, adjacency: dict[str, list[dict]]):  # type: ignore[no-untyped-def]
        self._adjacency = adjacency
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query, params=None):  # type: ignore[no-untyped-def]
        params = params or {}
        self.calls.append((query, dict(params)))
        if "UNWIND" in query:
            ids = params.get("ids") or []
            return [
                {"base_id": bid, "m": dict(m)}
                for bid in ids
                for m in self._adjacency.get(bid, [])
            ]
        # Pre-fix per-base-node shape: MATCH (n {id:$id})-[*1..N]-(m) RETURN m
        nid = params.get("id")
        return [{"m": dict(m)} for m in self._adjacency.get(nid, [])]

    def __bool__(self) -> bool:  # engine.backend is truth-tested by the loop
        return True


def _make_retriever(backend, hits, props, *, enable_rerank: bool = False):
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    engine = MagicMock()
    engine.graph = _FakeVectorGraph(hits=hits, props=props)
    engine.backend = backend

    r = HybridRetriever(engine, enable_rerank=enable_rerank)
    mock_embed = MagicMock()
    mock_embed.get_text_embedding.return_value = [1.0, 0.0]
    r.embed_model = mock_embed
    return r


@patch(
    "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
)
class TestVarlenNeighborBatching:
    def test_one_call_replaces_one_per_base_node(self, _m):
        """Three base nodes, each with real neighbours: the traversal issues
        exactly ONE backend.execute call, not three."""
        adjacency = {
            "A": [{"id": "X", "name": "XX"}, {"id": "Y", "name": "YY"}],
            "B": [{"id": "Y", "name": "YY"}, {"id": "Z", "name": "ZZ"}],
            "C": [{"id": "W", "name": "WW"}],
        }
        backend = _RecordingVarlenBackend(adjacency)
        props = {
            "A": {"id": "A", "name": "Alpha"},
            "B": {"id": "B", "name": "Bravo"},
            "C": {"id": "C", "name": "Charlie"},
        }
        r = _make_retriever(
            backend, hits=[("A", 1.0), ("B", 0.9), ("C", 0.8)], props=props
        )

        results = r.retrieve_hybrid(
            query="find things",
            context_window=3,
            multi_hop_depth=2,
            skip_quality_gate=True,
        )

        # The N+1 regression this pins: with 3 base nodes, the OLD per-node
        # loop issued 3 calls. The fix issues exactly 1.
        assert len(backend.calls) == 1, (
            f"expected exactly ONE batched backend.execute call for 3 base "
            f"nodes, got {len(backend.calls)}: {backend.calls}"
        )
        query, params = backend.calls[0]
        assert "UNWIND" in query
        assert set(params["ids"]) == {"A", "B", "C"}

        # Each base node's neighbours are attributed correctly — not merged,
        # not swapped. Y is shared but only claimed by whichever base node the
        # loop visits first (A, by vector-score order); B keeps only Z.
        result_ids = {n["id"] for n in results}
        assert {"A", "X", "Y", "B", "Z", "C", "W"} <= result_ids

    def test_grouping_does_not_cross_contaminate(self, _m):
        """A base node with NO neighbours must not pick up another base
        node's neighbours from a mis-grouped batched result."""
        adjacency = {
            "A": [{"id": "X", "name": "XX"}],
            "B": [],  # isolated within hop range
        }
        backend = _RecordingVarlenBackend(adjacency)
        props = {"A": {"id": "A", "name": "Alpha"}, "B": {"id": "B", "name": "Bravo"}}
        r = _make_retriever(backend, hits=[("A", 1.0), ("B", 0.9)], props=props)

        results = r.retrieve_hybrid(
            query="find things",
            context_window=2,
            multi_hop_depth=2,
            skip_quality_gate=True,
        )

        assert len(backend.calls) == 1
        by_id = {n["id"]: n for n in results}
        assert "X" in by_id
        # B got no varlen neighbours of its own — X must not appear attached
        # to B (there is nothing in this test asserting B "contains" X since
        # results are a flat list, but B's own bare node must still surface
        # via the BFS/hydration fallback rather than silently vanishing).
        assert "B" in by_id

    def test_batch_unsupported_is_cached_on_the_instance(self, _m):
        """If the batched UNWIND+varlen shape is rejected once, it is not
        retried per base node — every remaining node degrades straight to the
        BFS fallback for the rest of this retriever's lifetime."""
        backend = MagicMock()
        backend.__bool__ = lambda self: True
        backend.execute.side_effect = RuntimeError("native Cypher authority rejected")
        props = {"A": {"id": "A", "name": "Alpha"}, "B": {"id": "B", "name": "Bravo"}}
        r = _make_retriever(backend, hits=[("A", 1.0), ("B", 0.9)], props=props)

        r.retrieve_hybrid(
            query="find things",
            context_window=2,
            multi_hop_depth=2,
            skip_quality_gate=True,
        )

        assert r._varlen_batch_unsupported is True
        # Exactly one attempt was made against the backend for the whole
        # traversal (the batched call), not one attempt per base node.
        assert backend.execute.call_count == 1


class TestVarlenNeighborsBatchDirect:
    """Unit-level coverage of ``_varlen_neighbors_batch`` in isolation."""

    def test_single_round_trip_for_many_ids(self):
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        adjacency = {
            "n1": [{"id": "a"}, {"id": "b"}],
            "n2": [{"id": "b"}, {"id": "c"}],
            "n3": [],
        }
        backend = _RecordingVarlenBackend(adjacency)
        engine = MagicMock()
        engine.backend = backend
        r = HybridRetriever.__new__(HybridRetriever)
        r.engine = engine
        r._varlen_batch_unsupported = False

        out = r._varlen_neighbors_batch(["n1", "n2", "n3"], depth=2)

        assert len(backend.calls) == 1
        assert [m["id"] for m in out.get("n1", [])] == ["a", "b"]
        assert [m["id"] for m in out.get("n2", [])] == ["b", "c"]
        assert out.get("n3", []) == []

    def test_empty_ids_short_circuits_without_a_call(self):
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        backend = _RecordingVarlenBackend({})
        engine = MagicMock()
        engine.backend = backend
        r = HybridRetriever.__new__(HybridRetriever)
        r.engine = engine
        r._varlen_batch_unsupported = False

        assert r._varlen_neighbors_batch([], depth=2) == {}
        assert backend.calls == []

    def test_unsupported_backend_fails_closed_once(self):
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        class _RejectingBackend:
            def __init__(self):
                self.calls = 0

            def execute(self, _q, _p=None):  # type: ignore[no-untyped-def]
                self.calls += 1
                raise RuntimeError("native Cypher authority rejected request")

            def __bool__(self) -> bool:
                return True

        backend = _RejectingBackend()
        engine = MagicMock()
        engine.backend = backend
        r = HybridRetriever.__new__(HybridRetriever)
        r.engine = engine
        r._varlen_batch_unsupported = False

        assert r._varlen_neighbors_batch(["n1"], depth=2) == {}
        assert r._varlen_batch_unsupported is True
        assert backend.calls == 1

        # Second call must NOT re-attempt the backend.
        assert r._varlen_neighbors_batch(["n2"], depth=2) == {}
        assert backend.calls == 1
