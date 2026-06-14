"""Unit tests for the cross-backend graph migration (CONCEPT:KG-2.74).

`copy_graph` reads a source graph and writes it to a target backend via the
engine's portable MERGE upserts. These tests use fakes (no real backend) to
assert:

* nodes + edges + embeddings are copied;
* the WRITE cypher is clean portable MERGE — NOT the old reconstructed
  ``CREATE (n:Label {`k`: $k})`` that double-escaped reserved keys (the bug that
  broke native-cypher mirrors);
* a durable-cypher source (MATCH-based read) round-trips.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.migration import copy_graph


class RecBackend(GraphBackend):
    """Records every write; answers the label-lookup + count reads."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any] | None]] = []
        self.embeddings: list[str] = []

    def execute(self, query, params=None):
        if "RETURN" in query and "lbl" in query:  # edge label lookup
            return [{"lbl": "Doc"}]
        if "count(n)" in query:  # drift count
            return [{"c": 0}]
        self.writes.append((query, params))
        return []

    def execute_batch(self, query, batch):
        return []

    def create_schema(self):
        pass

    def add_embedding(self, node_id, embedding):
        self.embeddings.append(node_id)

    def semantic_search(self, query_embedding, n_results=5):
        return []

    def prune(self, criteria):
        pass

    def close(self):
        pass


class _L1GraphStub:
    """A fake L1 compute graph: nodes with reserved-word + nested props, one edge."""

    def _get_all_nodes(self):
        return ["a", "b"]

    def _get_node_properties(self, nid):
        return {
            "id": nid,
            "type": "Doc",  # reserved-ish key that broke the old reconcile cypher
            "metadata": {"k": "v"},  # nested → JSON-encoded for map-unsafe drivers
            "name": nid,
            "embedding": [0.1, 0.2, 0.3],
        }

    def _get_all_edges(self):
        return [("a", "b", {"type": "LINKS", "confidence": 0.9})]


class _L1Source:
    graph = _L1GraphStub()


def test_copy_graph_copies_nodes_edges_embeddings():
    tgt = RecBackend()
    summary = copy_graph(_L1Source(), tgt)
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    assert summary["embeddings"] == 2
    assert summary["errors"] == 0


class _CorruptKeyGraphStub:
    """L1 node whose ``type`` lives under a BACKTICKED key (legacy corruption)."""

    def _get_all_nodes(self):
        return ["x"]

    def _get_node_properties(self, nid):
        return {
            "id": nid,
            "`type`": "Memory",  # real type hidden behind backticks
            "`metadata`": {"k": "v"},
            "node_type": "raw",
            "embedding": [0.1],
        }

    def _get_all_edges(self):
        return []


class _CorruptKeySource:
    graph = _CorruptKeyGraphStub()


def test_copy_graph_recovers_backticked_keys_no_loss():
    """A node with backticked property keys must NOT be lost: keys are sanitised,
    the real label is recovered from the cleaned ``type``, and the emitted cypher
    has no double backticks."""
    tgt = RecBackend()
    summary = copy_graph(_CorruptKeySource(), tgt, copy_embeddings=False)
    assert summary["nodes"] == 1 and summary["errors"] == 0
    node_writes = [q for q, _ in tgt.writes if q.startswith("MERGE (n:")]
    assert node_writes, "node not written"
    # real label recovered from the cleaned `type`, not the default "Node"
    assert any("MERGE (n:Memory " in q for q in node_writes), node_writes
    for q, _ in tgt.writes:
        assert "``" not in q, f"double backtick leaked: {q}"


def test_copy_graph_emits_clean_merge_not_double_backticks():
    """The regression: writes must be portable MERGE with single-backtick keys,
    never the double-backticked `` ``type`` `` the old reconcile produced."""
    tgt = RecBackend()
    copy_graph(_L1Source(), tgt, copy_embeddings=False)
    node_writes = [q for q, _ in tgt.writes if q.startswith("MERGE (n:")]
    assert node_writes, "no node MERGE writes recorded"
    for q in tgt.writes:
        # no double-backtick anywhere, no $`param` form, no bare CREATE-with-map
        assert "``" not in q[0], f"double backtick leaked: {q[0]}"
        assert "$`" not in q[0], f"backtick-quoted param leaked: {q[0]}"
    edge_writes = [q for q, _ in tgt.writes if "MERGE (s)-[r:LINKS]->(t)" in q]
    assert edge_writes, "edge MERGE missing"


def test_copy_graph_reads_durable_cypher_source():
    """A source with no compute graph is read via MATCH (full-cypher path)."""

    class DurableSource(GraphBackend):
        @property
        def cypher_support(self):
            return "full"

        def execute(self, query, params=None):
            if query.startswith("MATCH (n) RETURN"):
                return [
                    {"id": "x", "label": "Doc", "node": {"id": "x", "name": "X"}},
                ]
            if query.startswith("MATCH (s)-[r]->(t)"):
                return [{"sid": "x", "tid": "x", "rel": "SELF", "edge": {}}]
            return []

        def execute_batch(self, q, b):
            return []

        def create_schema(self):
            pass

        def add_embedding(self, n, e):
            pass

        def semantic_search(self, q, n=5):
            return []

        def prune(self, c):
            pass

        def close(self):
            pass

    tgt = RecBackend()
    summary = copy_graph(DurableSource(), tgt, copy_embeddings=False)
    assert summary["nodes"] == 1
    assert summary["edges"] == 1
