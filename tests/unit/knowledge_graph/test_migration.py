"""Unit tests for the cross-backend graph migration (CONCEPT:AU-KG.backend.mirror-health-repair).

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


class _FalkorLike(GraphBackend):
    """Fake whose CLASS NAME enables the id-index path; records index DDL and lets
    the test inject a per-statement error to model FalkorDB's behaviour."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.index_ddl: list[str] = []

    def execute(self, query, params=None):
        if query.startswith("CREATE INDEX"):
            self.index_ddl.append(query)
            if self.error is not None:
                raise self.error
        return []

    def execute_batch(self, query, batch):
        return []

    def create_schema(self):
        pass

    def add_embedding(self, node_id, embedding):
        pass

    def semantic_search(self, q, n_results=5):
        return []

    def prune(self, criteria):
        pass

    def close(self):
        pass


# FalkorDB dialect detection is by class name.
_FalkorLike.__name__ = "FalkorDBBackend"


def test_ensure_id_indexes_uses_bare_form_only():
    from agent_utilities.knowledge_graph.migration import _ensure_id_indexes

    be = _FalkorLike()
    n = _ensure_id_indexes(be, {"Concept", "idea_block"})
    assert n == 2
    # bare form ONLY — never the IF NOT EXISTS variant that FalkorDB rejects.
    assert be.index_ddl and all("IF NOT EXISTS" not in q for q in be.index_ddl)
    assert all("ON (n.id)" in q for q in be.index_ddl)


def test_ensure_id_indexes_syntax_error_not_counted_as_success():
    from agent_utilities.knowledge_graph.migration import _ensure_id_indexes

    # A syntax error containing the word "EXISTS" must NOT be read as "already there".
    be = _FalkorLike(
        error=Exception("server:ResponseError errors near 'EXISTS' (syntax)")
    )
    n = _ensure_id_indexes(be, {"Concept"})
    assert n == 0


def test_ensure_id_indexes_already_present_is_success():
    from agent_utilities.knowledge_graph.migration import _ensure_id_indexes

    be = _FalkorLike(error=Exception("Index already indexed for label Concept"))
    n = _ensure_id_indexes(be, {"Concept"})
    assert n == 1
