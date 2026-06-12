"""Unit tests for TieredGraphBackend (CONCEPT:KG-2.7).

Verifies the write-through contract: reads served from L1, writes mirrored to
L3, L3 failures non-fatal, reconciliation, and L1 attribute delegation. Uses
recording fakes so no real Postgres/epistemic-graph server is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.tiered_backend import (
    TieredGraphBackend,
    _is_write,
)


class RecordingBackend(GraphBackend):
    """Fake GraphBackend that records calls and can be made to fail."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []

    def _maybe_fail(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} L3 down")

    def execute(self, query, params=None):
        self.calls.append(("execute", query))
        self._maybe_fail()
        return [{"backend": self.name, "query": query}]

    def execute_batch(self, query, batch):
        self.calls.append(("execute_batch", len(batch)))
        self._maybe_fail()
        return [{"backend": self.name}]

    def create_schema(self):
        self.calls.append(("create_schema", None))
        self._maybe_fail()

    def add_embedding(self, node_id, embedding):
        self.calls.append(("add_embedding", node_id))
        self._maybe_fail()

    def semantic_search(self, query_embedding, n_results=5):
        self.calls.append(("semantic_search", n_results))
        self._maybe_fail()
        return [{"backend": self.name}]

    def prune(self, criteria):
        self.calls.append(("prune", criteria))
        self._maybe_fail()

    def close(self):
        self.calls.append(("close", None))

    # L1-specific helper to exercise __getattr__ delegation
    def save_to_json(self, path):
        self.calls.append(("save_to_json", path))
        return path


def _ops(b: RecordingBackend) -> list[str]:
    return [c[0] for c in b.calls]


def test_is_write_classification():
    assert _is_write("CREATE (n:Foo {id:'x'})")
    assert _is_write("MATCH (n) SET n.x = 1")
    assert _is_write("MATCH (n) DETACH DELETE n")
    assert _is_write("MERGE (n:Foo {id:'x'})")
    assert not _is_write("MATCH (n:Foo) RETURN n")
    assert not _is_write("MATCH (n {id:$id}) RETURN n.name")


def test_read_served_from_l1_only():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    out = t.execute("MATCH (n:Foo) RETURN n")
    assert out == [{"backend": "l1", "query": "MATCH (n:Foo) RETURN n"}]
    assert "execute" in _ops(l1)
    assert l3.calls == []  # read never touches L3


def test_explicit_is_write_overrides_regex():
    # E5: a READ whose literal mentions a mutation keyword regex-false-positives;
    # an explicit is_write=False keeps it off L3. is_write=True forces a mirror
    # for a query the regex wouldn't flag.
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)

    # Regex would treat this read as a write (it contains 'SET'); explicit override.
    t.execute("MATCH (n) WHERE n.label = 'SET' RETURN n", is_write=False)
    assert l3.calls == [], "explicit read must not mirror to L3"

    # Explicit write of a query the keyword regex wouldn't match.
    t.execute("CALL custom.mutate()", is_write=True)
    assert _ops(l3) == ["execute"], "explicit write must mirror to L3"


def test_write_mirrored_to_both():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    out = t.execute("CREATE (n:Foo {id:'x'})")
    assert out[0]["backend"] == "l1"  # L1 result is authoritative
    assert _ops(l1) == ["execute"]
    assert _ops(l3) == ["execute"]
    assert t.durability_stats() == {
        "l3_writes": 1,
        "l3_failures": 0,
        "l1_reads": 0,
        "l3_reads": 0,
    }


def test_batch_and_embedding_mirror():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    t.execute_batch("UNWIND $rows AS r CREATE (n)", [{"a": 1}, {"a": 2}])
    t.add_embedding("n1", [0.1, 0.2])
    assert ("execute_batch", 2) in l3.calls
    assert ("add_embedding", "n1") in l3.calls
    assert t.durability_stats()["l3_writes"] == 2


def test_l3_failure_is_non_fatal():
    l1 = RecordingBackend("l1")
    l3 = RecordingBackend("l3", fail=True)
    t = TieredGraphBackend(l1, l3)
    # Write must still succeed and return L1's result despite L3 raising.
    out = t.execute("CREATE (n:Foo {id:'x'})")
    assert out[0]["backend"] == "l1"
    assert t.durability_stats() == {
        "l3_writes": 0,
        "l3_failures": 1,
        "l1_reads": 0,
        "l3_reads": 0,
    }


def test_semantic_search_from_l3():
    # Vector search is served from L3 (durable pgvector) where embeddings live;
    # L1's in-memory index is typically empty. (CONCEPT:KG-2.7)
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    res = t.semantic_search([0.1, 0.2], n_results=3)
    assert res == [{"backend": "l3"}]
    assert l1.calls == []  # L3 returned results; L1 not consulted


def test_semantic_search_falls_back_to_l1_when_l3_unavailable():
    # If L3 errors, fall back to L1 so vector search never silently dies.
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3", fail=True)
    t = TieredGraphBackend(l1, l3)
    res = t.semantic_search([0.1, 0.2], n_results=3)
    assert res == [{"backend": "l1"}]


def test_close_closes_both():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    t.close()
    assert ("close", None) in l1.calls
    assert ("close", None) in l3.calls


def test_getattr_delegates_to_l1():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    # save_to_json exists only on L1 — must resolve via __getattr__.
    assert t.save_to_json("/tmp/x.json") == "/tmp/x.json"
    assert ("save_to_json", "/tmp/x.json") in l1.calls
    with pytest.raises(AttributeError):
        _ = t.nonexistent_attribute


def test_reconcile_to_durable_copies_graph():
    """reconcile should MERGE every L1 node/edge into L3 (best-effort)."""

    class GraphStub:
        def _get_all_nodes(self):
            return ["a", "b"]

        def _get_node_properties(self, nid):
            return {"type": "Concept", "name": nid}

        def _get_all_edges(self):
            return [("a", "b", {"type": "RELATED_TO"})]

    class L1WithGraph(RecordingBackend):
        @property
        def graph(self):
            return GraphStub()

    l1 = L1WithGraph("l1")
    l3 = RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    summary = t.reconcile_to_durable()
    assert summary["nodes"] == 2  # node writes attempted
    assert summary["edges"] == 1  # edge writes attempted
    assert summary["errors"] == 0
    # L3 received node CREATE + edge MERGE writes (excluding the drift-count reads).
    writes = [
        c
        for c in l3.calls
        if c[0] == "execute" and ("CREATE (n:" in c[1] or "MERGE (a)" in c[1])
    ]
    assert len(writes) == 3  # 2 node CREATE + 1 edge MERGE
    # Exact-drift keys present (the recording L3 can't be counted → reported missing).
    assert "nodes_missing" in summary and "edges_missing" in summary
