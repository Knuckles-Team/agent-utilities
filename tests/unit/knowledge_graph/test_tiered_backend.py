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
    t.flush_backfeed()  # mirror is write-behind by default; drain before asserting
    assert _ops(l3) == ["execute"], "explicit write must mirror to L3"


def test_write_mirrored_to_both():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    out = t.execute("CREATE (n:Foo {id:'x'})")
    assert out[0]["backend"] == "l1"  # L1 result is authoritative
    t.flush_backfeed()  # mirror is write-behind by default; drain before asserting
    assert _ops(l1) == ["execute"]
    assert _ops(l3) == ["execute"]
    assert t.durability_stats() == {
        "l3_writes": 1,
        "l3_failures": 0,
        "l1_reads": 0,
        "l3_reads": 0,
        "backfeed_queued": 0,
        "backfeed_inline": 0,
    }


def test_batch_and_embedding_mirror():
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3)
    t.execute_batch("UNWIND $rows AS r CREATE (n)", [{"a": 1}, {"a": 2}])
    t.add_embedding("n1", [0.1, 0.2])
    t.flush_backfeed()  # execute_batch is write-behind; embedding is synchronous
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
    t.flush_backfeed()  # let the (failing) write-behind mirror run before asserting
    assert t.durability_stats() == {
        "l3_writes": 0,
        "l3_failures": 1,
        "l1_reads": 0,
        "l3_reads": 0,
        "backfeed_queued": 0,
        "backfeed_inline": 0,
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
    # Now delegated to copy_graph → portable MERGE upserts: node `MERGE (n:Label …)`
    # and edge `… MERGE (s)-[r:…]->(t)` (excluding the edge label-lookup + drift reads).
    writes = [
        c
        for c in l3.calls
        if c[0] == "execute" and ("MERGE (n:" in c[1] or "MERGE (s)" in c[1])
    ]
    assert len(writes) == 3  # 2 node MERGE + 1 edge MERGE
    # Exact-drift keys present (the recording L3 can't be counted → reported missing).
    assert "nodes_missing" in summary and "edges_missing" in summary


def test_write_behind_drains_async_and_flushes():
    """B4: with write_behind on, L1 acks immediately and L3 mirrors drain on a
    background thread; flush_backfeed/close wait for the backlog (no loss)."""
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3, write_behind=True)
    try:
        for i in range(5):
            t.execute(f"CREATE (n:Foo {{id:'{i}'}})")
        # L1 has all writes immediately; L3 drains asynchronously.
        assert len(l1.calls) == 5
        t.flush_backfeed()  # block until the backfeed catches up
        assert _ops(l3) == ["execute"] * 5
        assert t.durability_stats()["l3_writes"] == 5
        assert t.durability_stats()["backfeed_queued"] == 0
    finally:
        t.close()


def test_write_behind_embeddings_stay_synchronous():
    """Embeddings mirror synchronously even in write-behind mode (semantic_search
    reads L3, so the vector must be durable before the next query)."""
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3, write_behind=True)
    try:
        t.add_embedding("n1", [0.1, 0.2, 0.3])
        # No flush needed — the embedding mirror is synchronous.
        assert ("add_embedding", "n1") in l3.calls
    finally:
        t.close()


# ---------------------------------------------------------------------------
# CONCEPT:KG-2.149 — per-backend durable fan-out
# ---------------------------------------------------------------------------


def test_fanout_mirrors_write_to_every_target():
    """A list of durable targets each receives the write on its own channel."""
    l1 = RecordingBackend("l1")
    age, neo, falkor = (
        RecordingBackend("pg-age"),
        RecordingBackend("neo4j"),
        RecordingBackend("falkordb"),
    )
    t = TieredGraphBackend(
        l1, [age, neo, falkor], mirror_names=["pg-age", "neo4j", "falkordb"]
    )
    try:
        t.execute("CREATE (n:Foo {id:'x'})")
        t.flush_backfeed()
        assert _ops(age) == ["execute"]
        assert _ops(neo) == ["execute"]
        assert _ops(falkor) == ["execute"]
        stats = t.durability_stats()
        assert stats["l3_writes"] == 3  # aggregate across all targets
        assert stats["targets"]["pg-age"]["l3_writes"] == 1
        assert stats["targets"]["neo4j"]["l3_writes"] == 1
        assert stats["targets"]["falkordb"]["l3_writes"] == 1
    finally:
        t.close()


def test_fanout_one_failing_target_does_not_block_others():
    """A failing/slow target fails into ITS OWN channel; healthy mirrors and L1
    are unaffected (CONCEPT:KG-2.149 isolation)."""
    l1 = RecordingBackend("l1")
    good = RecordingBackend("good")
    bad = RecordingBackend("bad", fail=True)
    t = TieredGraphBackend(l1, [good, bad], mirror_names=["good", "bad"])
    try:
        out = t.execute("CREATE (n:Foo {id:'x'})")
        assert out[0]["backend"] == "l1"  # L1 ack unaffected
        t.flush_backfeed()
        stats = t.durability_stats()
        # Healthy target committed; failing target recorded a failure, isolated.
        assert stats["targets"]["good"]["l3_writes"] == 1
        assert stats["targets"]["good"]["l3_failures"] == 0
        assert stats["targets"]["bad"]["l3_writes"] == 0
        assert stats["targets"]["bad"]["l3_failures"] == 1
        assert stats["l3_writes"] == 1 and stats["l3_failures"] == 1
    finally:
        t.close()


def test_fanout_primary_serves_reads():
    """The primary (index 0) is the read/semantic_search authority; secondaries
    are pure mirrors and never serve reads."""
    l1 = RecordingBackend("l1")
    primary, secondary = RecordingBackend("primary"), RecordingBackend("secondary")
    t = TieredGraphBackend(l1, [primary, secondary])
    try:
        assert t.l3 is primary  # back-compat: l3 == primary target
        t.semantic_search([0.1, 0.2], n_results=2)
        assert ("semantic_search", 2) in primary.calls
        assert secondary.calls == []  # secondary is mirror-only, no reads
    finally:
        t.close()


def test_single_target_back_compat_no_targets_key():
    """A single durable backend (the default) behaves exactly as before — no
    per-target breakdown key is emitted."""
    l1, l3 = RecordingBackend("l1"), RecordingBackend("l3")
    t = TieredGraphBackend(l1, l3, write_behind=False)
    try:
        t.execute("CREATE (n:Foo {id:'x'})")
        stats = t.durability_stats()
        assert "targets" not in stats
        assert stats["l3_writes"] == 1
    finally:
        t.close()
