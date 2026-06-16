"""Tiered backend traversal routing (CONCEPT:KG-2.7 P1).

Verifies that an id-anchored relationship traversal is served from L1 (the engine
resolves it natively), while a traversal L1 can't anchor falls through to L3.
"""

from agent_utilities.knowledge_graph.backends.tiered_backend import (
    TieredGraphBackend,
)


class FakeBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append(query)
        return [{"tier": self.name}]


def _tiered():
    return TieredGraphBackend(l1=FakeBackend("L1"), l3=FakeBackend("L3"))


def test_id_anchored_single_hop_hits_l1_only():
    t = _tiered()
    t.execute("MATCH (a {id: $x})-[:REL]->(b) RETURN b", {"x": "A"})
    assert t.l1.calls and not t.l3.calls
    assert t.durability_stats()["l1_reads"] == 1
    assert t.durability_stats()["l3_reads"] == 0


def test_id_anchored_var_length_hits_l1_only():
    t = _tiered()
    t.execute("MATCH (n)-[*1..3]-(t {id: $x}) RETURN n", {"x": "T"})
    assert t.l1.calls and not t.l3.calls
    assert t.durability_stats()["l1_reads"] == 1


def test_unanchored_traversal_falls_to_l3():
    t = _tiered()
    t.execute("MATCH (a)-[:REL]->(b) RETURN b")
    assert t.l3.calls and not t.l1.calls
    assert t.durability_stats()["l3_reads"] == 1


def test_node_read_stays_on_l1():
    t = _tiered()
    t.execute("MATCH (n {id: $x}) RETURN n", {"x": "A"})
    assert t.l1.calls and not t.l3.calls
    # node read is not a traversal, so neither traversal counter moves
    assert t.durability_stats()["l1_reads"] == 0
    assert t.durability_stats()["l3_reads"] == 0


def test_where_label_anchored_traversal_hits_l1():
    # The code-nav find_references shape: labeled pattern + WHERE prop anchor.
    # L1 holds the resolved code graph and resolves this natively (CONCEPT:KG-2.9g).
    t = _tiered()
    t.execute(
        "MATCH (caller:Code)-[:calls]->(def:Code) WHERE def.name = $n RETURN caller.name AS name",
        {"n": "NewAnalyzer"},
    )
    assert t.l1.calls and not t.l3.calls
    assert t.durability_stats()["l1_reads"] == 1


def test_var_length_where_anchored_traversal_hits_l1():
    # trace_call_graph / impact_of_change: bounded var-length + WHERE anchor.
    t = _tiered()
    t.execute(
        "MATCH (s:Code)-[:calls*1..3]->(x:Code) WHERE s.name = $n RETURN DISTINCT x.name AS name",
        {"n": "top"},
    )
    assert t.l1.calls and not t.l3.calls


def test_labeled_traversal_without_where_still_defers_to_l3():
    # No id anchor AND no WHERE prop anchor → L1 has no entry point → L3.
    t = _tiered()
    t.execute("MATCH (a:Code)-[:calls]->(b:Code) RETURN b")
    assert t.l3.calls and not t.l1.calls
    assert t.durability_stats()["l3_reads"] == 1


def test_write_goes_to_l1_then_mirrors_l3():
    t = _tiered()
    t.execute("MERGE (n:Thing {id: $x})", {"x": "A"})
    assert t.l1.calls and t.l3.calls  # authoritative L1 + durable mirror
