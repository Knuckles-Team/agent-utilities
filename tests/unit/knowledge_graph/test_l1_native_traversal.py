"""L1 native traversal (CONCEPT:AU-KG.query.vendor-agnostic-traversal P1).

Exercises EpistemicGraphBackend's relationship interpreters against an injected
fake compute graph (no running engine needed): single-hop outbound/inbound,
bounded variable-length BFS, and the critical guard that an unhandled
relationship read returns [] rather than the whole graph.
"""

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class FakeGraph:
    """Tiny directed graph: A-[REL]->B-[REL]->C-[REL]->D, all type=Thing."""

    def __init__(self):
        self.nodes = {n: {"type": "Thing", "name": n} for n in "ABCD"}
        self.succ = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
        self.pred = {"A": [], "B": ["A"], "C": ["B"], "D": ["C"]}

    def has_node(self, nid):
        return nid in self.nodes

    def get_successors(self, nid):
        return list(self.succ.get(nid, []))

    def get_predecessors(self, nid):
        return list(self.pred.get(nid, []))

    def get_neighbors(self, nid):
        return list(
            dict.fromkeys(self.get_successors(nid) + self.get_predecessors(nid))
        )

    def _get_node_properties(self, nid):
        return dict(self.nodes.get(nid, {}))

    def _get_edge_properties(self, src, tgt):
        # Every edge in this fixture is a REL edge.
        if tgt in self.succ.get(src, []):
            return {"rel_type": "REL"}
        return {}

    def _get_all_nodes(self):
        return list(self.nodes)

    def _get_all_nodes_with_properties(self):
        return [(n, dict(p)) for n, p in self.nodes.items()]


def _backend():
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = FakeGraph()
    b._embeddings = {}
    b._node_counter = 0
    return b


def _ids(rows, var):
    out = []
    for r in rows:
        v = r.get(var)
        if isinstance(v, dict):
            out.append(v.get("id"))
    return out


def test_single_hop_outbound():
    b = _backend()
    rows = b.execute("MATCH (a {id: 'A'})-[:REL]->(b) RETURN b")
    assert _ids(rows, "b") == ["B"]


def test_single_hop_inbound():
    b = _backend()
    rows = b.execute("MATCH (a {id: 'B'})<-[:REL]-(b) RETURN b")
    assert _ids(rows, "b") == ["A"]


def test_var_length_undirected():
    b = _backend()
    # within 2 undirected hops of A: B (1 hop), C (2 hops)
    rows = b.execute("MATCH (n)-[*1..2]-(a {id: 'A'}) RETURN n")
    assert set(_ids(rows, "n")) == {"B", "C"}


def test_var_length_directed_outbound():
    b = _backend()
    rows = b.execute("MATCH (a {id: 'A'})-[*1..3]->(n) RETURN n")
    assert set(_ids(rows, "n")) == {"B", "C", "D"}


def test_var_length_directed_inbound():
    b = _backend()
    rows = b.execute("MATCH (a {id: 'D'})<-[*1..3]-(n) RETURN n")
    assert set(_ids(rows, "n")) == {"A", "B", "C"}


def test_unhandled_relationship_read_returns_empty_not_all_nodes():
    # No {id:...} anchor → L1 can't traverse → must return [], NOT every node.
    b = _backend()
    rows = b.execute("MATCH (a)-[:REL]->(b) RETURN b")
    assert rows == []
