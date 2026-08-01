"""L1 native traversal (CONCEPT:AU-KG.query.vendor-agnostic-traversal P1).

Exercises EpistemicGraphBackend's relationship reads: single-hop
outbound/inbound, bounded variable-length BFS, and an anchor-less match.

These used to run against an injected fake compute graph simulating a
Python-side "relationship interpreter" with its own silent-narrowing guard
(a read with no ``{id: ...}`` anchor returned ``[]`` rather than the whole
graph). That interpreter no longer exists: ``EpistemicGraphBackend.execute_read``
now always delegates the literal Cypher text to the engine's own native
Cypher executor (``GraphComputeEngine.query_cypher``, whose docstring is
explicit: "no Python-side regex interpretation... there is NO client-side
fallback that silently narrows or drops the query"). A ``FakeGraph`` lacking
a ``query_cypher`` method just raised ``AttributeError`` on every read here.
These tests now run against a real, isolated engine graph instead, and the
anchor-less-read assertion reflects the new (correct, fail-loud, no silent
narrowing) contract: a real, non-empty match result rather than [].
"""

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _backend() -> EpistemicGraphBackend:
    """A real, isolated backend seeded with A-[REL]->B-[REL]->C-[REL]->D."""
    compute = GraphComputeEngine(backend_type="rust")
    b = object.__new__(EpistemicGraphBackend)
    b._graph = compute
    b.graph_name = compute.graph_name
    b.create_schema()
    for n in "ABCD":
        b.add_node(n, node_type="Thing", name=n)
    for src, dst in [("A", "B"), ("B", "C"), ("C", "D")]:
        b.add_edge(src, dst, relationship="REL")
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
    # A match with no {id:...} anchor is now a perfectly ordinary native
    # Cypher query (no Python-side interpreter, no silent narrowing/fallback
    # -- see the module docstring) -- it returns every node with an incoming
    # :REL edge, not [] and not a silently-dropped/narrowed result.
    b = _backend()
    rows = b.execute("MATCH (a)-[:REL]->(b) RETURN b")
    assert set(_ids(rows, "b")) == {"B", "C", "D"}
