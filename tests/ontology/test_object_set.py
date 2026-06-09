"""Unit tests for the Object Set Service (CONCEPT:KG-2.45).

These exercise :mod:`agent_utilities.knowledge_graph.ontology.object_set`
against a small, self-contained in-memory graph implementing the duck-typed read
surface :class:`GraphView` requires (``node_ids`` / ``_get_node_properties`` /
``has_node`` / ``get_successors`` / ``get_predecessors`` / ``out_edges`` /
``in_edges`` / ``_get_edge_properties``). No external backend, embedding model,
or Wave-1 ontology module is needed, so the suite is stable standalone.

Covers: static + dynamic evaluation, filter, lexical search, search-around
N-hop traversal, pivot, aggregate (count/sum/avg, grouped + global), TTL-bounded
temporary sets, and set algebra (union/intersect/subtract).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.object_set import (
    AggregationResult,
    ObjectSet,
    ObjectSetKind,
    PropertyFilter,
    dynamic_object_set,
    object_set_from_ids,
)


# ── a minimal in-memory graph implementing the GraphView read surface ─────────
class FakeGraph:
    """Directed labelled-property graph with the small read API GraphView needs."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        # adjacency: src -> list[(tgt, props)]
        self._out: dict[str, list[tuple[str, dict]]] = {}
        self._in: dict[str, list[tuple[str, dict]]] = {}

    def add_node(self, node_id: str, **props) -> None:
        self._nodes[node_id] = {"id": node_id, **props}
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])

    def add_edge(self, src: str, tgt: str, edge_type: str, **props) -> None:
        ep = {"type": edge_type, **props}
        self._out.setdefault(src, []).append((tgt, ep))
        self._in.setdefault(tgt, []).append((src, ep))

    # read surface ----------------------------------------------------------
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def _get_node_properties(self, node_id: str) -> dict:
        return dict(self._nodes.get(node_id, {}))

    def get_successors(self, node_id: str) -> list[str]:
        return [t for t, _ in self._out.get(node_id, [])]

    def get_predecessors(self, node_id: str) -> list[str]:
        return [s for s, _ in self._in.get(node_id, [])]

    def out_edges(self, node_id: str, data: bool = False):
        if data:
            return [(node_id, t, p) for t, p in self._out.get(node_id, [])]
        return [(node_id, t) for t, _ in self._out.get(node_id, [])]

    def in_edges(self, node_id: str, data: bool = False):
        if data:
            return [(s, node_id, p) for s, p in self._in.get(node_id, [])]
        return [(s, node_id) for s, _ in self._in.get(node_id, [])]

    def _get_edge_properties(self, src: str, tgt: str) -> dict:
        for t, p in self._out.get(src, []):
            if t == tgt:
                return dict(p)
        return {}


@pytest.fixture()
def graph() -> FakeGraph:
    g = FakeGraph()
    # Authors
    g.add_node("author:alice", type="person", name="Alice Researcher")
    g.add_node("author:bob", type="person", name="Bob Analyst")
    # Documents authored by them, with a department + a numeric cost
    g.add_node(
        "doc:1", type="document", name="Quarterly markets memo",
        department="research", cost=1000, content="markets and equities outlook",
    )
    g.add_node(
        "doc:2", type="document", name="Risk review",
        department="research", cost=500, content="risk exposure and hedging",
    )
    g.add_node(
        "doc:3", type="document", name="Ops handbook",
        department="ops", cost=250, content="operational runbook",
    )
    # A non-document node that must never leak into a type-scoped set
    g.add_node("place:hq", type="place", name="Headquarters")

    g.add_edge("author:alice", "doc:1", "authored")
    g.add_edge("author:alice", "doc:2", "authored")
    g.add_edge("author:bob", "doc:3", "authored")
    # A non-authored link that search-around by link-type must ignore
    g.add_edge("author:alice", "place:hq", "located_at")
    # 2-hop: documents cite other documents
    g.add_edge("doc:1", "doc:2", "cites")
    return g


# ── static + dynamic membership ───────────────────────────────────────────────


def test_static_set_membership(graph: FakeGraph) -> None:
    s = object_set_from_ids(graph, ["doc:1", "doc:2"], name="picked")
    assert s.kind is ObjectSetKind.STATIC
    assert sorted(s.ids()) == ["doc:1", "doc:2"]
    assert s.count() == 2
    assert "doc:1" in s
    objs = {o["id"]: o for o in s.objects()}
    assert objs["doc:1"]["name"] == "Quarterly markets memo"


def test_dynamic_set_auto_updates(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")], name="all-docs"
    )
    assert docs.kind is ObjectSetKind.DYNAMIC
    assert sorted(docs.ids()) == ["doc:1", "doc:2", "doc:3"]
    # Mutate the live graph -> the dynamic set reflects it on next read.
    graph.add_node("doc:4", type="document", name="Addendum", department="ops", cost=99)
    assert "doc:4" in docs.ids()
    assert docs.count() == 4
    # The non-document place is never a member.
    assert "place:hq" not in docs.ids()


# ── filter ────────────────────────────────────────────────────────────────────


def test_filter_with_typed_property_filters(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")]
    )
    research = docs.filter(
        filters=[PropertyFilter("department", "eq", "research")]
    )
    assert research.kind is ObjectSetKind.STATIC
    assert sorted(research.ids()) == ["doc:1", "doc:2"]
    expensive = docs.filter(filters=[PropertyFilter("cost", "gte", 1000)])
    assert expensive.ids() == ["doc:1"]


def test_filter_with_callable_predicate(graph: FakeGraph) -> None:
    docs = object_set_from_ids(graph, ["doc:1", "doc:2", "doc:3"])
    cheap = docs.filter(lambda p: p.get("cost", 0) < 600)
    assert sorted(cheap.ids()) == ["doc:2", "doc:3"]


# ── search (lexical degraded path; no embedding model in test env) ────────────


def test_search_substring_scan(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")]
    )
    hits = docs.search("risk hedging")
    assert "doc:2" in hits.ids()
    assert "doc:3" not in hits.ids()
    # Search is intersected with membership: searching a subset can't escape it.
    subset = object_set_from_ids(graph, ["doc:1"])
    assert subset.search("risk").ids() == []


# ── SEARCH-AROUND: typed N-hop traversal ──────────────────────────────────────


def test_search_around_one_hop_typed(graph: FakeGraph) -> None:
    authors = object_set_from_ids(graph, ["author:alice", "author:bob"])
    authored = authors.search_around("authored", hops=1, direction="out")
    assert sorted(authored.ids()) == ["doc:1", "doc:2", "doc:3"]
    # The 'located_at' edge type must be excluded by the typed traversal.
    assert "place:hq" not in authored.ids()


def test_search_around_two_hops(graph: FakeGraph) -> None:
    alice = object_set_from_ids(graph, ["author:alice"])
    # hop1: authored -> doc:1, doc:2 ; but 'cites' is a different type, so a
    # link-typed 2-hop on 'authored' stays within authored edges only.
    authored_only = alice.search_around("authored", hops=2, direction="out")
    assert sorted(authored_only.ids()) == ["doc:1", "doc:2"]
    # Any-type 2-hop reaches doc:1 -> (cites) -> doc:2 too (already present) and
    # alice -> place:hq via located_at.
    anyhop = alice.search_around(None, hops=2, direction="out")
    assert {"doc:1", "doc:2", "place:hq"}.issubset(set(anyhop.ids()))
    # Seed excluded by default.
    assert "author:alice" not in anyhop.ids()


def test_search_around_cap(graph: FakeGraph) -> None:
    authors = object_set_from_ids(graph, ["author:alice", "author:bob"])
    capped = authors.search_around("authored", hops=1, cap=1)
    assert capped.count() == 1


# ── PIVOT: linked set grouped by a target property ────────────────────────────


def test_pivot_across_link_type(graph: FakeGraph) -> None:
    authors = object_set_from_ids(graph, ["author:alice", "author:bob"])
    pv = authors.pivot("authored", group_by="department", direction="out")
    assert pv.link_type == "authored"
    assert pv.group_by == "department"
    assert sorted(pv.groups["research"]) == ["doc:1", "doc:2"]
    assert pv.groups["ops"] == ["doc:3"]
    # The linked set is the related document set.
    assert sorted(pv.linked_set.ids()) == ["doc:1", "doc:2", "doc:3"]


# ── aggregations ──────────────────────────────────────────────────────────────


def test_aggregate_count_global_and_grouped(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")]
    )
    total = docs.aggregate("count")
    assert isinstance(total, AggregationResult)
    assert total.value == 3.0
    by_dept = docs.aggregate("count", group_by="department")
    assert by_dept.value is None  # grouped → no single scalar
    assert by_dept.groups == {"research": 2.0, "ops": 1.0}


def test_aggregate_sum_and_avg(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")]
    )
    total_cost = docs.aggregate("sum", field="cost")
    assert total_cost.value == pytest.approx(1750.0)
    sum_by_dept = docs.aggregate("sum", field="cost", group_by="department")
    assert sum_by_dept.groups["research"] == pytest.approx(1500.0)
    assert sum_by_dept.groups["ops"] == pytest.approx(250.0)
    avg_by_dept = docs.aggregate("avg", field="cost", group_by="department")
    assert avg_by_dept.groups["research"] == pytest.approx(750.0)
    assert avg_by_dept.groups["ops"] == pytest.approx(250.0)
    # min/max
    assert docs.aggregate("min", field="cost").value == pytest.approx(250.0)
    assert docs.aggregate("max", field="cost").value == pytest.approx(1000.0)


def test_aggregate_requires_field_for_numeric_metric(graph: FakeGraph) -> None:
    docs = object_set_from_ids(graph, ["doc:1"])
    with pytest.raises(ValueError):
        docs.aggregate("sum")


# ── set algebra ───────────────────────────────────────────────────────────────


def test_set_algebra(graph: FakeGraph) -> None:
    a = object_set_from_ids(graph, ["doc:1", "doc:2"])
    b = object_set_from_ids(graph, ["doc:2", "doc:3"])
    assert sorted(a.union(b).ids()) == ["doc:1", "doc:2", "doc:3"]
    assert a.intersect(b).ids() == ["doc:2"]
    assert a.subtract(b).ids() == ["doc:1"]
    # operator sugar
    assert sorted((a | b).ids()) == ["doc:1", "doc:2", "doc:3"]
    assert (a & b).ids() == ["doc:2"]
    assert (a - b).ids() == ["doc:1"]


# ── TEMPORARY set TTL semantics ───────────────────────────────────────────────


def test_temporary_set_ttl_resnapshot(graph: FakeGraph) -> None:
    docs = dynamic_object_set(
        graph, filters=[PropertyFilter("type", "eq", "document")]
    )
    temp = docs.as_temporary(ttl_seconds=1000.0)
    assert temp.kind is ObjectSetKind.TEMPORARY
    snap = sorted(temp.ids())
    assert snap == ["doc:1", "doc:2", "doc:3"]
    # Add a doc: the temporary snapshot does NOT change while the TTL holds.
    graph.add_node("doc:9", type="document", name="Late add", department="ops", cost=1)
    assert sorted(temp.ids()) == snap
    # Force expiry → it re-snapshots from the live dynamic source.
    temp._snapshot_at -= 10_000.0  # simulate elapsed TTL
    assert temp.is_expired()
    assert "doc:9" in temp.ids()


# ── error guards ──────────────────────────────────────────────────────────────


def test_dynamic_requires_predicate(graph: FakeGraph) -> None:
    with pytest.raises(ValueError):
        ObjectSet(graph, kind=ObjectSetKind.DYNAMIC)
