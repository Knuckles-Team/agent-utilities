"""Bounded subgraph checkout + delta write-back (CONCEPT:KG-2.7 P2)."""

from agent_utilities.knowledge_graph.core.subgraph_checkout import (
    CheckedOutSubgraph,
)


class FakeEngine:
    """In-memory stand-in for GraphComputeEngine (the L1 scratchpad)."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str], dict] = {}

    def add_node(self, nid, props=None, **_):
        self.nodes[nid] = dict(props or {})

    def add_edge(self, s, t, props=None, **_):
        self.edges[(s, t)] = dict(props or {})

    def remove_node(self, nid):
        self.nodes.pop(nid, None)
        self.edges = {k: v for k, v in self.edges.items() if nid not in k}

    def remove_edge(self, s, t, key=None):
        self.edges.pop((s, t), None)

    def has_node(self, nid):
        return nid in self.nodes

    def _get_node_properties(self, nid):
        return dict(self.nodes.get(nid, {}))

    def node_count(self):
        return len(self.nodes)


class FakeDurable:
    """Records execute() calls; can serve a conflict-detection node version."""

    def __init__(self, versions=None):
        self.calls: list[tuple[str, dict]] = []
        self._versions = versions or {}

    def execute(self, query, params=None):
        self.calls.append((query, params or {}))
        if "RETURN n" in query and params and "id" in params:
            nid = params["id"]
            if nid in self._versions:
                n = dict(self._versions[nid])
                n["id"] = nid
                return [{"n": n}]
            return []
        return []


def _checked_out(preload=None, durable=None, baseline=None):
    eng = FakeEngine()
    for nid, props in (preload or {}).items():
        eng.nodes[nid] = dict(props)
    return CheckedOutSubgraph(eng, durable=durable or FakeDurable(), baseline=baseline)


def test_mutations_recorded_as_deltas():
    sub = _checked_out()
    sub.add_node("x", {"type": "Thing", "name": "X"})
    sub.add_edge("x", "y", {"type": "REL"})
    assert sub.is_dirty()
    assert sub.delta_summary() == {
        "nodes_changed": 1,
        "nodes_deleted": 0,
        "edges_changed": 1,
        "edges_deleted": 0,
    }


def test_flush_writes_only_deltas_not_full_graph():
    # 100 pre-loaded nodes simulate a checked-out subgraph; we touch ONE.
    preload = {f"n{i}": {"type": "Thing", "name": f"n{i}"} for i in range(100)}
    durable = FakeDurable()
    sub = _checked_out(preload=preload, durable=durable)
    sub.add_node("n5", {"type": "Thing", "name": "n5", "status": "done"})
    summary = sub.flush_deltas_to_durable()
    assert summary["nodes_written"] == 1
    # Exactly one durable write — delta, not a 100-node enumeration.
    assert len(durable.calls) == 1
    q, params = durable.calls[0]
    assert q.startswith("MERGE (n:Thing {id: $id})") and params["id"] == "n5"
    assert not sub.is_dirty()  # ledger cleared on success


def test_flush_node_and_edge_upsert_forms():
    durable = FakeDurable()
    sub = _checked_out(durable=durable)
    sub.add_node("a", {"type": "Doc", "name": "A"})
    sub.add_edge("a", "b", {"type": "CITES"})
    summary = sub.flush_deltas_to_durable()
    assert summary["nodes_written"] == 1 and summary["edges_written"] == 1
    qs = [c[0] for c in durable.calls]
    assert any(q.startswith("MERGE (n:Doc {id: $id}) SET") for q in qs)
    assert any("MERGE (a)-[r:CITES]->(b)" in q for q in qs)


def test_flush_node_delete():
    durable = FakeDurable()
    sub = _checked_out(preload={"a": {"type": "Doc"}}, durable=durable)
    sub.remove_node("a")
    summary = sub.flush_deltas_to_durable()
    assert summary["nodes_deleted"] == 1
    assert any("DETACH DELETE" in c[0] for c in durable.calls)


def test_conflict_skip_leaves_conflicted_node():
    # Baseline = props at checkout; durable now holds different props → conflict.
    checkout_props = {"type": "Doc", "status": "open"}
    baseline = {"a": CheckedOutSubgraph._version_of(checkout_props)}
    durable = FakeDurable(versions={"a": {"type": "Doc", "status": "CHANGED_REMOTELY"}})
    sub = _checked_out(
        preload={"a": checkout_props}, durable=durable, baseline=baseline
    )
    sub.add_node("a", {"type": "Doc", "status": "mine"})
    summary = sub.flush_deltas_to_durable(on_conflict="skip")
    assert summary["conflicts"] == 1 and summary["skipped_conflicts"] == 1
    assert summary["nodes_written"] == 0  # the conflicted node was not overwritten


def test_conflict_error_raises():
    checkout_props = {"type": "Doc", "status": "open"}
    baseline = {"a": CheckedOutSubgraph._version_of(checkout_props)}
    durable = FakeDurable(versions={"a": {"type": "Doc", "status": "CHANGED"}})
    sub = _checked_out(
        preload={"a": checkout_props}, durable=durable, baseline=baseline
    )
    sub.add_node("a", {"type": "Doc", "status": "mine"})
    import pytest

    with pytest.raises(RuntimeError):
        sub.flush_deltas_to_durable(on_conflict="error")


def test_async_flush_writes_and_clears():
    durable = FakeDurable()
    sub = _checked_out(durable=durable)
    sub.add_node("a", {"type": "Doc", "name": "A"})
    t = sub.flush_deltas_async()
    t.join(timeout=5)
    assert not t.is_alive()
    assert any(c[0].startswith("MERGE (n:Doc") for c in durable.calls)
    assert not sub.is_dirty()  # snapshot owned the deltas; our ledger is clear


def test_reads_delegate_to_inner_engine():
    sub = _checked_out(preload={"a": {"type": "X"}, "b": {"type": "Y"}})
    # node_count is only on the inner engine — must delegate through __getattr__.
    assert sub.node_count() == 2
