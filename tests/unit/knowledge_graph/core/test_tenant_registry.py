"""Durable tenant-hierarchy registry (CONCEPT:AU-KG.compute.data-is-private-its).

``tenant_sharing.accessible_graphs()`` + ``read_union()`` were already
overlay/underlay with most-specific-wins; the only missing piece was somewhere
durable to record that a tenant HAS a parent. These tests pin that piece:

- a multi-level chain resolves in correct precedence order, commons always last
- a cycle is refused (and a cycle written behind the registry's back is survived)
- depth is bounded on BOTH the write and the read side
- ``read_union``'s first-in-chain-wins still prefers the most specific tenant
- only ``kg:admin`` may shape the tree (the engine has no ``can_write_row``)
- the chain survives a process restart (a fresh cache re-reads the same store)
- an unreachable registry degrades to flat tenancy, never an exception
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.knowledge_graph.core import tenant_registry as tr
from agent_utilities.knowledge_graph.core import tenant_sharing as ts
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

CFG = type("C", (), {"kg_default_graph": "kg"})()


def _user(actor_id="alice", tenant="acme", roles=()):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )


def _admin(tenant="acme"):
    return _user("root", tenant, roles=("kg:admin",))


class FakeControlGraph:
    """Stand-in for ``EpistemicGraphBackend().for_graph('__control__')``.

    Counts reads so the tests can assert the registry costs ONE label-indexed
    RPC per TTL window rather than one per ``accessible_graphs()`` call.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.reads = 0

    def add_node(self, node_id: str, **properties) -> None:
        self.nodes[node_id] = {"id": node_id, **properties}

    def nodes_by_label(self, label: str, limit: int = 0):
        self.reads += 1
        return [
            (nid, dict(props))
            for nid, props in self.nodes.items()
            if props.get("node_type") == label
        ]


@pytest.fixture
def store(monkeypatch):
    fake = FakeControlGraph()
    monkeypatch.setattr(tr, "_control_backend", lambda: fake)
    tr.invalidate_cache()
    yield fake
    tr.invalidate_cache()


def _register(store, chain: list[tuple[str, str]]):
    for child, parent in chain:
        tr.set_parent(child, parent, actor=_admin())
    tr.invalidate_cache()


# --- multi-level chain, precedence order, commons last ---------------------


def test_multi_level_chain_resolves_in_precedence_order(store):
    _register(store, [("eng", "acme"), ("acme", "holdings")])
    assert tr.ancestor_chain("eng") == ["acme", "holdings"]

    graphs = ts.accessible_graphs(_user("alice", "eng"), config=CFG)
    assert graphs == [
        "tenant__eng__kg",  # most specific: where alice's writes land
        "tenant__acme__kg",  # parent
        "tenant__holdings__kg",  # grandparent
        "kg",  # commons, ALWAYS last (GOC-61)
    ]


def test_commons_is_always_last_at_every_depth(store):
    _register(store, [("eng", "acme"), ("acme", "holdings")])
    for tenant in ("eng", "acme", "holdings"):
        graphs = ts.accessible_graphs(_user("alice", tenant), config=CFG)
        assert graphs[-1] == "kg"
        assert graphs[0] == f"tenant__{tenant}__kg"
        assert len(set(graphs)) == len(graphs)


def test_flat_tenant_is_unchanged_two_graphs(store):
    graphs = ts.accessible_graphs(_user("alice", "acme"), config=CFG)
    assert graphs == ["tenant__acme__kg", "kg"]


# --- cycle prevention -------------------------------------------------------


def test_self_parent_is_refused(store):
    with pytest.raises(ValueError, match="own parent"):
        tr.set_parent("acme", "acme", actor=_admin())


def test_cycle_is_refused(store):
    _register(store, [("eng", "acme"), ("acme", "holdings")])
    with pytest.raises(ValueError, match="cycle"):
        tr.set_parent("holdings", "eng", actor=_admin())
    # ...and nothing was written.
    tr.invalidate_cache()
    assert tr.parent_of("holdings") is None


def test_read_side_survives_a_cycle_written_behind_the_registry(store):
    """Defense in depth: the engine has no write-side row-ownership check, so a
    cycle can be planted directly. The read walk must terminate, not hang."""
    store.add_node(
        tr.registry_node_id("a"),
        node_type=tr.TENANT_HIERARCHY_LABEL,
        tenant_id="a",
        parent_tenant_id="b",
    )
    store.add_node(
        tr.registry_node_id("b"),
        node_type=tr.TENANT_HIERARCHY_LABEL,
        tenant_id="b",
        parent_tenant_id="a",
    )
    tr.invalidate_cache()
    assert tr.ancestor_chain("a") == ["b"]  # stops at the cycle, no repeat
    graphs = ts.accessible_graphs(_user("alice", "a"), config=CFG)
    assert graphs == ["tenant__a__kg", "tenant__b__kg", "kg"]


# --- bounded depth ----------------------------------------------------------


def test_depth_is_bounded_on_the_write_side(store):
    assert tr.MAX_TENANT_DEPTH == 4
    _register(store, [("l3", "l2"), ("l2", "l1"), ("l1", "l0")])
    assert tr.ancestor_chain("l3") == ["l2", "l1", "l0"]
    with pytest.raises(ValueError, match="MAX_TENANT_DEPTH"):
        tr.set_parent("l0", "l_minus_1", actor=_admin())


def test_depth_bound_accounts_for_the_subtree_below_the_tenant(store):
    """Re-parenting a tenant that already HAS descendants must count them."""
    _register(store, [("leaf", "mid"), ("mid", "top")])
    # leaf..mid..top..p is 4 levels — still allowed.
    tr.set_parent("top", "p", actor=_admin())
    # ...but leaf..mid..top..p..gp would be 5, counted through top's SUBTREE.
    with pytest.raises(ValueError, match="MAX_TENANT_DEPTH"):
        tr.set_parent("p", "gp", actor=_admin())


def test_depth_is_bounded_on_the_read_side_too(store):
    """A chain planted directly in ``__control__`` is truncated at read time."""
    for child, parent in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")]:
        store.add_node(
            tr.registry_node_id(child),
            node_type=tr.TENANT_HIERARCHY_LABEL,
            tenant_id=child,
            parent_tenant_id=parent,
        )
    tr.invalidate_cache()
    assert tr.ancestor_chain("a") == ["b", "c", "d"]  # MAX_TENANT_DEPTH - 1
    graphs = ts.accessible_graphs(_user("alice", "a"), config=CFG)
    assert len(graphs) == tr.MAX_TENANT_DEPTH + 1  # + commons
    assert graphs[-1] == "kg"


# --- read_union conflict resolution still prefers the most specific ---------


def test_read_union_prefers_the_most_specific_tenant_across_three_levels(store):
    """``read_union`` was NOT changed: first-in-chain-wins already generalises.

    Same node id present at all four levels -> the nearest tenant's row wins.
    """
    _register(store, [("eng", "acme"), ("acme", "holdings")])
    data = {
        "tenant__eng__kg": [{"id": "n1", "src": "eng"}],
        "tenant__acme__kg": [{"id": "n1", "src": "acme"}, {"id": "n2", "src": "acme"}],
        "tenant__holdings__kg": [
            {"id": "n1", "src": "holdings"},
            {"id": "n2", "src": "holdings"},
            {"id": "n3", "src": "holdings"},
        ],
        "kg": [
            {"id": "n1", "src": "commons", "node_type": "Tool"},
            {"id": "n4", "src": "commons", "node_type": "Tool"},
        ],
    }
    rows = ts.read_union(
        "MATCH (n) RETURN n",
        {},
        lambda g, c, p: data.get(g, []),
        _user("alice", "eng"),
        config=CFG,
    )
    by_id = {r["id"]: r["src"] for r in rows}
    assert by_id == {
        "n1": "eng",  # most specific wins over acme, holdings AND commons
        "n2": "acme",  # nearer ancestor wins over holdings
        "n3": "holdings",
        "n4": "commons",
    }


# --- authorization ----------------------------------------------------------


def test_only_kg_admin_may_set_a_parent(store):
    with pytest.raises(PermissionError, match="kg:admin"):
        tr.set_parent("eng", "acme", actor=_user("alice", "eng"))
    with pytest.raises(PermissionError, match="kg:admin"):
        tr.clear_parent("eng", actor=_user("alice", "eng"))
    # A generic application "admin" role is deliberately NOT graph authority.
    with pytest.raises(PermissionError, match="kg:admin"):
        tr.set_parent("eng", "acme", actor=_user("bob", "eng", roles=("admin",)))
    tr.invalidate_cache()
    assert tr.parent_of("eng") is None


def test_unauthenticated_actor_is_refused(store):
    with pytest.raises(PermissionError):
        tr.set_parent("eng", "acme", actor=ActorContext(actor_id="nobody"))


def test_invalid_tenant_id_is_refused(store):
    with pytest.raises(ValueError, match="invalid tenant id"):
        tr.set_parent("eng", "acme'; MATCH (n) DETACH DELETE n //", actor=_admin())


# --- durability -------------------------------------------------------------


def test_hierarchy_survives_a_process_restart(store):
    _register(store, [("eng", "acme")])
    # Simulate a fresh process: every in-memory cache is gone, the store is not.
    tr.invalidate_cache()
    assert tr.parent_of("eng") == "acme"
    assert ts.accessible_graphs(_user("alice", "eng"), config=CFG) == [
        "tenant__eng__kg",
        "tenant__acme__kg",
        "kg",
    ]


def test_clear_parent_detaches(store):
    _register(store, [("eng", "acme")])
    assert tr.clear_parent("eng", actor=_admin()).parent_tenant_id == ""
    assert tr.ancestor_chain("eng") == []
    assert ts.accessible_graphs(_user("alice", "eng"), config=CFG) == [
        "tenant__eng__kg",
        "kg",
    ]


# --- degradation ------------------------------------------------------------


def test_unreachable_registry_degrades_to_flat_tenancy(monkeypatch):
    def boom():
        raise ConnectionError("__control__ unreachable")

    monkeypatch.setattr(tr, "_control_backend", boom)
    tr.invalidate_cache()
    assert tr.ancestor_chain("eng") == []
    assert ts.accessible_graphs(_user("alice", "eng"), config=CFG) == [
        "tenant__eng__kg",
        "kg",
    ]
    tr.invalidate_cache()


# --- read cost --------------------------------------------------------------


def test_registry_costs_one_rpc_per_ttl_window_not_one_per_read(store):
    """``accessible_graphs`` is on the read hot path: the registry must not add
    an engine round-trip to every request."""
    _register(store, [("eng", "acme")])
    tr.invalidate_cache()
    before = store.reads
    for _ in range(50):
        ts.accessible_graphs(_user("alice", "eng"), config=CFG)
    assert store.reads - before == 1


def test_three_level_fanout_stays_inside_one_read_union_pool_wave(store):
    """The depth bound exists so a deeper chain costs ~1 RPC of WALL CLOCK, not
    N: ``read_union`` fans the per-graph queries out concurrently, bounded by
    ``_READ_UNION_MAX_WORKERS`` (8), and MAX_TENANT_DEPTH+1 = 5 <= 8."""
    assert tr.MAX_TENANT_DEPTH + 1 <= ts._READ_UNION_MAX_WORKERS
    _register(store, [("eng", "acme"), ("acme", "holdings")])
    rpc = 0.05  # stand-in for the engine's ~1s fixed RPC overhead

    def slow(graph, cypher, params):
        nonlocal rpc
        time.sleep(rpc)
        return [{"id": graph, "node_type": "Tool", "tenant_id": "eng"}]

    started = time.monotonic()
    rows = ts.read_union(
        "MATCH (n) RETURN n", {}, slow, _user("alice", "eng"), config=CFG
    )
    elapsed = time.monotonic() - started
    assert len(rows) == 4  # four graphs queried
    # Four sequential RPCs would be ~4x; concurrency keeps it near one.
    assert elapsed < rpc * 2.5, f"3-level read took {elapsed:.3f}s for a {rpc}s RPC"
