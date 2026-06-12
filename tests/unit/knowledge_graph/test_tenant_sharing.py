"""Tests for hierarchical org→user data segmentation + sharing (CONCEPT:KG-2.60).

Covers:
- private-by-default ownership stamping (skips privileged/system writers)
- the owner/scope visibility predicate + Cypher injection
- accessible_graphs ordering (org first, commons last)
- read_union dedup (org rows win over commons rows)
- explicit sharing transitions (share_with_org / make_private / promote_to_commons)
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core import tenant_sharing as ts
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _user(actor_id="alice", tenant="acme", roles=()):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )


# --- ownership stamping ----------------------------------------------------


def test_stamp_ownership_private_by_default():
    props: dict = {}
    ts.stamp_ownership(props, _user("alice", "acme"))
    assert props[ts.TENANT_KEY] == "acme"  # drives the scope() predicate
    assert props[ts.OWNER_KEY] == "alice"
    assert props[ts.SCOPE_KEY] == ts.SCOPE_PRIVATE


def test_stamp_ownership_skips_privileged():
    props: dict = {}
    ts.stamp_ownership(props, _user("root", "acme", roles=("admin",)))
    assert ts.OWNER_KEY not in props  # privileged writes stay unowned/visible
    assert props[ts.TENANT_KEY] == "acme"  # but still tenant-attributed


def test_stamp_ownership_skips_system_actor():
    props: dict = {}
    ts.stamp_ownership(props, ActorContext(actor_id="system"))
    assert ts.OWNER_KEY not in props


def test_stamp_ownership_does_not_overwrite_existing_share():
    props = {ts.SCOPE_KEY: ts.SCOPE_ORG}
    ts.stamp_ownership(props, _user("alice", "acme"))
    # An already-shared node is not silently reset to private.
    assert props[ts.SCOPE_KEY] == ts.SCOPE_ORG


# --- visibility predicate --------------------------------------------------


def test_visibility_predicate_for_user():
    pred = ts.visibility_predicate(_user("alice", "acme"))
    assert "n._owner_id = 'alice'" in pred
    assert "n._shared_scope IN ['org', 'commons']" in pred
    assert "n._owner_id IS NULL" in pred


def test_visibility_predicate_none_for_privileged():
    assert ts.visibility_predicate(_user("root", roles=("admin",))) is None
    assert ts.visibility_predicate(ActorContext(actor_id="system", roles=("system",))) is None


def test_visibility_predicate_unsafe_id_fails_closed():
    pred = ts.visibility_predicate(_user("alice' OR '1'='1", "acme"))
    assert "__no_such_owner__" in pred
    assert "alice' OR" not in pred


def test_apply_visibility_injects_into_where():
    out = ts.apply_visibility("MATCH (n) WHERE n.x = 1 RETURN n", _user("alice"))
    assert "WHERE (n._owner_id = 'alice'" in out
    assert "AND n.x = 1" in out


def test_apply_visibility_injects_before_return():
    out = ts.apply_visibility("MATCH (n) RETURN n", _user("alice"))
    assert "WHERE (n._owner_id = 'alice'" in out
    assert out.rstrip().endswith("RETURN n")


def test_apply_visibility_noop_for_privileged():
    q = "MATCH (n) RETURN n"
    assert ts.apply_visibility(q, _user("root", roles=("admin",))) == q


# --- accessible graphs -----------------------------------------------------


def test_accessible_graphs_org_first_commons_last():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    graphs = ts.accessible_graphs(_user("alice", "acme"), config=cfg)
    assert graphs[0] == "tenant__acme__kg"
    assert graphs[-1] == "kg"  # commons always last
    assert len(set(graphs)) == len(graphs)  # de-duplicated


def test_accessible_graphs_tenantless_is_commons_only():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    assert ts.accessible_graphs(ActorContext(actor_id="x"), config=cfg) == ["kg"]


# --- read union ------------------------------------------------------------


def test_read_union_dedups_org_wins():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    data = {
        "tenant__acme__kg": [{"id": "n1", "src": "org"}, {"id": "n2", "src": "org"}],
        "kg": [{"id": "n1", "src": "commons"}, {"id": "n3", "src": "commons"}],
    }

    def executor(graph, cypher, params):
        return data.get(graph, [])

    rows = ts.read_union("MATCH (n) RETURN n", {}, executor, _user("alice", "acme"), config=cfg)
    by_id = {r["id"]: r["src"] for r in rows}
    assert by_id == {"n1": "org", "n2": "org", "n3": "commons"}  # org wins n1


def test_read_union_tolerates_missing_commons():
    cfg = type("C", (), {"kg_default_graph": "kg"})()

    def executor(graph, cypher, params):
        if graph == "kg":
            raise ConnectionError("commons down")
        return [{"id": "n1"}]

    rows = ts.read_union("MATCH (n) RETURN n", {}, executor, _user("alice", "acme"), config=cfg)
    assert [r["id"] for r in rows] == ["n1"]  # degrades to org-only


# --- sharing transitions ---------------------------------------------------


class _FakeStore:
    def __init__(self, rows=None):
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows or []

    def execute(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        return self._rows


def test_share_with_org_sets_scope():
    store = _FakeStore()
    ts.share_with_org("n1", store=store)
    cypher, params = store.calls[-1]
    assert "_shared_scope = $scope" in cypher
    assert params["scope"] == ts.SCOPE_ORG
    assert params["id"] == "n1"


def test_make_private_sets_owner_to_caller():
    store = _FakeStore()
    ts.make_private("n1", store=store, actor=_user("bob", "acme"))
    cypher, params = store.calls[-1]
    assert params["scope"] == ts.SCOPE_PRIVATE
    assert params["owner"] == "bob"


def test_promote_to_commons_copies_node():
    src = _FakeStore(rows=[{"props": {"id": "n1", "title": "x"}, "labels": ["Doc"]}])
    dst = _FakeStore()
    ok = ts.promote_to_commons("n1", store=src, commons_store=dst, actor=_user("alice", "acme"))
    assert ok is True
    # Wrote into commons with commons scope.
    dst_cypher, dst_params = dst.calls[0]
    assert dst_params["props"][ts.SCOPE_KEY] == ts.SCOPE_COMMONS


def test_promote_to_commons_missing_node_returns_false():
    src = _FakeStore(rows=[])
    dst = _FakeStore()
    assert ts.promote_to_commons("nope", store=src, commons_store=dst) is False
