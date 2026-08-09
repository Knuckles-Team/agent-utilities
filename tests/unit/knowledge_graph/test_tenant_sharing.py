"""Tests for hierarchical org→user data segmentation + sharing (CONCEPT:AU-KG.compute.data-is-private-its).

Covers:
- private-by-default ownership stamping (skips privileged/system writers)
- the owner/scope visibility predicate + Cypher injection
- accessible_graphs ordering (org first, commons last)
- read_union dedup (org rows win over commons rows)
- explicit sharing transitions (share_with_org / make_private / promote_to_commons)
"""

from __future__ import annotations

import pytest

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
    ts.stamp_ownership(props, _user("root", "acme", roles=("kg:admin",)))
    assert ts.OWNER_KEY not in props  # privileged writes stay unowned/visible
    assert props[ts.TENANT_KEY] == "acme"  # but still tenant-attributed


def test_stamp_ownership_rejects_unverified_system_actor():
    props: dict = {}
    with pytest.raises(PermissionError):
        ts.stamp_ownership(props, ActorContext(actor_id="system"))


def test_stamp_ownership_does_not_overwrite_existing_share():
    props = {ts.SCOPE_KEY: ts.SCOPE_ORG}
    ts.stamp_ownership(props, _user("alice", "acme"))
    # An already-shared node is not silently reset to private.
    assert props[ts.SCOPE_KEY] == ts.SCOPE_ORG


# --- visibility predicate --------------------------------------------------


def test_visibility_predicate_for_user():
    pred, extra_params = ts.visibility_predicate(_user("alice", "acme"))
    assert "n._owner_id = $_visibility_owner_id" in pred
    assert "n._shared_scope IN ['org', 'commons']" in pred
    assert "n._owner_id IS NULL" in pred
    # D-W2T-2: the owner id is a bound parameter, never spliced into the text.
    assert extra_params == {"_visibility_owner_id": "alice"}
    assert "alice" not in pred


def test_visibility_predicate_none_for_privileged():
    assert ts.visibility_predicate(_user("root", roles=("kg:admin",))) is None
    with pytest.raises(PermissionError):
        ts.visibility_predicate(ActorContext(actor_id="system", roles=("system",)))


def test_generic_admin_role_is_not_graph_privileged():
    result = ts.visibility_predicate(_user("app-admin", roles=("admin",)))
    assert result is not None
    predicate, extra_params = result
    assert "_owner_id = $_visibility_owner_id" in predicate
    assert extra_params == {"_visibility_owner_id": "app-admin"}


def test_visibility_predicate_unsafe_id_fails_closed():
    pred, extra_params = ts.visibility_predicate(_user("alice' OR '1'='1", "acme"))
    assert extra_params == {"_visibility_owner_id": "__no_such_owner__"}
    assert "__no_such_owner__" not in pred  # now a param name, not spliced text
    assert "alice' OR" not in pred


def test_apply_visibility_injects_into_where():
    out, extra_params = ts.apply_visibility(
        "MATCH (n) WHERE n.x = 1 RETURN n", _user("alice")
    )
    assert "WHERE (n._owner_id = $_visibility_owner_id" in out
    # The pre-existing WHERE body is parenthesized as its own unit before being
    # ANDed with the visibility predicate: a bare `<visibility> AND n.x = 1`
    # splice would silently mis-group against any top-level OR already in the
    # caller's own predicate (AND binds tighter than OR), letting a disjunct
    # bypass visibility scoping entirely. See cypher_scoping.inject_and_predicate.
    assert "AND (n.x = 1)" in out
    assert extra_params == {"_visibility_owner_id": "alice"}


def test_apply_visibility_parenthesizes_existing_or_predicate():
    """A caller's own top-level OR must not let a disjunct bypass the injected
    visibility predicate (the same class of hole this fix closes for the
    tenant predicate in TenancyManager.scope_cypher_query)."""
    out, extra_params = ts.apply_visibility(
        "MATCH (p:Policy) WHERE p.name CONTAINS 'x' OR p.description CONTAINS 'x' RETURN p",
        _user("alice"),
    )
    assert "AND (p.name CONTAINS 'x' OR p.description CONTAINS 'x')" in out, out
    assert extra_params == {"_visibility_owner_id": "alice"}


def test_apply_visibility_injects_before_return():
    out, extra_params = ts.apply_visibility("MATCH (n) RETURN n", _user("alice"))
    assert "WHERE (n._owner_id = $_visibility_owner_id" in out
    assert out.rstrip().endswith("RETURN n")
    assert extra_params == {"_visibility_owner_id": "alice"}


def test_apply_visibility_noop_for_privileged():
    q = "MATCH (n) RETURN n"
    assert ts.apply_visibility(q, _user("root", roles=("kg:admin",))) == (q, {})


# --- accessible graphs -----------------------------------------------------


def test_accessible_graphs_org_first_commons_last():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    graphs = ts.accessible_graphs(_user("alice", "acme"), config=cfg)
    assert graphs[0] == "tenant__acme__kg"
    assert graphs[-1] == "kg"  # commons always last
    assert len(set(graphs)) == len(graphs)  # de-duplicated


def test_accessible_graphs_tenantless_is_rejected():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    with pytest.raises(PermissionError):
        ts.accessible_graphs(ActorContext(actor_id="x"), config=cfg)


# --- read union ------------------------------------------------------------


def test_read_union_dedups_org_wins():
    cfg = type("C", (), {"kg_default_graph": "kg"})()
    data = {
        "tenant__acme__kg": [{"id": "n1", "src": "org"}, {"id": "n2", "src": "org"}],
        "kg": [{"id": "n1", "src": "commons"}, {"id": "n3", "src": "commons"}],
    }

    def executor(graph, cypher, params):
        return data.get(graph, [])

    rows = ts.read_union(
        "MATCH (n) RETURN n", {}, executor, _user("alice", "acme"), config=cfg
    )
    by_id = {r["id"]: r["src"] for r in rows}
    assert by_id == {"n1": "org", "n2": "org", "n3": "commons"}  # org wins n1


def test_read_union_tolerates_missing_commons():
    cfg = type("C", (), {"kg_default_graph": "kg"})()

    def executor(graph, cypher, params):
        if graph == "kg":
            raise ConnectionError("commons down")
        return [{"id": "n1"}]

    rows = ts.read_union(
        "MATCH (n) RETURN n", {}, executor, _user("alice", "acme"), config=cfg
    )
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
    # BUG-6 (kg-exhaustive-smoke.md): _set_scope now existence-checks first —
    # the fake store must report the node as found for the SET call to happen.
    store = _FakeStore(
        rows=[
            {
                "id": "n1",
                "props": {
                    "id": "n1",
                    ts.TENANT_KEY: "acme",
                    ts.OWNER_KEY: "alice",
                },
            }
        ]
    )
    assert ts.share_with_org("n1", store=store, actor=_user("alice", "acme")) is True
    cypher, params = store.calls[-1]
    assert "_shared_scope = $scope" in cypher
    assert params["scope"] == ts.SCOPE_ORG
    assert params["id"] == "n1"


def test_make_private_sets_owner_to_caller():
    store = _FakeStore(
        rows=[
            {
                "id": "n1",
                "props": {
                    "id": "n1",
                    ts.TENANT_KEY: "acme",
                    ts.OWNER_KEY: "bob",
                },
            }
        ]
    )
    assert ts.make_private("n1", store=store, actor=_user("bob", "acme")) is True
    cypher, params = store.calls[-1]
    assert params["scope"] == ts.SCOPE_PRIVATE
    assert params["owner"] == "bob"


# --- BUG-6: share_with_org / make_private no-op (and report False) for a
# nonexistent node id, instead of silently "succeeding" with a MATCH that
# matched zero rows ------------------------------------------------------


def test_share_with_org_missing_node_returns_false():
    store = _FakeStore(rows=[])  # existence check finds nothing
    assert (
        ts.share_with_org("does-not-exist", store=store, actor=_user("alice", "acme"))
        is False
    )
    # No SET was ever issued — only the existence-check read happened.
    assert all("SET" not in cypher for cypher, _ in store.calls)


def test_make_private_missing_node_returns_false():
    store = _FakeStore(rows=[])
    assert (
        ts.make_private("does-not-exist", store=store, actor=_user("bob", "acme"))
        is False
    )
    assert all("SET" not in cypher for cypher, _ in store.calls)


def test_promote_to_commons_copies_node():
    src = _FakeStore(
        rows=[
            {
                "props": {
                    "id": "n1",
                    "title": "x",
                    ts.TENANT_KEY: "acme",
                    ts.OWNER_KEY: "alice",
                },
                "labels": ["Doc"],
            }
        ]
    )
    dst = _FakeStore()
    ok = ts.promote_to_commons(
        "n1", store=src, commons_store=dst, actor=_user("alice", "acme")
    )
    assert ok is True
    # Wrote into commons with commons scope. The write is per-property SET
    # (not a ``SET n += $props`` map-merge assignment, which exceeds the
    # engine's native Cypher write subset —
    # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184), so params
    # are flattened rather than nested under a single "props" key.
    dst_cypher, dst_params = dst.calls[0]
    assert "+=" not in dst_cypher
    assert dst_params[ts.SCOPE_KEY] == ts.SCOPE_COMMONS
    assert dst_params["id"] == "n1"


def test_promote_to_commons_missing_node_returns_false():
    src = _FakeStore(rows=[])
    dst = _FakeStore()
    assert (
        ts.promote_to_commons(
            "nope",
            store=src,
            commons_store=dst,
            actor=_user("alice", "acme"),
        )
        is False
    )


@pytest.mark.parametrize("transition", ["org", "commons", "private"])
def test_share_transition_rejects_non_owner_without_mutation(transition):
    src = _FakeStore(
        rows=[
            {
                "id": "n1",
                "props": {
                    "id": "n1",
                    ts.TENANT_KEY: "acme",
                    ts.OWNER_KEY: "alice",
                },
            }
        ]
    )
    dst = _FakeStore()

    with pytest.raises(PermissionError, match="owner or administrator"):
        if transition == "org":
            ts.share_with_org("n1", store=src, actor=_user("bob", "acme"))
        elif transition == "commons":
            ts.promote_to_commons(
                "n1",
                store=src,
                commons_store=dst,
                actor=_user("bob", "acme"),
            )
        else:
            ts.make_private("n1", store=src, actor=_user("bob", "acme"))

    assert all(" SET " not in f" {cypher} " for cypher, _ in src.calls)
    assert dst.calls == []


def test_share_transition_rejects_cross_tenant_admin():
    store = _FakeStore(
        rows=[
            {
                "props": {
                    "id": "n1",
                    ts.TENANT_KEY: "other-tenant",
                    ts.OWNER_KEY: "alice",
                }
            }
        ]
    )

    with pytest.raises(PermissionError, match="same-tenant"):
        ts.share_with_org(
            "n1",
            store=store,
            actor=_user("root", "acme", roles=("kg:admin",)),
        )


# --- ACL classification stamping (CONCEPT:AU-KG.backend.company-brain-write-guard) -----


def test_stamp_classification_defaults_to_confidential():
    props: dict = {}
    ts.stamp_classification(props, "Memory")
    assert props["classification"] == "confidential"


def test_stamp_classification_public_catalog_labels():
    for label in sorted(ts.PUBLIC_CATALOG_LABELS):
        props: dict = {}
        ts.stamp_classification(props, label)
        assert props["classification"] == "public"


def test_stamp_classification_unknown_label_is_conservative():
    props: dict = {}
    ts.stamp_classification(props, None)
    assert props["classification"] == "confidential"
    props2: dict = {}
    ts.stamp_classification(props2, "SomeBrandNewNodeType")
    assert props2["classification"] == "confidential"


def test_stamp_classification_never_overwrites_existing():
    props = {"classification": "restricted"}
    ts.stamp_classification(props, "ToolMetadata")
    assert props["classification"] == "restricted"


def test_stamp_classification_does_not_require_an_actor():
    # Unlike stamp_ownership, classification is label-driven, not
    # actor-driven -- it must not raise even with zero ambient identity.
    import contextvars

    def isolated():
        props: dict = {}
        ts.stamp_classification(props, "Memory")
        assert props["classification"] == "confidential"

    contextvars.Context().run(isolated)


# --- GOC-61 phase-1 system-graph write gate (W04 + 2026-08-09 owner ruling) -


def test_check_system_graph_write_noop_for_non_system_graph():
    # Not a system graph at all -> no-op regardless of actor/type.
    ts.check_system_graph_write("tenant__acme____commons__", "Memory", _user("mallory", "acme"))
    ts.check_system_graph_write("code:agent-utilities", "Memory", _user("mallory", "acme"))


def test_check_system_graph_write_denies_unprivileged_caller_of_shareable_type():
    """Known-bad input (a): an unprivileged caller writing a SHAREABLE type is refused.

    This is the AUTHORITY condition acting alone -- the node type (``Tool``)
    is legitimate commons content, but the caller holds no ``kg:admin`` and is
    not inside an authorized share verb.
    """
    mallory = _user("mallory", "acme")  # authenticated, no kg:admin
    with pytest.raises(PermissionError):
        ts.check_system_graph_write("__commons__", "Tool", mallory)


def test_check_system_graph_write_denies_privileged_caller_of_private_type():
    """Known-bad input (b): a FULLY PRIVILEGED (kg:admin) caller writing a
    PRIVATE-class type is refused. This is the one a coarser, single-condition
    gate gets wrong -- admin authority authorizes WHO may write to a system
    graph, never WHAT may be published there (2026-08-09 owner ruling).
    """
    root = _user("root", "acme", roles=("kg:admin",))
    for private_type in ("Memory", "Message", "InboundMessage", "Memento", "EvictedBlock", "Thread"):
        with pytest.raises(PermissionError):
            ts.check_system_graph_write("__commons__", private_type, root)


def test_check_system_graph_write_allows_privileged_caller_of_shareable_type():
    root = _user("root", "acme", roles=("kg:admin",))
    for shareable_type in sorted(ts.COMMONS_SHAREABLE_NODE_TYPES):
        ts.check_system_graph_write("__commons__", shareable_type, root)  # must not raise


def test_check_system_graph_write_denies_unprivileged_caller_even_for_control_graph():
    mallory = _user("mallory", "acme")
    with pytest.raises(PermissionError):
        ts.check_system_graph_write("__control__", "Tool", mallory)


def test_check_system_graph_write_share_verb_bypasses_authority_not_content():
    """The ambient _SHARE_VERB_ACTIVE context (promote_to_commons) waives the
    AUTHORITY condition but must never waive the CONTENT condition.
    """
    mallory = _user("mallory", "acme")  # no kg:admin
    token = ts._SHARE_VERB_ACTIVE.set(True)
    try:
        # Authority waived: an unprivileged actor may write a shareable type
        # from inside an already-authorized share verb.
        ts.check_system_graph_write("__commons__", "Skill", mallory)
        # Content NOT waived: the same context still refuses a private type.
        with pytest.raises(PermissionError):
            ts.check_system_graph_write("__commons__", "Memory", mallory)
    finally:
        ts._SHARE_VERB_ACTIVE.reset(token)


def test_check_system_graph_write_content_gate_is_commons_only_not_every_system_graph():
    """Regression guard: the CONTENT condition must be scoped to the commons
    graph specifically, not every ``is_system_graph()`` name -- otherwise
    this gate would break already-legitimate control-plane/secrets writes
    that were never part of the owner's commons-sharing ruling (e.g.
    ingest_profile.py's ``ProfileSpan`` nodes into ``__control__``,
    secrets_client.py's ``Secret`` nodes into ``__secrets__``). AUTHORITY
    still applies to those graphs (an unprivileged actor is still refused).
    """
    root = _user("root", "acme", roles=("kg:admin",))
    # Non-catalog types are NOT on COMMONS_SHAREABLE_NODE_TYPES, but writing
    # them into __control__/__secrets__ (not commons) must still succeed for
    # a privileged actor -- the content allowlist does not apply there.
    ts.check_system_graph_write("__control__", "ProfileSpan", root)  # must not raise
    ts.check_system_graph_write("__secrets__", "Secret", root)  # must not raise
    # AUTHORITY is still enforced for those graphs.
    mallory = _user("mallory", "acme")
    with pytest.raises(PermissionError):
        ts.check_system_graph_write("__control__", "ProfileSpan", mallory)


def test_check_system_graph_write_unauthenticated_system_path_exempted_from_authority_only():
    import contextvars

    def isolated():
        # No bound actor at all (genuine background/system write) -- exempt
        # from the AUTHORITY condition (matches stamp_ownership's existing
        # best-effort exemption), but the CONTENT gate still applies.
        ts.check_system_graph_write("__commons__", "MCPServer", None)  # must not raise
        with pytest.raises(PermissionError):
            ts.check_system_graph_write("__commons__", "Memory", None)

    contextvars.Context().run(isolated)
