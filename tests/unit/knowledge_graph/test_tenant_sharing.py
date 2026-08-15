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


def test_stamp_ownership_privileged_still_gets_shared_scope_marker():
    """U-77: an unowned privileged write must still carry a native-recognized
    visibility marker (``_shared_scope``), or the engine's row-level guard
    (which denies any row with neither an owner nor a scope marker) makes the
    row invisible to native Cypher after a process restart even though it is
    present on disk and visible through the raw node/property API."""
    props: dict = {}
    ts.stamp_ownership(props, _user("root", "acme", roles=("kg:admin",)))
    assert ts.OWNER_KEY not in props  # still unowned -- no personal owner
    assert props[ts.SCOPE_KEY] == ts.SCOPE_ORG  # but now natively visible


def test_stamp_ownership_privileged_preserves_narrower_caller_scope():
    """A caller-supplied narrower scope (e.g. an explicit private share) must
    survive privileged stamping -- the ``org`` default is only a ``setdefault``,
    never an overwrite."""
    props: dict = {ts.SCOPE_KEY: ts.SCOPE_PRIVATE}
    ts.stamp_ownership(props, _user("root", "acme", roles=("kg:admin",)))
    assert props[ts.SCOPE_KEY] == ts.SCOPE_PRIVATE
    assert ts.OWNER_KEY not in props


def test_stamp_ownership_system_actor_still_gets_shared_scope_marker():
    """The ``actor_id == "system"`` early-return path is the same intentional
    "unowned platform write" case as ``is_privileged`` -- see the module
    docstring's "privileged/system writes are left unowned" -- so it must get
    the same native-visibility marker."""
    props: dict = {}
    actor = ActorContext(
        actor_id="system",
        actor_type=ActorType.SYSTEM,
        tenant_id="acme",
        authenticated=True,
    )
    ts.stamp_ownership(props, actor)
    assert ts.OWNER_KEY not in props
    assert props[ts.SCOPE_KEY] == ts.SCOPE_ORG


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


# --- GOC-61 phase-1 system-graph WRITE gate (W04 + 2026-08-09 owner ruling,
# write/read split) --------------------------------------------------------
#
# CONTENT is now a DENYLIST (COMMONS_PRIVATE_NODE_TYPES, six confirmed
# private-class types) instead of the original allowlist -- an evidence-
# backed audit found WorkItem/RuntimeSignal writers are actor-agnostic and
# reached from live MCP tool entrypoints under real tenant-bound sessions
# (mcp/tasks_extension.py:377, mcp/tools/job_tools.py:147,
# orchestration/agent_runner.py:467, messaging/router.py:334) -- an
# allowlist-on-write would have refused those already-working writes.
# COMMONS_SHAREABLE_NODE_TYPES (the old allowlist) now governs READ
# visibility instead -- see the "commons READ catalog restriction" section
# below.


def test_check_system_graph_write_noop_for_non_system_graph():
    # Not a system graph at all -> no-op regardless of actor/type.
    ts.check_system_graph_write("tenant__acme____commons__", "Message", _user("mallory", "acme"))
    ts.check_system_graph_write("code:agent-utilities", "Message", _user("mallory", "acme"))


def test_check_system_graph_write_denies_privileged_caller_of_private_type():
    """Known-bad input (a), re-verified against the narrowed denylist gate:
    a FULLY PRIVILEGED (kg:admin) caller writing a confirmed PRIVATE-class
    type is refused. This is the one a coarser, allowlist-shaped gate got
    wrong in the other direction -- admin authority authorizes WHO may write
    to a system graph, never WHAT may be published there (2026-08-09 owner
    ruling).
    """
    root = _user("root", "acme", roles=("kg:admin",))
    for private_type in sorted(ts.COMMONS_PRIVATE_NODE_TYPES):
        with pytest.raises(PermissionError):
            ts.check_system_graph_write("__commons__", private_type, root)


def test_check_system_graph_write_private_denylist_is_exactly_the_owner_six():
    assert ts.COMMONS_PRIVATE_NODE_TYPES == frozenset(
        {"Message", "Thread", "Memento", "ChatSummary", "InboundMessage", "EvictedBlock"}
    )


def test_check_system_graph_write_privileged_operational_type_now_allowed():
    """Operational/provenance/ontology types (not on the private denylist)
    are no longer refused by CONTENT for a privileged writer -- they were
    never the leak; the six confirmed private types are."""
    root = _user("root", "acme", roles=("kg:admin",))
    for operational_type in ("WorkItem", "RuntimeSignal", "Concept", "Evidence", "Tool", "Skill"):
        ts.check_system_graph_write("__commons__", operational_type, root)  # must not raise


def test_check_system_graph_write_tenant_bound_nonadmin_writes_workitem_permitted():
    """Regression proof (2026-08-09 owner ruling, second correction): a
    tenant-bound, non-admin caller writing WorkItem/RuntimeSignal into
    commons must be PERMITTED -- this is exactly the traffic the evidence
    audit found already working (mcp/tasks_extension.py:377,
    mcp/tools/job_tools.py:147, orchestration/agent_runner.py:467,
    messaging/router.py:334). AUTHORITY no longer requires kg:admin for
    commons specifically (see check_system_graph_write's docstring for why
    this is safe only in combination with the read-side catalog
    restriction -- filter_commons_catalog/apply_commons_catalog_restriction
    below are what makes this permissiveness non-leaking).
    """
    mallory = _user("mallory", "acme")  # tenant-bound, no kg:admin
    ts.check_system_graph_write("__commons__", "WorkItem", mallory)  # must not raise
    ts.check_system_graph_write("__commons__", "RuntimeSignal", mallory)  # must not raise


def test_check_system_graph_write_commons_authority_exempt_but_control_graph_is_not():
    """AUTHORITY's new commons exemption must be commons-specific -- an
    unprivileged actor is still refused for __control__/__secrets__ (see
    also test_check_system_graph_write_content_gate_is_commons_only_not_every_system_graph,
    which proves the converse: CONTENT does not apply to those graphs)."""
    mallory = _user("mallory", "acme")
    ts.check_system_graph_write("__commons__", "WorkItem", mallory)  # commons: no authority check
    with pytest.raises(PermissionError):
        ts.check_system_graph_write("__control__", "ProfileSpan", mallory)  # control: authority still applies


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
        # Authority waived: an unprivileged actor may write a non-private
        # type from inside an already-authorized share verb.
        ts.check_system_graph_write("__commons__", "Skill", mallory)
        # Content NOT waived: the same context still refuses a private type.
        with pytest.raises(PermissionError):
            ts.check_system_graph_write("__commons__", "Message", mallory)
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
    # ProfileSpan/Secret are not on the private denylist either, but writing
    # them into __control__/__secrets__ (not commons) must succeed for a
    # privileged actor regardless -- the content denylist only ever applies
    # to commons.
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
            ts.check_system_graph_write("__commons__", "Message", None)

    contextvars.Context().run(isolated)


# --- GOC-61 phase-1 commons READ catalog restriction (2026-08-09 owner
# ruling, write/read split) -------------------------------------------------


def _row(node_id, node_type=None, tenant=None, **extra):
    row = {"id": node_id}
    if node_type is not None:
        row["node_type"] = node_type
    if tenant is not None:
        row[ts.TENANT_KEY] = tenant
    row.update(extra)
    return row


_CATALOG_READ_ROWS = [
    _row("s1", "Skill"),
    _row("t1", "Tool"),
    _row("cr1", "CallableResource", resource_type="AGENT_SKILL"),
    _row("cr2", "CallableResource", resource_type="WORKFLOW"),  # NOT shareable
    _row("w1", "WorkItem", tenant="acme"),
    _row("rs1", "RuntimeSignal", tenant="acme"),
    _row("c1", "Concept", tenant="acme"),
    _row("m1", "Message", tenant="acme"),  # confirmed private
    _row("u1"),  # unclassifiable: no node_type at all
]


def test_filter_commons_catalog_cross_tenant_sees_only_catalog_types():
    """Read proof (new): a cross-tenant reader sees the catalog and nothing else."""
    bob = _user("bob", "globex")  # different tenant, no admin
    seen = {r["id"] for r in ts.filter_commons_catalog(_CATALOG_READ_ROWS, bob, "__commons__")}
    assert seen == {"s1", "t1", "cr1"}  # catalog types + AGENT_SKILL CallableResource only
    assert "cr2" not in seen  # CallableResource without resource_type=AGENT_SKILL
    assert not ({"w1", "rs1", "c1", "m1"} & seen)  # no operational or private data
    assert "u1" not in seen  # unclassifiable -> fails CLOSED


def test_filter_commons_catalog_same_tenant_still_sees_own_operational_data():
    alice = _user("alice", "acme")  # SAME tenant as the operational rows
    seen = {r["id"] for r in ts.filter_commons_catalog(_CATALOG_READ_ROWS, alice, "__commons__")}
    assert {"w1", "rs1", "c1", "m1"} <= seen  # own tenant's data, including Message here


def test_filter_commons_catalog_novel_type_not_visible_cross_tenant():
    """Read proof (new): deny-by-default survived the write/read split -- an
    unclassified/novel node type is invisible cross-tenant."""
    bob = _user("bob", "globex")
    alice = _user("alice", "acme")
    novel = [_row("n1", "SomeBrandNewNodeType2077", tenant="acme")]
    assert ts.filter_commons_catalog(novel, bob, "__commons__") == []
    assert len(ts.filter_commons_catalog(novel, alice, "__commons__")) == 1  # own tenant, not destroyed


def test_filter_commons_catalog_noop_for_privileged_actor():
    root = _user("root", "acme", roles=("kg:admin",))
    assert len(ts.filter_commons_catalog(_CATALOG_READ_ROWS, root, "__commons__")) == len(
        _CATALOG_READ_ROWS
    )


def test_filter_commons_catalog_noop_for_non_commons_graph():
    bob = _user("bob", "globex")
    assert len(
        ts.filter_commons_catalog(_CATALOG_READ_ROWS, bob, "tenant__acme____commons__")
    ) == len(_CATALOG_READ_ROWS)


def test_commons_catalog_predicate_noop_cases():
    bob = _user("bob", "globex")
    root = _user("root", "acme", roles=("kg:admin",))
    assert ts.commons_catalog_predicate(root, graph_name="__commons__") is None  # privileged
    assert ts.commons_catalog_predicate(bob, graph_name="__control__") is None  # not commons


def test_commons_catalog_predicate_carries_reader_tenant_and_types():
    bob = _user("bob", "globex")
    pred = ts.commons_catalog_predicate(bob, var="n", graph_name="__commons__")
    assert pred is not None
    cond, extra = pred
    assert "node_type" in cond
    assert ts.TENANT_KEY in cond
    assert extra == {"_commons_catalog_tenant_id": "globex"}


def test_apply_commons_catalog_restriction_injects_into_aggregate_query():
    bob = _user("bob", "globex")
    cyp, params = ts.apply_commons_catalog_restriction(
        "MATCH (n:WorkItem) RETURN count(n) AS c", bob, "__commons__", var="n"
    )
    assert "node_type" in cyp
    assert "count(n)" in cyp
    assert params == {"_commons_catalog_tenant_id": "globex"}


def test_apply_commons_catalog_restriction_noop_for_non_commons_graph():
    bob = _user("bob", "globex")
    q = "MATCH (n:ProfileSpan) RETURN count(n) AS c"
    assert ts.apply_commons_catalog_restriction(q, bob, "__control__", var="n") == (q, {})
