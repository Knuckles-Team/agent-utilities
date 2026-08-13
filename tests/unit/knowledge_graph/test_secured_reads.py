"""Fail-closed read-path permission, tenant, and audit tests."""

from __future__ import annotations

import contextvars
import json

import pytest

from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.models.company_brain import (
    ActorType,
    DataClassification,
    NodeACL,
)
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.brain_context import ActorContext, use_actor


def _actor(*roles: str, tenant: str = "tenant-a") -> ActorContext:
    return ActorContext(
        "principal:verified",
        ActorType.AI_AGENT,
        roles=roles,
        tenant_id=tenant,
        authenticated=True,
    )


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def _public_acl(node_id: str) -> NodeACL:
    return NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)


def test_missing_identity_fails_closed():
    def isolated():
        with pytest.raises(PermissionError):
            sr.permit(["node-a"])

    contextvars.Context().run(isolated)


def test_missing_acl_is_denied(brain):
    with use_actor(_actor("reader")):
        assert sr.permit(["unclassified"]) == []


def test_missing_acl_hydrates_once_from_durable_access(monkeypatch, brain):
    calls: list[list[str]] = []

    def durable(node_ids: list[str]):
        calls.append(node_ids)
        return {
            "trace-1": {
                "tenant_id": "tenant-a",
                "classification": "internal",
                "external_access": {
                    "is_public": False,
                    "user_emails": [],
                    "group_ids": [],
                    "read_roles": ["kg:read"],
                    "markings": [],
                },
            }
        }

    monkeypatch.setattr(sr, "_durable_access_rows", durable)
    with use_actor(_actor("kg:read")):
        assert sr.permit(["trace-1"]) == ["trace-1"]
        assert sr.permit(["trace-1"]) == ["trace-1"]
    assert calls == [["trace-1"]]


def test_durable_acl_hydration_rejects_cross_tenant_and_inconsistent_policy(
    monkeypatch, brain
):
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "other-tenant": {
                "tenant_id": "tenant-b",
                "classification": "public",
                "external_access": ExternalAccess.public().model_dump(),
            }
        },
    )
    with use_actor(_actor("kg:read")):
        assert sr.permit(["other-tenant"]) == []

    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "inconsistent": {
                "tenant_id": "tenant-a",
                "classification": "internal",
                "external_access": ExternalAccess.public().model_dump(),
            }
        },
    )
    with use_actor(_actor("kg:read")), pytest.raises(PermissionError):
        sr.permit(["inconsistent"])


def test_confidential_node_filtered_for_unauthorized(brain):
    brain.permissions.set_acl(
        NodeACL(
            node_id="salary",
            classification=DataClassification.CONFIDENTIAL,
            read_roles=["hr"],
        )
    )
    brain.permissions.set_acl(_public_acl("public-node"))
    with use_actor(_actor("marketing")):
        assert sr.permit(["salary", "public-node"]) == ["public-node"]
    with use_actor(_actor("hr")):
        assert set(sr.permit(["salary", "public-node"])) == {
            "salary",
            "public-node",
        }


def test_read_emits_audit(brain):
    before = brain.provenance.read_count
    with use_actor(_actor("reader")):
        sr.audit_read(["node-a"], summary="test")
    assert brain.provenance.read_count == before + 1


def test_filter_rows_drops_denied_and_requires_governed_ids(brain):
    brain.permissions.set_acl(
        NodeACL(node_id="secret", classification=DataClassification.RESTRICTED)
    )
    brain.permissions.set_acl(_public_acl("public-node"))
    with use_actor(_actor("marketing")):
        assert sr.filter_rows(
            [{"id": "secret", "value": 1}, {"id": "public-node", "value": 2}]
        ) == [{"id": "public-node", "value": 2}]
        with pytest.raises(PermissionError, match="governed node id"):
            sr.filter_rows([{"value": 3}])


def test_scope_injects_verified_tenant(brain):
    with use_actor(_actor("reader")):
        scoped, extra_params = sr.scope("MATCH (n) RETURN n")
    # D-W2T-2: the tenant id is a bound parameter, not spliced into the text.
    assert "tenant_id = $_tenant_scope_id" in scoped
    assert extra_params == {"_tenant_scope_id": "tenant-a"}


def test_tenantless_actor_is_rejected(brain):
    with use_actor(_actor("reader", tenant="")), pytest.raises(PermissionError):
        sr.scope("MATCH (n) RETURN n")


def test_permission_infrastructure_failure_never_returns_unfiltered(monkeypatch):
    monkeypatch.setattr(
        sr, "get_company_brain", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    with use_actor(_actor("reader")), pytest.raises(PermissionError):
        sr.permit(["node-a"])


def test_inherit_inferred_acl_propagates_restriction(brain):
    brain.permissions.set_acl(
        NodeACL(node_id="parent", classification=DataClassification.RESTRICTED)
    )
    sr.inherit_inferred_acl("parent", "derived")
    acl = brain.permissions.get_acl("derived")
    assert acl is not None
    assert acl.classification == DataClassification.RESTRICTED


class _FakeBackendReader:
    """Records ``execute_read`` calls; source of truth for durable ACL rows.

    Mirrors what a platform-node write (``IngestionMixin._upsert_node``, the
    seam ``ingest_mcp_server``/``add_atomic_skill`` share) actually populates:
    the node's governance fields live ONLY here, never in a compute
    scratchpad.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict, **_kw) -> list[dict]:
        self.queries.append((query, params))
        wanted = set(params.get("ids", []))
        return [row for row in self.rows if row.get("id") in wanted]


class _EmptyGraphComputeNodes:
    @staticmethod
    def properties_batch(_ids):
        return {}


class _EmptyGraphComputeClient:
    nodes = _EmptyGraphComputeNodes()


class _EmptyGraphCompute:
    """A distinct, never-written-to compute scratchpad (the pre-fix read target).

    Any attempt to read ACL material from here (instead of the backend) must
    come back empty, proving the fix no longer reaches into this object.

    ``graph_name`` mirrors whatever GraphSession is ambient (the suite-wide
    ``isolate_graph_compute_engine`` fixture always scopes one -- there is no
    "no session" case in this file's tests) so these tests, which are about
    ACL-hydration SOURCE, not physical-graph selection (R-22/GOC-67's
    ``_durable_access_rows`` narrowing lives elsewhere -- see
    ``test_governed_graph_retrieval.py``), never trip the narrowing branch by
    accident.
    """

    client = _EmptyGraphComputeClient()

    @property
    def graph_name(self) -> str:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
        return session.graph if session is not None else ""


class _FakeEngine:
    def __init__(self, backend: _FakeBackendReader) -> None:
        self.backend = backend
        self.graph_compute = _EmptyGraphCompute()
        self.graph = self.graph_compute

    def for_graph(self, _graph_name: str) -> "_FakeEngine":
        """Never narrows to a distinct object -- these tests are about ACL
        hydration SOURCE (backend vs. compute scratchpad), not multi-graph
        routing, so a for_graph() call (if the narrowing branch is ever
        entered) degrades to this SAME engine/backend rather than denying."""
        return self


def test_durable_acl_hydration_reads_the_backend_not_the_compute_scratchpad(
    monkeypatch, brain
):
    """D-W2-3 (secured_reads split-read): ``ingest_mcp_server`` and every other
    platform-node writer persist governance fields through ``self.backend``
    only (``IngestionMixin._upsert_node`` returns without touching
    ``self.graph``/``self.graph_compute`` whenever a backend is present). The
    pre-fix ``_durable_access_rows`` read ``active.graph_compute`` instead — a
    distinct object whenever the backend doesn't happen to alias its
    ``.graph`` to the same store — so a freshly ingested platform node's ACL
    was never found and the fail-closed guard denied it. This proves the read
    now goes through ``active.backend`` (the ONLY place the write landed) and
    a real platform node is granted.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    backend = _FakeBackendReader(
        rows=[
            {
                "id": "srv:demo-mcp-server",
                "tenant_id": "tenant-a",
                "classification": "internal",
                # Some backends serialize Map properties to a JSON string on
                # read; the hydrator must accept either shape.
                "external_access": json.dumps(
                    {
                        "is_public": False,
                        "user_emails": [],
                        "group_ids": [],
                        "read_roles": ["kg:read"],
                        "markings": [],
                    }
                ),
            }
        ]
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("kg:read")):
        assert sr.permit(["srv:demo-mcp-server"]) == ["srv:demo-mcp-server"]

    # The read went through the backend (source of truth) — not the empty
    # compute scratchpad the old implementation targeted.
    assert backend.queries
    assert backend.queries[0][1]["ids"] == ["srv:demo-mcp-server"]


def test_owner_fallback_grants_the_creator_read_access(monkeypatch, brain):
    """The core defect: a first-party write (no external_access descriptor)
    stamps classification/_owner_id via the write-time chokepoint
    (tenant_sharing.stamp_ownership/stamp_classification); the creator must be
    able to read their own data back without any explicit ACL call."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "mem-1": {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:verified",
            }
        },
    )
    with use_actor(_actor()):
        assert sr.permit(["mem-1"]) == ["mem-1"]


def test_owner_fallback_widens_only_the_owner_not_other_tenant_or_other_actor(
    monkeypatch, brain
):
    """Proves this fix restores the OWNER's access without widening anyone
    else's: a same-tenant non-owner and a cross-tenant actor both stay denied."""
    rows = {
        "mem-1": {
            "tenant_id": "tenant-a",
            "classification": "confidential",
            "external_access": None,
            "owner_id": "principal:verified",
        }
    }
    monkeypatch.setattr(sr, "_durable_access_rows", lambda _ids: dict(rows))

    # The owner, same tenant: granted.
    reset_company_brain()
    with use_actor(_actor()):
        assert sr.permit(["mem-1"]) == ["mem-1"]

    # A different actor, SAME tenant: still denied (not the owner, no PUBLIC,
    # no explicit grant).
    other_same_tenant = ActorContext(
        "principal:someone-else",
        ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    reset_company_brain()
    with use_actor(other_same_tenant):
        assert sr.permit(["mem-1"]) == []

    # The owner's own actor_id, but a DIFFERENT tenant: the pre-existing
    # cross-tenant hydration gate still denies before ownership is even
    # considered.
    cross_tenant_same_id = ActorContext(
        "principal:verified",
        ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant-b",
        authenticated=True,
    )
    reset_company_brain()
    with use_actor(cross_tenant_same_id):
        assert sr.permit(["mem-1"]) == []


def test_public_classification_fallback_grants_any_authenticated_actor(
    monkeypatch, brain
):
    """CallableResource/ToolMetadata-style catalog nodes: PUBLIC classification
    synthesizes regardless of owner, matching the deliberate (not blanket)
    classification policy for platform capability catalog labels."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "srv:tool-1": {
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
                "owner_id": None,
            }
        },
    )
    with use_actor(_actor("no-special-role")):
        assert sr.permit(["srv:tool-1"]) == ["srv:tool-1"]


def test_unowned_non_public_node_stays_denied(monkeypatch, brain):
    """A node with no owner, no PUBLIC classification, and no external_access
    (e.g. a system/background write with no bound actor to stamp) has nothing
    to synthesize from and must stay denied -- unchanged, fail-closed
    behaviour, not a new gap."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "sys-1": {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": None,
            }
        },
    )
    with use_actor(_actor()):
        assert sr.permit(["sys-1"]) == []


def test_durable_access_rows_parses_json_and_native_external_access(monkeypatch, brain):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    backend = _FakeBackendReader(
        rows=[
            {
                "id": "native-node",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": {"is_public": True},  # already a dict
            },
            {
                "id": "json-node",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": json.dumps({"is_public": True}),
            },
        ]
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["native-node", "json-node"])
    assert rows["native-node"]["external_access"] == {"is_public": True}
    assert rows["json-node"]["external_access"] == {"is_public": True}


# ── R-22/GOC-67 (defect 1): ACL hydration follows the SESSION's selected ────
# ── physical graph, never the active engine's own default backend ──────────


class _NamedGraphCompute:
    """Mirrors the ``graph_compute.graph_name`` shape ``IntelligenceGraphEngine.
    for_graph`` compares against to decide whether a view is even needed."""

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name


class _MultiGraphFakeEngine:
    """A fake ``IntelligenceGraphEngine`` bound to ONE physical graph, with a
    real ``for_graph()`` returning a lightweight, PRE-BUILT view bound to a
    different graph's own backend -- exactly the zero-transport,
    no-new-socket contract ``IntelligenceGraphEngine.for_graph`` documents in
    production. An unknown graph name has no view and raises, mirroring the
    real ``for_graph``'s ``RuntimeError`` when the backend exposes no
    named-graph view.
    """

    def __init__(
        self,
        graph_name: str,
        backend: _FakeBackendReader,
        *,
        views: dict[str, "_MultiGraphFakeEngine"] | None = None,
    ) -> None:
        self.backend = backend
        self.graph_compute = _NamedGraphCompute(graph_name)
        self._views = views or {}

    def for_graph(self, graph_name: str) -> "_MultiGraphFakeEngine":
        if graph_name == self.graph_compute.graph_name:
            return self
        view = self._views.get(graph_name)
        if view is None:
            raise RuntimeError(f"backend has no named-graph view for {graph_name!r}")
        return view


def _graph_session(actor: ActorContext, graph: str) -> "GraphSession":
    from agent_utilities.knowledge_graph.core.session import GraphSession

    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        graph=graph,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


def test_durable_access_rows_hydrates_from_session_graph_not_active_default(
    monkeypatch, brain
):
    """The core defect-1 proof: the ACTIVE engine's own bound graph is
    'graph-a' (its backend carries NO acl material for a graph-b node -- this
    stands in for "the process/default engine backend" the pre-fix code
    always read from) while the verified session has narrowed to 'graph-b'.
    ACL hydration must follow the SESSION, landing on graph-b's own backend
    via ``for_graph`` -- never on graph-a's, which is left untouched.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    backend_a = _FakeBackendReader(rows=[])  # graph-a: no matching ACL rows
    backend_b = _FakeBackendReader(
        rows=[
            {
                "id": "node-in-graph-b",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
            }
        ]
    )
    view_b = _MultiGraphFakeEngine("graph-b", backend_b)
    active = _MultiGraphFakeEngine("graph-a", backend_a, views={"graph-b": view_b})
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)

    actor = _actor("kg:read")
    session = _graph_session(actor, "graph-b")
    with use_actor(actor), use_session(session):
        assert sr.permit(["node-in-graph-b"]) == ["node-in-graph-b"]

    # graph-b's own backend served the hydration query...
    assert backend_b.queries
    assert backend_b.queries[0][1]["ids"] == ["node-in-graph-b"]
    # ...and graph-a's ("the process/default engine backend") was NEVER
    # touched -- pre-fix, EVERY hydration landed here instead and found
    # nothing, so this node stayed default-denied despite being real,
    # governed data on graph-b.
    assert backend_a.queries == []


def test_durable_access_rows_no_narrowing_when_session_matches_active_graph(
    monkeypatch, brain
):
    """No session/active mismatch -> no ``for_graph`` detour; hydration reads
    ``active.backend`` directly, byte-for-byte the pre-existing behavior."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    backend_a = _FakeBackendReader(
        rows=[
            {
                "id": "node-in-graph-a",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
            }
        ]
    )
    active = _MultiGraphFakeEngine("graph-a", backend_a)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)

    actor = _actor("kg:read")
    session = _graph_session(actor, "graph-a")
    with use_actor(actor), use_session(session):
        assert sr.permit(["node-in-graph-a"]) == ["node-in-graph-a"]
    assert backend_a.queries


def test_durable_access_rows_no_ambient_session_falls_back_to_active_default(
    monkeypatch, brain
):
    """Backward compatibility: a caller with no ``GraphSession`` at all (only
    the ambient ``ActorContext``, e.g. legacy call sites this fix must not
    break) keeps hydrating from the active engine's own backend exactly as
    before this fix existed. ``suspend_session()`` genuinely clears the
    ambient GraphSession -- the suite-wide ``isolate_graph_compute_engine``
    fixture otherwise always scopes one, so a bare ``use_actor`` block alone
    is not actually session-less in this test suite."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import suspend_session

    backend_a = _FakeBackendReader(
        rows=[
            {
                "id": "legacy-node",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
            }
        ]
    )
    active = _MultiGraphFakeEngine("graph-a", backend_a)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)

    with use_actor(_actor("kg:read")), suspend_session():
        from agent_utilities.knowledge_graph.core.session import current_session

        assert current_session() is None  # genuinely no ambient GraphSession
        assert sr.permit(["legacy-node"]) == ["legacy-node"]
    assert backend_a.queries


def test_durable_access_rows_unknown_graph_and_hydration_failure_both_deny_closed(
    monkeypatch, brain
):
    """Fail-closed, indistinguishably: a graph absent from the engine's own
    named-graph views (no ``for_graph`` target -- e.g. unknown/never
    materialized) and a graph whose view exists but whose backend rejects the
    hydration query (e.g. an authorization failure at the durable store) both
    raise the SAME exception type -- never a silent empty-ACL fallback that
    would look identical to "no rows matched" instead of "denied"."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    class _RejectingBackend:
        def execute_read(self, _query, _params, **_kw):
            raise RuntimeError("simulated authorization failure at the durable store")

    backend_a = _FakeBackendReader(rows=[])
    view_denied = _MultiGraphFakeEngine("graph-denied", _RejectingBackend())
    active = _MultiGraphFakeEngine(
        "graph-a", backend_a, views={"graph-denied": view_denied}
    )
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)
    actor = _actor("kg:read")

    # (1) Unknown graph -- no view for it at all.
    with use_actor(actor), use_session(_graph_session(actor, "graph-unknown")):
        with pytest.raises(PermissionError) as unknown_exc:
            sr.permit(["node-x"])

    # (2) Known graph, but its own backend rejects the hydration query.
    with use_actor(actor), use_session(_graph_session(actor, "graph-denied")):
        with pytest.raises(PermissionError) as denied_exc:
            sr.permit(["node-x"])

    assert type(unknown_exc.value) is PermissionError
    assert type(denied_exc.value) is PermissionError
    # Indistinguishable denial: ``permit()``'s own outer boundary re-wraps
    # EVERY internal failure into the identical caller-facing message -- an
    # unknown graph and a real authorization rejection cannot be told apart
    # from the outside, which is the point (never leak "does this graph
    # exist" as a side channel).
    assert str(unknown_exc.value) == str(denied_exc.value) == (
        "Node permission evaluation failed"
    )
