"""Fail-closed read-path permission, tenant, and audit tests."""

from __future__ import annotations

import contextvars
import json
import re

import pytest

from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.models.company_brain import (
    DataClassification,
    NodeACL,
)
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.actor_identity import ActorType
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

    def for_graph(self, _graph_name: str) -> _FakeEngine:
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


def test_durable_access_rows_preserves_shared_scope(monkeypatch, brain):
    """D-P0-U119 regression: `_durable_access_rows`' Cypher fallback must return
    `_shared_scope` (aliased `shared_scope`) alongside tenant/classification/
    owner/external_access -- it was silently dropped from both the query and
    the mapped result, so `_hydrate_missing_acls` never saw organization-share
    evidence for a durable row at all."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    backend = _FakeBackendReader(
        rows=[
            {
                "id": "artifact-1",
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:owner",
                "shared_scope": "org",
            }
        ]
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["artifact-1"])
    assert "n._shared_scope AS shared_scope" in backend.queries[0][0]
    assert rows["artifact-1"]["shared_scope"] == "org"


class _LabelAwareBackendReader:
    """Like ``_FakeBackendReader``, but actually enforces the query's label —
    proving ``_durable_access_rows`` issues a label-scoped ``MATCH (n:Label)``
    (CONCEPT: hot-lookup-labels) instead of always falling straight through to
    an unlabeled ``MATCH (n)`` full-graph scan."""

    _LABEL_RE = re.compile(r"MATCH \(n(?::(\w+))?\)")

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows  # each row also carries a "_label" key
        self.queries: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict, **_kw) -> list[dict]:
        self.queries.append((query, params))
        match = self._LABEL_RE.search(query)
        label = match.group(1) if match else None
        wanted = set(params.get("ids", []))
        return [
            {k: v for k, v in row.items() if k != "_label"}
            for row in self.rows
            if row.get("id") in wanted and (label is None or row.get("_label") == label)
        ]


def test_durable_access_rows_resolves_on_the_first_verified_label_query(
    monkeypatch, brain
):
    """A node whose label is in the verified fleet set is found by the FIRST
    label-scoped query — no unlabeled scan is ever issued for it."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    backend = _LabelAwareBackendReader(
        rows=[
            {
                "id": "tool_demo_thing",
                "_label": "Tool",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
            }
        ]
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["tool_demo_thing"])

    assert rows["tool_demo_thing"]["tenant_id"] == "tenant-a"
    assert len(backend.queries) == 1
    assert "MATCH (n:Tool)" in backend.queries[0][0]


def test_durable_access_rows_falls_back_to_unlabeled_for_a_non_fleet_label(
    monkeypatch, brain
):
    """A node OUTSIDE the verified fleet label set (e.g. a Memory node reached
    through the general ``permit()`` read path, not fleet registration) must
    still resolve correctly — every verified-label candidate misses, then the
    unlabeled fallback finds it. Correctness for every node type is preserved;
    only the fleet hot path gets the speedup."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    backend = _LabelAwareBackendReader(
        rows=[
            {
                "id": "memory:some-id",
                "_label": "Memory",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
            }
        ]
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["memory:some-id"])

    assert rows["memory:some-id"]["tenant_id"] == "tenant-a"
    assert len(backend.queries) == len(sr._LABELED_HYDRATION_CANDIDATES) + 1
    last_query = backend.queries[-1][0]
    assert last_query.startswith("MATCH (n) WHERE")


def test_org_shared_node_grants_any_same_tenant_reader(monkeypatch, brain):
    """D-P0-U119 core fix: a durable row stamped `_shared_scope=org` (always
    written alongside `_owner_id` by `tenant_sharing.stamp_ownership`) must be
    readable by a NON-OWNER in the SAME tenant through the governed per-node
    ACL gate (`permit`/`filter_rows`), not just by the owner -- this is the
    exact mechanism `filter_rows` (used by every `/graph/query`-style governed
    projection) applies BEFORE the raw-row `tenant_sharing.visible()` filter,
    so a false denial here drops the row before `visible()` is ever reached."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "artifact-1": {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:owner",
                "shared_scope": "org",
            }
        },
    )
    reader = ActorContext(
        "principal:someone-else",
        ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    with use_actor(reader):
        assert sr.permit(["artifact-1"]) == ["artifact-1"]


def test_org_shared_scope_never_grants_a_cross_tenant_reader(monkeypatch, brain):
    """The pre-existing cross-tenant hydration gate (tenant match, checked
    before any classification/owner/scope branch) must still deny an
    org-shared node to an actor in a DIFFERENT tenant -- org sharing is
    intra-tenant only, never a cross-tenant grant."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "artifact-1": {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:owner",
                "shared_scope": "org",
            }
        },
    )
    other_tenant_reader = ActorContext(
        "principal:someone-else",
        ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant-b",
        authenticated=True,
    )
    with use_actor(other_tenant_reader):
        assert sr.permit(["artifact-1"]) == []


def test_private_scope_does_not_grant_a_non_owner_same_tenant_reader(
    monkeypatch, brain
):
    """A node stamped `_shared_scope=private` (or empty) must NOT be widened by
    this fix -- only an explicit `org`/`commons` scope grants a non-owner
    reader; the private-by-default owner-only gate is unchanged."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {
            "artifact-1": {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:owner",
                "shared_scope": "private",
            }
        },
    )
    other_same_tenant = ActorContext(
        "principal:someone-else",
        ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    with use_actor(other_same_tenant):
        assert sr.permit(["artifact-1"]) == []


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
        views: dict[str, _MultiGraphFakeEngine] | None = None,
    ) -> None:
        self.backend = backend
        self.graph_compute = _NamedGraphCompute(graph_name)
        self._views = views or {}

    def for_graph(self, graph_name: str) -> _MultiGraphFakeEngine:
        if graph_name == self.graph_compute.graph_name:
            return self
        view = self._views.get(graph_name)
        if view is None:
            raise RuntimeError(f"backend has no named-graph view for {graph_name!r}")
        return view


def _graph_session(actor: ActorContext, graph: str) -> GraphSession:
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
    assert (
        str(unknown_exc.value)
        == str(denied_exc.value)
        == ("Node permission evaluation failed")
    )


# ---------------------------------------------------------------------------
# PERF-SR-1: id-indexed batch hydration accelerator
# (`secured_reads._id_indexed_batch_rows`) — closes the O(graph) unlabeled
# `MATCH (n)` cliff for ANY label, not just the ones hardcoded into
# `_LABELED_HYDRATION_CANDIDATES`.
# ---------------------------------------------------------------------------


class _IdIndexedNodeStore:
    """Stands in for `GraphComputeEngine`'s id-primary-key batch property
    read (`_get_node_properties_batch`) — the SAME capability
    `EpistemicGraphBackend.semantic_search` already uses to hydrate a
    candidate id set's properties in one round trip, independent of label.
    """

    def __init__(self, properties_by_id: dict[str, dict]) -> None:
        self._properties_by_id = properties_by_id
        self.calls: list[list[str]] = []

    def _get_node_properties_batch(self, node_ids: list[str]) -> dict[str, dict]:
        self.calls.append(list(node_ids))
        return {
            node_id: self._properties_by_id[node_id]
            for node_id in node_ids
            if node_id in self._properties_by_id
        }


class _AcceleratedBackendReader(_FakeBackendReader):
    """A backend exposing BOTH `execute_read` (the required Cypher capability
    check) and `.graph` (the id-indexed accelerator) — models the production
    `EpistemicGraphBackend`, which exposes both surfaces over the same
    underlying node store."""

    def __init__(self, rows: list[dict], node_store: _IdIndexedNodeStore) -> None:
        super().__init__(rows)
        self.graph = node_store


def test_durable_access_rows_resolves_non_fleet_label_via_id_indexed_batch(
    monkeypatch, brain
):
    """A label OUTSIDE `_LABELED_HYDRATION_CANDIDATES` (e.g. `Preference`,
    the toggle-state case this fix must unblock) resolves WITHOUT any Cypher
    round trip at all when the backend exposes the id-indexed accelerator --
    proving the cliff (a non-allowlisted label falling through every
    candidate then paying the O(graph) unlabeled `MATCH (n)` scan) is closed
    generically, not by adding "Preference" to the tuple. Assert on the
    query/call SHAPE (zero Cypher queries issued), never on wall-clock."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    node_store = _IdIndexedNodeStore(
        {
            "pref:dark-mode": {
                "tenant_id": "tenant-a",
                "classification": "internal",
                "external_access": None,
                "_owner_id": "principal:verified",
                "_shared_scope": "private",
            }
        }
    )
    backend = _AcceleratedBackendReader(rows=[], node_store=node_store)
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["pref:dark-mode"])

    assert rows["pref:dark-mode"]["tenant_id"] == "tenant-a"
    assert rows["pref:dark-mode"]["classification"] == "internal"
    assert rows["pref:dark-mode"]["owner_id"] == "principal:verified"
    assert rows["pref:dark-mode"]["shared_scope"] == "private"
    # The load-bearing assertion: no labeled-candidate loop, and no unlabeled
    # full-graph scan -- zero Cypher queries issued at all.
    assert backend.queries == []
    assert node_store.calls == [["pref:dark-mode"]]


def test_id_indexed_accelerator_falls_through_when_it_cannot_resolve(
    monkeypatch, brain
):
    """The accelerator is purely additive: an id it cannot answer for
    (absent from its store) must still fall through to the existing labeled/
    unlabeled Cypher path exactly as before this fix -- never silently
    drops the id."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    node_store = _IdIndexedNodeStore({})  # never resolves anything
    backend = _AcceleratedBackendReader(
        rows=[
            {
                "id": "memory:some-id",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
                "owner_id": None,
                "shared_scope": None,
            }
        ],
        node_store=node_store,
    )
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    rows = sr._durable_access_rows(["memory:some-id"])

    assert rows["memory:some-id"]["tenant_id"] == "tenant-a"
    # Accelerator was consulted first (and declined), THEN Cypher answered.
    assert node_store.calls == [["memory:some-id"]]
    assert backend.queries  # fell through to the (still correct) Cypher path


def test_id_indexed_accelerator_produces_identical_authorization_to_cypher_path(
    monkeypatch, brain
):
    """The load-bearing safety test. Whether or not the id-indexed
    accelerator answers a lookup, the FINAL authorization decision
    (``permit()``'s grant/deny) must be IDENTICAL -- checked for both an
    allowlisted label (Tool) and a non-allowlisted one (Preference), and for
    both a granted and a denied case. The accelerator must never grant
    something the slow Cypher path would deny, or vice versa."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    actor = _actor("kg:read")

    # (node_id, label, raw node properties, expected permit() outcome)
    scenarios = [
        (
            "tool_owned",
            "Tool",
            {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "_owner_id": "principal:verified",
                "_shared_scope": "private",
            },
            True,
        ),
        (
            "pref_owned",
            "Preference",
            {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "_owner_id": "principal:verified",
                "_shared_scope": "private",
            },
            True,
        ),
        (
            "tool_other_tenant",
            "Tool",
            {
                "tenant_id": "tenant-b",
                "classification": "confidential",
                "external_access": None,
                "_owner_id": "someone-else",
                "_shared_scope": "private",
            },
            False,
        ),
        (
            "pref_other_tenant",
            "Preference",
            {
                "tenant_id": "tenant-b",
                "classification": "confidential",
                "external_access": None,
                "_owner_id": "someone-else",
                "_shared_scope": "private",
            },
            False,
        ),
        (
            "pref_unowned",
            "Preference",
            {
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "_owner_id": "",
                "_shared_scope": "",
            },
            False,
        ),
    ]

    for node_id, label, props, expect_granted in scenarios:
        cypher_row = {
            "id": node_id,
            "_label": label,
            "tenant_id": props["tenant_id"],
            "classification": props["classification"],
            "external_access": props["external_access"],
            "owner_id": props["_owner_id"],
            "shared_scope": props["_shared_scope"],
        }

        # -- Slow path: execute_read-only backend, label-loop then unlabeled
        # scan (no accelerator exposed at all).
        reset_company_brain()
        slow_backend = _LabelAwareBackendReader(rows=[cypher_row])
        monkeypatch.setattr(
            IntelligenceGraphEngine, "_ACTIVE_ENGINE", _FakeEngine(slow_backend)
        )
        with use_actor(actor):
            slow_granted = sr.permit([node_id], actor) == [node_id]

        # -- Fast path: the SAME underlying data through the id-indexed
        # accelerator instead of Cypher.
        reset_company_brain()
        node_store = _IdIndexedNodeStore({node_id: props})
        fast_backend = _AcceleratedBackendReader(rows=[], node_store=node_store)
        monkeypatch.setattr(
            IntelligenceGraphEngine, "_ACTIVE_ENGINE", _FakeEngine(fast_backend)
        )
        with use_actor(actor):
            fast_granted = sr.permit([node_id], actor) == [node_id]

        assert node_store.calls, f"{node_id}: accelerator was never consulted"
        assert slow_granted == expect_granted, f"{node_id}: unexpected slow-path result"
        assert fast_granted == expect_granted, f"{node_id}: unexpected fast-path result"
        assert slow_granted == fast_granted, (
            f"{node_id}: accelerator authorization diverged from the Cypher path"
        )


def test_row_without_governed_id_still_raises_with_accelerator_active(
    monkeypatch, brain
):
    """The identity requirement (``row_node_ids`` -> ``PermissionError``) is
    upstream of, and independent from, ``_durable_access_rows``/the new
    accelerator: a raw graph row that never carries an id must still raise,
    even when the active backend exposes the id-indexed accelerator this fix
    adds. Proves the accelerator cannot be used to smuggle an ungoverned row
    past the identity gate."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    node_store = _IdIndexedNodeStore(
        {"pref:x": {"tenant_id": "tenant-a", "classification": "public"}}
    )
    backend = _AcceleratedBackendReader(rows=[], node_store=node_store)
    engine = _FakeEngine(backend)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("kg:read")):
        with pytest.raises(PermissionError, match="governed node id"):
            sr.filter_rows([{"value": "no identity here"}])
    # row_node_ids rejects before hydration is ever attempted -- the
    # accelerator is never even reached.
    assert node_store.calls == []
