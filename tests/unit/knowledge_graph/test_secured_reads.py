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
        scoped = sr.scope("MATCH (n) RETURN n")
    assert "tenant_id = 'tenant-a'" in scoped


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
    """

    client = _EmptyGraphComputeClient()


class _FakeEngine:
    def __init__(self, backend: _FakeBackendReader) -> None:
        self.backend = backend
        self.graph_compute = _EmptyGraphCompute()
        self.graph = self.graph_compute


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
