"""Fail-closed read-path permission, tenant, and audit tests."""

from __future__ import annotations

import contextvars

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
