"""Read-path permission/tenancy/audit enforcement tests (CONCEPT:KG-2.6)."""

from __future__ import annotations

import os

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
from agent_utilities.security.brain_context import ActorContext, use_actor


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("KG_BRAIN_ENFORCE", "1")
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("KG_BRAIN_ENFORCE", raising=False)
    reset_company_brain()
    yield
    reset_company_brain()


def test_permit_is_noop_when_disabled(off):
    assert sr.permit(["a", "b"]) == ["a", "b"]


def test_confidential_node_filtered_for_unauthorized(enforced):
    brain = enforced
    brain.permissions.set_acl(
        NodeACL(
            node_id="hr:salary",
            classification=DataClassification.CONFIDENTIAL,
            read_roles=["hr"],
        )
    )
    # A marketing actor cannot see the HR node...
    with use_actor(ActorContext("agent:mk", ActorType.AI_AGENT, roles=("marketing",))):
        assert sr.permit(["hr:salary", "public:x"]) == ["public:x"]
    # ...but an HR actor can.
    with use_actor(ActorContext("agent:hr", ActorType.AI_AGENT, roles=("hr",))):
        assert set(sr.permit(["hr:salary", "public:x"])) == {"hr:salary", "public:x"}


def test_restricted_read_emits_audit(enforced):
    brain = enforced
    before = brain.provenance.read_count
    with use_actor(ActorContext("agent:x", ActorType.AI_AGENT)):
        sr.audit_read(["n:1", "n:2"], summary="test")
    assert brain.provenance.read_count == before + 1


def test_filter_rows_drops_denied(enforced):
    brain = enforced
    brain.permissions.set_acl(
        NodeACL(node_id="secret:1", classification=DataClassification.RESTRICTED)
    )
    rows: list[dict] = [{"id": "secret:1", "v": 1}, {"id": "ok:1", "v": 2}, {"v": 3}]
    with use_actor(ActorContext("agent:mk", ActorType.AI_AGENT, roles=("marketing",))):
        out = sr.filter_rows(rows)
    ids = [r.get("id") for r in out]
    assert "secret:1" not in ids
    assert "ok:1" in ids
    assert {"v": 3} in out  # unidentifiable rows are kept


def test_scope_injects_tenant_case_insensitive(enforced):
    with use_actor(ActorContext("agent:x", ActorType.AI_AGENT, tenant_id="acme")):
        scoped = sr.scope("match (n) return n")
    assert "tenant_id = 'acme'" in scoped
    assert scoped.lower().strip().endswith("return n")


def test_scope_rejects_injection(enforced):
    with use_actor(ActorContext("a", ActorType.AI_AGENT, tenant_id="x' OR '1'='1")):
        scoped = sr.scope("MATCH (n) RETURN n")
    assert "OR '1'='1" not in scoped  # unsafe id neutralized


def test_inherit_inferred_acl_propagates_restriction(enforced):
    brain = enforced
    brain.permissions.set_acl(
        NodeACL(node_id="parent", classification=DataClassification.RESTRICTED)
    )
    sr.inherit_inferred_acl("parent", "derived")
    acl = brain.permissions.get_acl("derived")
    assert acl is not None
    assert acl.classification == DataClassification.RESTRICTED
