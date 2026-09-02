"""Contract tests for the security-owned verified request authority."""

from __future__ import annotations

import time
from typing import Any, cast

import pytest

from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, CredentialLease
from agent_utilities.security.request_identity import (
    VerifiedRequestAuthority,
    build_verified_request_authority,
)


def _actor(**overrides: object) -> ActorContext:
    values: dict[str, object] = {
        "actor_id": "subject-a",
        "actor_type": ActorType.HUMAN,
        "roles": ("admin", "kg:write"),
        "tenant_id": "tenant-a",
        "authenticated": True,
        "groups": ("engineering",),
        "credential_expires_at": int(time.time()) + 300,
    }
    values.update(overrides)
    return ActorContext(**cast(Any, values))


def _authority(actor: ActorContext | None = None) -> VerifiedRequestAuthority:
    return build_verified_request_authority(
        actor or _actor(),
        audience="graph-service",
        policy_version="policy-7",
    )


def test_contract_projects_only_verified_lower_authority() -> None:
    actor = _actor()

    authority = _authority(actor)

    assert authority.actor is actor
    assert authority.subject == "subject-a"
    assert authority.actor_type is ActorType.HUMAN
    assert authority.tenant == "tenant-a"
    assert authority.scopes == frozenset({"kg:read", "kg:write"})
    assert authority.groups == ("engineering",)
    assert authority.audience == "graph-service"
    assert authority.policy_version == "policy-7"
    assert authority.expires_at == actor.credential_expires_at
    assert not hasattr(authority, "graph")
    assert not hasattr(authority, "trace_context")
    assert not hasattr(authority, "placement")
    assert not hasattr(authority, "session")


def test_generic_admin_never_widens_graph_scope() -> None:
    authority = _authority(_actor(roles=("admin", "kg:read")))

    assert authority.scopes == frozenset({"kg:read"})
    assert "kg:admin" not in authority.scopes


@pytest.mark.parametrize(
    ("actor", "audience", "policy_version", "message"),
    [
        (_actor(authenticated=False), "graph-service", "policy-7", "authenticated"),
        (_actor(actor_id=""), "graph-service", "policy-7", "subject"),
        (_actor(tenant_id=""), "graph-service", "policy-7", "tenant"),
        (_actor(), "", "policy-7", "audience or policy"),
        (_actor(), "graph-service", "", "audience or policy"),
        (
            _actor(credential_expires_at=None),
            "graph-service",
            "policy-7",
            "bounded expiry",
        ),
        (
            _actor(credential_expires_at=True),
            "graph-service",
            "policy-7",
            "bounded expiry",
        ),
        (
            _actor(groups=("engineering", "engineering")),
            "graph-service",
            "policy-7",
            "duplicate groups",
        ),
    ],
)
def test_contract_rejects_incomplete_or_ambiguous_authority(
    actor: ActorContext,
    audience: str,
    policy_version: str,
    message: str,
) -> None:
    with pytest.raises(PermissionError, match=message):
        build_verified_request_authority(
            actor,
            audience=audience,
            policy_version=policy_version,
        )


def test_contract_rejects_expired_actor() -> None:
    with pytest.raises(PermissionError, match="expired"):
        _authority(_actor(credential_expires_at=int(time.time()) - 1))


def test_contract_requires_fresh_projection_after_lease_renewal() -> None:
    lease = CredentialLease(int(time.time()) + 60)
    actor = _actor(credential_expires_at=None, credential_lease=lease)
    authority = _authority(actor)

    lease.renew(int(time.time()) + 120)

    with pytest.raises(PermissionError, match="expiry drifted"):
        authority.ensure_current()
    renewed = _authority(actor)
    renewed.ensure_current()
    assert renewed.expires_at == lease.expires_at


def test_authorities_for_two_tenants_remain_distinct() -> None:
    tenant_a = _authority(_actor(actor_id="subject-a", tenant_id="tenant-a"))
    tenant_b = _authority(_actor(actor_id="subject-b", tenant_id="tenant-b"))

    assert tenant_a != tenant_b
    assert tenant_a.tenant == "tenant-a"
    assert tenant_b.tenant == "tenant-b"
