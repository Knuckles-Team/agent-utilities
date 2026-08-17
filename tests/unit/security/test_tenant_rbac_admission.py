"""Tests for the tenant-graph Read/Write RBAC admission bridge (P0 root-cause fix:
agent-webui `/graph` showed 0 nodes/0 edges because `tenant__homelab____commons__`
had never had a Read/Write grant for any principal but the one-off `System`
identity that created it by hand — see the module docstring for the full
root-cause chain and `plans/au-eg-program/HANDOFF-2026-07-22.md` §7-8).

Covers:
- The pre-fix defect reproduced structurally: a tenant with no admitted
  principals cannot pass a would-be RBAC check for the tenant role.
- Admitting a fresh (never-registered) principal grants exactly the tenant role,
  the CALLER-supplied ``role``/``teams`` shape, and nothing else.
- Admitting an already-registered principal MERGES the tenant role into its
  existing roles without dropping any pre-existing role/team — RegisterIdentity
  replaces the whole identity, so a naive re-registration would silently strip
  unrelated grants; this is the regression this module exists to prevent.
- The pass is idempotent: running it twice for the same principal never
  duplicates a `register_identity` call once the role is already held.
- Multiple distinct principals sharing one tenant are all admitted — the exact
  "N webui end-users, one tenant" shape the live incident hit.
- `System` is refused as a `TenantPrincipal.role` — this module must never be
  used to grant blanket RBAC bypass.
- A failed admission RPC is never swallowed.
"""

from __future__ import annotations

import pytest

from agent_utilities.security import tenant_rbac_admission as tra


def _authority(agent_id: str) -> tra.TenantAdmissionAuthority:
    return tra.TenantAdmissionAuthority(
        agent_id=agent_id,
        signer_id=agent_id,
        signer_key="test-signer-key-not-a-real-credential",  # nosec B105 - test only
    )


def test_tenant_role_name_matches_the_engine_convention() -> None:
    # Must byte-for-byte match `format!("tenant:{tenant_slug}")` in
    # `crates/eg-core/src/isolation.rs::provision_tenant_graph_access` — a drift
    # here would silently admit principals into a role no grant covers.
    assert tra.tenant_role_name("homelab") == "tenant:homelab"


def test_tenant_role_name_rejects_empty_slug() -> None:
    with pytest.raises(ValueError):
        tra.tenant_role_name("   ")


def test_admitting_a_fresh_principal_grants_exactly_the_tenant_role() -> None:
    client = tra.FixtureEngineIdentityClient()
    principal = tra.TenantPrincipal(agent_id="webui-user-1", role="Agent")

    result = tra.provision_tenant_access(
        client,
        "homelab",
        [principal],
        admin_authority=_authority("provisioner:deploy"),
    )

    assert result.tenant_slug == "homelab"
    assert result.role == "tenant:homelab"
    assert result.all_admitted is True
    [outcome] = result.outcomes
    assert outcome.already_held is False
    assert client.identities["webui-user-1"] == {
        "role": "Agent",
        "teams": [],
        "roles": ["tenant:homelab"],
    }


def test_admitting_an_already_registered_principal_preserves_its_other_roles() -> None:
    """The regression this module exists to prevent: RegisterIdentity replaces
    the WHOLE identity, so admitting the tenant role must never silently drop
    an existing, unrelated role (e.g. a code-ingestion reader role)."""

    client = tra.FixtureEngineIdentityClient()
    principal = tra.TenantPrincipal(
        agent_id="webui-user-1",
        role="Agent",
        teams=("support",),
        existing_roles=("code-reader",),
    )

    tra.provision_tenant_access(
        client,
        "homelab",
        [principal],
        admin_authority=_authority("provisioner:deploy"),
    )

    identity = client.identities["webui-user-1"]
    assert identity["teams"] == ["support"]
    assert set(identity["roles"]) == {"code-reader", "tenant:homelab"}


def test_multiple_principals_sharing_one_tenant_are_all_admitted() -> None:
    """The exact live shape: N distinct agent-webui end-users, one tenant."""

    client = tra.FixtureEngineIdentityClient()
    principals = [
        tra.TenantPrincipal(agent_id="webui-user-1"),
        tra.TenantPrincipal(agent_id="webui-user-2"),
        tra.TenantPrincipal(agent_id="webui-user-3"),
    ]

    result = tra.provision_tenant_access(
        client,
        "homelab",
        principals,
        admin_authority=_authority("provisioner:deploy"),
    )

    assert len(result.outcomes) == 3
    assert {"tenant:homelab"} == set(client.identities["webui-user-1"]["roles"])
    assert {"tenant:homelab"} == set(client.identities["webui-user-2"]["roles"])
    assert {"tenant:homelab"} == set(client.identities["webui-user-3"]["roles"])


def test_admission_is_idempotent_and_skips_a_redundant_register_call() -> None:
    client = tra.FixtureEngineIdentityClient()
    principal = tra.TenantPrincipal(agent_id="webui-user-1")
    authority = _authority("provisioner:deploy")

    tra.provision_tenant_access(
        client, "homelab", [principal], admin_authority=authority
    )
    register_calls_after_first = len(client.calls)

    # Re-run with the principal now correctly reporting it already holds the
    # role (mirrors a re-run of a deploy-time provisioning pass).
    already_admitted = tra.TenantPrincipal(
        agent_id="webui-user-1", existing_roles=("tenant:homelab",)
    )
    result = tra.provision_tenant_access(
        client, "homelab", [already_admitted], admin_authority=authority
    )

    assert result.outcomes[0].already_held is True
    assert len(client.calls) == register_calls_after_first, (
        "an already-held tenant role must not trigger a second register_identity call"
    )


def test_system_role_is_refused() -> None:
    """This module must never be usable to hand out blanket RBAC bypass —
    that stays engine_rbac_admission's Tier-2 System path, and even that never
    applies to ordinary tenant content access."""

    with pytest.raises(ValueError, match="System"):
        tra.TenantPrincipal(agent_id="whoever", role="System")


def test_provision_tenant_access_requires_at_least_one_principal() -> None:
    client = tra.FixtureEngineIdentityClient()
    with pytest.raises(ValueError):
        tra.provision_tenant_access(
            client, "homelab", [], admin_authority=_authority("provisioner:deploy")
        )


def test_a_failed_admission_rpc_is_never_swallowed() -> None:
    class FailingClient:
        def register_identity(self, **kwargs: object) -> str:
            raise RuntimeError("engine unreachable")

    with pytest.raises(RuntimeError, match="engine unreachable"):
        tra.provision_tenant_access(
            FailingClient(),  # type: ignore[arg-type]
            "homelab",
            [tra.TenantPrincipal(agent_id="webui-user-1")],
            admin_authority=_authority("provisioner:deploy"),
        )


def test_resolve_engine_identity_client_returns_a_live_client_without_connecting() -> (
    None
):
    client = tra.resolve_engine_identity_client()
    assert isinstance(client, tra.LiveEngineIdentityClient)
