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
- A principal whose `existing_roles` is unknown (`None`, the default) is
  refused outright, and nothing is written — the actual root cause of the
  live incident (a caller omitting `existing_roles`, not a bug in the merge
  math itself).
- A principal admitting ITSELF (`agent_id == admin_authority.signer_id`,
  `agent-webui`'s `ensure_tenant_admission` shape) is skipped outright, even
  with `existing_roles` unset -- proven against the exact live incident data
  (a principal already holding `control:system` and `tenant:homelab` keeps
  both after a self-admission pass).
"""

from __future__ import annotations

import pytest

from agent_utilities.security import tenant_rbac_admission as tra


def _authority(agent_id: str) -> tra.AdmissionAuthority:
    return tra.AdmissionAuthority(
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
    principal = tra.TenantPrincipal(
        agent_id="webui-user-1", role="Agent", existing_roles=()
    )

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
        tra.TenantPrincipal(agent_id="webui-user-1", existing_roles=()),
        tra.TenantPrincipal(agent_id="webui-user-2", existing_roles=()),
        tra.TenantPrincipal(agent_id="webui-user-3", existing_roles=()),
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
    principal = tra.TenantPrincipal(agent_id="webui-user-1", existing_roles=())
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


def test_admitting_a_principal_with_unknown_existing_roles_fails_loudly() -> None:
    """The regression this module's ``existing_roles`` field ALREADY protected
    against on paper but not in practice: a caller (e.g. ``agent-webui``'s
    ``ensure_tenant_admission``) that omits ``existing_roles`` entirely used
    to fall through to an empty-tuple default and silently register only the
    tenant role — dropping whatever else the principal held (this is the
    live incident: it dropped ``control:system`` off graph-os's own
    principal). It must now fail loudly and write nothing instead."""

    client = tra.FixtureEngineIdentityClient()
    principal = tra.TenantPrincipal(agent_id="webui-user-1")  # existing_roles unset

    assert principal.existing_roles is None

    with pytest.raises(tra.TenantAdmissionError, match="existing_roles is unknown"):
        tra.provision_tenant_access(
            client,
            "homelab",
            [principal],
            admin_authority=_authority("provisioner:deploy"),
        )

    assert client.calls == [], (
        "an unknown prior role set must never reach register_identity — "
        "fail closed, never write a possibly-reduced set"
    )


def test_self_admission_is_skipped_and_never_drops_the_admitting_principals_own_roles() -> (
    None
):
    """The exact incident, reproduced and proven fixed: graph-os's own
    principal (`5102c7f9-...` in the live incident) already carries BOTH
    `control:system` (which itself carries `security:admin`) and
    `tenant:homelab`. `agent-webui`'s `ensure_tenant_admission` then admits
    that SAME principal into its own tenant, signing as itself, exactly the
    way `TenantPrincipal(agent_id=agent_id)` is constructed in production —
    no `existing_roles` supplied. Before this fix that silently re-registered
    `roles=['tenant:homelab']` only, dropping `control:system` and bricking
    every future admin-gated admission call. Now: no exception, no
    `register_identity` call, and the principal's pre-existing roles
    (established by a wholly separate admission pass this module has no way
    to read back) are left completely untouched."""

    client = tra.FixtureEngineIdentityClient()
    principal_id = "5102c7f9-f264-4732-932f-f49b1bebce09"
    # Simulates system_rbac_admission having already granted control:system
    # (and some earlier pass having granted tenant:homelab) directly against
    # the engine -- this module never wrote it and cannot read it back.
    client.identities[principal_id] = {
        "role": "Agent",
        "teams": [],
        "roles": ["control:system", "tenant:homelab"],
    }
    # The engine only accepts signer == the admitted principal's own key, so
    # a principal admitting itself signs as itself -- exactly the shape
    # `resolve_admission_authority()` produces for `ensure_tenant_admission`.
    authority = _authority(principal_id)

    result = tra.provision_tenant_access(
        client,
        "homelab",
        [tra.TenantPrincipal(agent_id=principal_id)],  # no existing_roles
        admin_authority=authority,
    )

    assert result.all_admitted is True
    [outcome] = result.outcomes
    assert outcome.already_held is True
    assert client.calls == [], "self-admission must never call register_identity"
    # The whole point: control:system (and security:admin with it) survives.
    assert client.identities[principal_id]["roles"] == [
        "control:system",
        "tenant:homelab",
    ]


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
            [tra.TenantPrincipal(agent_id="webui-user-1", existing_roles=())],
            admin_authority=_authority("provisioner:deploy"),
        )


def test_resolve_engine_identity_client_returns_a_live_client_without_connecting() -> (
    None
):
    client = tra.resolve_engine_identity_client()
    assert isinstance(client, tra.LiveEngineIdentityClient)
