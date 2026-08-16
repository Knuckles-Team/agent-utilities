"""Tests for the Tier-2 engine RBAC admission bridge (GOC-62 D3(a), closes the
missing half of BUG-030/BUG-038: "a fresh-store deploy would fail Tier 2 with
ACCESS_DENIED because the engine-side admission half of provisioning is still
missing").

Covers:
- A fresh store (no admins, no RBAC state) starts with EVERY Tier-2 action
  unreachable for a service that only holds a Tier-1 Keycloak-shaped grant —
  reproducing the defect BEFORE admission, using the exact
  ``IsolationLayer::has_admin_capability`` semantics the fixture mirrors.
- Running :func:`provision_tier2_admission` against that fresh store makes every
  manifested service's Tier-2 action reachable — the fresh-store proof.
- The pass is idempotent: running it again (bootstrap already consumed) does not
  raise and does not re-admit incorrectly.
- All 5 Tier-2 authz_action prefixes (DEC-015 §2.3 / GOC-62 §2.1) are expressible
  in the manifest.
- Known-bad-input proofs: a non-Tier-2 action is rejected by the manifest entry
  itself; a failed admission RPC is never swallowed (propagates, does not mark
  the entry admitted).
"""

from __future__ import annotations

import pytest

from agent_utilities.security import engine_rbac_admission as era


def _authority(
    agent_id: str, *, signer_id: str | None = None
) -> era.AdmissionAuthority:
    return era.AdmissionAuthority(
        agent_id=agent_id,
        signer_id=signer_id or agent_id,
        signer_key="test-signer-key-not-a-real-credential",  # nosec B105 - test only
    )


# ---------------------------------------------------------------------------
# The fresh-store defect, reproduced, then closed
# ---------------------------------------------------------------------------


def test_fresh_store_starts_with_every_tier2_action_unreachable() -> None:
    """BUG-030/BUG-038's exact claim: on a pristine store, a service holding only
    a Tier-1 Keycloak scope (e.g. `admin:cluster-read`) cannot satisfy
    `has_admin_capability` — no Keycloak grant, however broad, reaches Tier 2."""

    client = era.FixtureEngineAdmissionClient()
    assert client.identity_bootstrap_pending() is True
    # The webui/AU service account exists in Keycloak's Tier-1 world but has
    # never been engine-admitted — exactly BUG-038's "fresh deploy" state.
    assert client.has_admin_capability("service:webui") is False


def test_provision_tier2_admission_closes_the_gap_on_a_fresh_store() -> None:
    client = era.FixtureEngineAdmissionClient()
    bootstrap = _authority("provisioner:deploy")
    manifest = [
        era.ServiceAdmissionEntry(
            agent_id="service:webui",
            tier2_actions=("admin:cluster-read",),
            grant_mode="system_role",
        ),
    ]

    result = era.provision_tier2_admission(
        client,
        manifest,
        admin_authority=bootstrap,
        bootstrap_authority=bootstrap,
    )

    assert result.bootstrap_attempted is True
    assert result.bootstrap_succeeded is True
    assert result.bootstrap_already_consumed is False
    assert result.all_admitted is True

    # The fresh-store proof: PlacementRoute's gate (admin:cluster-read) is now
    # reachable for the service that only ever held a Keycloak Tier-1 scope.
    assert client.has_admin_capability("service:webui") is True


def test_admission_is_idempotent_on_a_non_fresh_store() -> None:
    """A real deploy runs this pass on EVERY deploy, not just the first. The
    second run must not fail just because bootstrap is already consumed — that
    is the expected, idempotent steady state, not an error."""

    client = era.FixtureEngineAdmissionClient()
    bootstrap = _authority("provisioner:deploy")
    manifest = [
        era.ServiceAdmissionEntry(
            agent_id="service:webui",
            tier2_actions=("admin:cluster-read",),
        ),
    ]

    first = era.provision_tier2_admission(
        client, manifest, admin_authority=bootstrap, bootstrap_authority=bootstrap
    )
    assert first.bootstrap_succeeded is True

    second = era.provision_tier2_admission(
        client, manifest, admin_authority=bootstrap, bootstrap_authority=bootstrap
    )
    assert second.bootstrap_attempted is True
    assert second.bootstrap_succeeded is False
    assert second.bootstrap_already_consumed is True
    assert second.all_admitted is True
    assert client.has_admin_capability("service:webui") is True


def test_admission_via_narrow_admin_grant_not_full_system_role() -> None:
    """A service that only needs, e.g., admin:backup should not necessarily be
    handed unconditional System — the narrower `admin_grant` path (RBAC Admin
    grant on Graph("__admin__")) must also reach has_admin_capability."""

    client = era.FixtureEngineAdmissionClient()
    bootstrap = _authority("provisioner:deploy")
    manifest = [
        era.ServiceAdmissionEntry(
            agent_id="service:backup-runner",
            tier2_actions=("admin:backup",),
            grant_mode="admin_grant",
            role="owner:backup-runner",
        ),
    ]

    result = era.provision_tier2_admission(
        client, manifest, admin_authority=bootstrap, bootstrap_authority=bootstrap
    )
    assert result.all_admitted is True
    # The grant is bound to the ROLE, not the agent_id directly — mirror that in
    # the fixture by registering the agent under that role first (a real
    # deployment's manifest/engine-side role assignment does this at admission
    # time via register_identity with `roles=[role]`; this fixture test only
    # needs to prove the ADMIN GRANT ITSELF is reachable via has_admin_capability
    # once an agent holds the role).
    client.agents["service:backup-runner"] = "owner:backup-runner"
    assert client.has_admin_capability("service:backup-runner") is True


@pytest.mark.parametrize("action", list(era.TIER2_ACTION_PREFIXES))
def test_manifest_can_express_every_tier2_action_class(action: str) -> None:
    """DEC-015 §2.3 names exactly 5 Tier-2 authz_action strings covering 20
    Method variants. Every one must be constructible in this manifest — none
    left unreachable, which is this record's headline claim."""

    entry = era.ServiceAdmissionEntry(
        agent_id=f"service:{action.replace(':', '-')}",
        tier2_actions=(action,),
    )
    assert entry.tier2_actions == (action,)


# ---------------------------------------------------------------------------
# Known-bad-input proofs (hard rule: every control ships with one)
# ---------------------------------------------------------------------------


def test_manifest_entry_rejects_a_non_tier2_action() -> None:
    """A Tier-1-only action (e.g. plain `kg:write`) does not belong in this
    manifest — it is already reachable via Keycloak alone and admitting it here
    would be silent scope creep of what this bridge grants."""

    with pytest.raises(ValueError, match="not a Tier-2 authz_action"):
        era.ServiceAdmissionEntry(agent_id="service:x", tier2_actions=("kg:write",))


def test_manifest_entry_rejects_empty_tier2_actions() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        era.ServiceAdmissionEntry(agent_id="service:x", tier2_actions=())


def test_admin_grant_without_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an explicit role name"):
        era.ServiceAdmissionEntry(
            agent_id="service:x",
            tier2_actions=("admin:backup",),
            grant_mode="admin_grant",
        )


def test_a_failed_admission_rpc_is_never_swallowed() -> None:
    """Restated from this standard's §6 (applied to provisioning instead of a
    graph write): a failed admission must fail the pass, never leave a service
    silently under-admitted. Prove it with a genuinely bad admin authority
    (never bootstrapped, so it cannot register anyone)."""

    client = era.FixtureEngineAdmissionClient()
    unadmitted = _authority("nobody:not-an-admin")
    manifest = [
        era.ServiceAdmissionEntry(
            agent_id="service:webui", tier2_actions=("admin:cluster-read",)
        ),
    ]

    with pytest.raises(era.EngineAdmissionError, match="lacks admin capability"):
        era.provision_tier2_admission(
            client, manifest, admin_authority=unadmitted, bootstrap_authority=None
        )

    # The failed attempt must not have silently admitted the service.
    assert client.has_admin_capability("service:webui") is False


def test_bootstrap_requires_matching_signer_and_agent_id() -> None:
    """Mirrors isolation.rs's bootstrap gate: only an identity bootstrapping
    ITSELF (agent_id == signer_id, security:bootstrap-only authority) may
    consume first-run authority — never a third party bootstrapping on someone
    else's behalf."""

    client = era.FixtureEngineAdmissionClient()
    mismatched = era.AdmissionAuthority(
        agent_id="provisioner:deploy",
        signer_id="someone:else",
        signer_key="k",  # nosec B105 - test only
    )
    with pytest.raises(era.EngineAdmissionError, match="matching explicit identities"):
        client.bootstrap_system_identity(
            agent_id=mismatched.agent_id,
            signer_id=mismatched.signer_id,
            signer_key=mismatched.signer_key,
        )


def test_admission_authority_never_reprs_its_signer_key() -> None:
    """A secret handled by this module must never be printable into a log/trace
    by accident (this repo's own credential-handling doctrine)."""

    authority = _authority("service:x")
    assert "test-signer-key-not-a-real-credential" not in repr(authority)
    assert "redacted" in repr(authority)
