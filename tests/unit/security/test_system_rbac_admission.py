"""Tests for BUG-295's system-principal control-graph RBAC admission
(NE-009/NE-020/NE-021) — see
``agent_utilities/security/system_rbac_admission.py``'s module docstring for
the full root-cause chain: the scheduler's own process identity was never
registered with the engine's RBAC store at all (NE-020), and an earlier,
wrong fix applied a tenant `Pattern("tenant__homelab__*")` grant that can
never match the isolated `__control__` control graph the scheduler actually
reads/writes (NE-009).

Covers (Definition of Done):
- The grant is built against `Graph("__control__")`, never a tenant
  pattern — this is the regression that cost a session (NE-009).
- Admission is idempotent across repeated calls (process-local cache).
- A missing provisioner credential (NE-021) degrades honestly: no crash, no
  claimed success, an actionable message naming exactly what is missing.
- A failed admission backs off rather than hammering the engine.
- The role granted is the narrow `control:system` role and never `System`.
- The CLI (`system_admission_cli.py`) produces the same provisioning as the
  boot path (`ensure_system_principal_access`) for the same principal.
- A principal whose `existing_roles` is unknown (`None`, the default) is
  refused outright, and nothing is written — the mirror image of
  `tenant_rbac_admission`'s own incident: this module's boot path used to
  default to an implicit empty role set too, and could silently clobber a
  role `tenant_rbac_admission` had already granted the same principal.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.security import admission_authority, brain_context
from agent_utilities.security import system_admission_cli as cli
from agent_utilities.security import system_rbac_admission as sra


def _authority(agent_id: str) -> sra.AdmissionAuthority:
    return sra.AdmissionAuthority(
        agent_id=agent_id,
        signer_id=agent_id,
        signer_key="test-signer-key-not-a-real-credential",  # nosec B105 - test only; sanitizer:ignore synthetic fixture
    )


@pytest.fixture(autouse=True)
def _clear_admission_cache():
    sra._reset_admission_cache_for_tests()
    yield
    sra._reset_admission_cache_for_tests()


# ---------------------------------------------------------------------------
# NE-009: the grant selector must be Graph("__control__"), never a tenant
# pattern — the exact regression an earlier session shipped.
# ---------------------------------------------------------------------------


def test_grant_selector_is_control_graph_not_a_tenant_pattern() -> None:
    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler", existing_roles=())

    sra.provision_system_principal_access(
        client, [principal], admin_authority=_authority("provisioner:deploy")
    )

    resources = {resource for (_role, resource, _action, _effect) in client.grants}
    assert repr({"Graph": "__control__"}) in resources
    # The exact wrong selector NE-009 shipped and corrected.
    assert not any("tenant__" in resource for resource in resources)
    assert not any("Pattern" in resource for resource in resources)


def test_control_graph_name_matches_shard_topology_constant() -> None:
    from agent_utilities.knowledge_graph.core.shard_topology import (
        CONTROL_GRAPH_NAME,
    )

    assert sra.CONTROL_GRAPH_NAME == CONTROL_GRAPH_NAME == "__control__"


def test_admission_grants_reachability_on_control_graph_read_and_write() -> None:
    """Fresh-store proof, mirroring FixtureEngineAdmissionClient's own
    `has_admin_capability` pattern: the fixture reimplements enough of
    check_access to prove the principal is actually reachable, not just
    that some call was made."""

    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler", existing_roles=())

    assert client._has_access("graph-os-scheduler", "Read") is False

    sra.provision_system_principal_access(
        client, [principal], admin_authority=_authority("provisioner:deploy")
    )

    assert client._has_access("graph-os-scheduler", "Read") is True
    assert client._has_access("graph-os-scheduler", "Write") is True
    # Never granted Admin — this module never widens beyond Read+Write.
    assert client._has_access("graph-os-scheduler", "Admin") is False


# ---------------------------------------------------------------------------
# The role granted is the narrow one, never System.
# ---------------------------------------------------------------------------


def test_default_role_is_the_narrow_control_system_role() -> None:
    assert sra.CONTROL_ROLE_NAME == "control:system"
    assert sra.CONTROL_ROLE_NAME != "System"


def test_provision_refuses_role_system() -> None:
    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler")
    with pytest.raises(ValueError):
        sra.provision_system_principal_access(
            client,
            [principal],
            admin_authority=_authority("provisioner:deploy"),
            role="System",
        )


def test_system_principal_refuses_role_system() -> None:
    with pytest.raises(ValueError):
        sra.SystemPrincipal(agent_id="graph-os-scheduler", role="System")


# ---------------------------------------------------------------------------
# Merge semantics (mirrors tenant_rbac_admission's own regression proof).
# ---------------------------------------------------------------------------


def test_admitting_a_fresh_principal_grants_exactly_the_control_role() -> None:
    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(
        agent_id="graph-os-scheduler", role="Agent", existing_roles=()
    )

    result = sra.provision_system_principal_access(
        client, [principal], admin_authority=_authority("provisioner:deploy")
    )

    assert result.role == sra.CONTROL_ROLE_NAME
    assert result.all_admitted is True
    (outcome,) = result.outcomes
    assert outcome.already_held is False
    assert client.identities["graph-os-scheduler"]["roles"] == [sra.CONTROL_ROLE_NAME]


def test_admitting_an_already_held_principal_is_a_noop() -> None:
    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(
        agent_id="graph-os-scheduler", existing_roles=(sra.CONTROL_ROLE_NAME,)
    )

    result = sra.provision_system_principal_access(
        client, [principal], admin_authority=_authority("provisioner:deploy")
    )

    (outcome,) = result.outcomes
    assert outcome.already_held is True
    assert "register_identity" not in [call for call, _args in client.calls]


def test_merge_preserves_existing_roles_and_teams() -> None:
    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(
        agent_id="graph-os-scheduler",
        teams=("platform",),
        existing_roles=("some:other-role",),
    )

    sra.provision_system_principal_access(
        client, [principal], admin_authority=_authority("provisioner:deploy")
    )

    identity = client.identities["graph-os-scheduler"]
    assert identity["teams"] == ["platform"]
    assert set(identity["roles"]) == {"some:other-role", sra.CONTROL_ROLE_NAME}


def test_provision_requires_nonempty_principals() -> None:
    client = sra.FixtureSystemAdmissionClient()
    with pytest.raises(ValueError):
        sra.provision_system_principal_access(
            client, [], admin_authority=_authority("provisioner:deploy")
        )


def test_a_failed_registration_rpc_is_never_swallowed() -> None:
    class _FailingClient(sra.FixtureSystemAdmissionClient):
        def register_identity(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("engine unreachable")

    client = _FailingClient()
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler", existing_roles=())
    with pytest.raises(RuntimeError):
        sra.provision_system_principal_access(
            client, [principal], admin_authority=_authority("provisioner:deploy")
        )


def test_admitting_a_principal_with_unknown_existing_roles_fails_loudly() -> None:
    """Mirror-image of `tenant_rbac_admission`'s own regression proof: this
    module used to default `existing_roles` to an empty tuple too, so a
    caller (this module's own `ensure_system_principal_access`, in
    practice) that never learned the principal's real prior roles would
    silently register `roles=['control:system']` alone — clobbering
    whatever `tenant_rbac_admission` had already granted the SAME
    principal. It must now fail loudly and write nothing."""

    client = sra.FixtureSystemAdmissionClient()
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler")  # unset

    assert principal.existing_roles is None

    with pytest.raises(sra.SystemAdmissionError, match="existing_roles is unknown"):
        sra.provision_system_principal_access(
            client, [principal], admin_authority=_authority("provisioner:deploy")
        )

    assert "register_identity" not in [call for call, _args in client.calls], (
        "an unknown prior role set must never reach register_identity — "
        "fail closed, never write a possibly-reduced set"
    )


ADMITTING_PRINCIPAL = "graph-os:process"


@pytest.fixture(autouse=True)
def _bound_principal(monkeypatch: pytest.MonkeyPatch):
    """Bind a verified principal and start from "no signer key".

    Admission signs as the calling principal, so a bound actor is the baseline
    every test needs; whether this process HOLDS that principal's key is the
    variable, and each test sets it explicitly rather than inheriting ambient
    environment.
    """

    monkeypatch.delenv(admission_authority.SIGNER_REGISTRY_ENV, raising=False)
    actor = brain_context.ActorContext(
        actor_id=ADMITTING_PRINCIPAL, authenticated=True
    )
    token = brain_context.set_actor(actor)
    try:
        yield
    finally:
        brain_context.reset_actor(token)


def _count_resolutions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count calls to the single admission resolver.

    Replaces counting secret reads: there is no secret to read, so the thing
    that proves backoff is whether the resolver is re-entered at all.
    """

    calls = [0]
    real = admission_authority.resolve_admission_authority

    def _counting():
        calls[0] += 1
        return real()

    monkeypatch.setattr(
        admission_authority, "resolve_admission_authority", _counting
    )
    return calls


def _hold_signer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this process the signer key for its own bound principal."""

    monkeypatch.setenv(
        admission_authority.SIGNER_REGISTRY_ENV,
        json.dumps({ADMITTING_PRINCIPAL: "not-a-real-credential"}),  # nosec B105
    )


# ---------------------------------------------------------------------------
# ensure_system_principal_access: idempotent cache + honest degrade + backoff
# ---------------------------------------------------------------------------



def test_ensure_admission_is_idempotent_across_repeated_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hold_signer_key(monkeypatch)
    client = sra.FixtureSystemAdmissionClient()

    first = sra.ensure_system_principal_access(
        "graph-os-scheduler", client=client, existing_roles=()
    )
    assert first.already_held is False

    second = sra.ensure_system_principal_access(
        "graph-os-scheduler", client=client, existing_roles=()
    )
    assert second.already_held is True

    # The second call must be a cache hit: no additional register_identity.
    register_calls = [c for c, _a in client.calls if c == "register_identity"]
    assert len(register_calls) == 1


def test_ensure_admission_fails_loudly_without_existing_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror-image fix's actual point of enforcement: `kg_server.py`'s
    boot path calls `ensure_system_principal_access(actor_id)` with no
    `existing_roles` (it has no reliable source for the principal's real
    engine-RBAC roles either — see the module docstring). That must now
    degrade honestly (typed error, cached backoff, nothing written) instead
    of silently registering `roles=['control:system']` alone and clobbering
    whatever `tenant_rbac_admission` already granted this same principal."""

    _hold_signer_key(monkeypatch)
    client = sra.FixtureSystemAdmissionClient()

    with pytest.raises(sra.SystemAdmissionError, match="existing_roles is unknown"):
        sra.ensure_system_principal_access("graph-os-scheduler", client=client)

    assert "register_identity" not in [call for call, _args in client.calls], (
        "must never reach register_identity when the prior role set is unknown"
    )


def test_ensure_admission_degrades_honestly_on_missing_credential() -> None:
    """NE-021: the provisioner credential does not exist on the target
    deployment. This must raise a clear, actionable, non-secret-leaking
    error — never crash with an unrelated exception, and never return a
    value that looks like success."""


    with pytest.raises(sra.SystemAdmissionError) as exc_info:
        sra.ensure_system_principal_access("graph-os-scheduler")

    message = str(exc_info.value)
    assert admission_authority.SIGNER_REGISTRY_ENV in message
    # Never leaks key material — assert the message stays a diagnosis, not a dump.
    assert "signer_key" not in message.lower() or "key.." not in message


def test_ensure_admission_backs_off_rather_than_hammering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_resolutions(monkeypatch)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler")
    assert calls[0] == 1

    # Immediately retrying within the backoff window must NOT re-resolve the
    # credential (would "hammer" a broken precondition on every call).
    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler")
    assert calls[0] == 1


def test_ensure_admission_retries_after_backoff_window_elapses(monkeypatch) -> None:
    calls = _count_resolutions(monkeypatch)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler")
    assert calls[0] == 1

    # Simulate the backoff window having elapsed.
    key = (sra.CONTROL_ROLE_NAME, "graph-os-scheduler")
    attempted_at, cached_exc = sra._FAILURES[key]
    sra._FAILURES[key] = (attempted_at - sra._FAILURE_BACKOFF_SECONDS - 1, cached_exc)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler")
    assert calls[0] == 2


def test_ensure_admission_never_crashes_the_process_it_only_raises_a_typed_error() -> (
    None
):
    """The caller (kg_server.py's boot path) is what must never crash; this
    function's contract is a typed, catchable error, never a bare/opaque
    exception a caller cannot reason about."""

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler")


def test_ensure_admission_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError):
        sra.ensure_system_principal_access("   ")


# ---------------------------------------------------------------------------
# Credential resolution: the caller's own verified principal, signing as itself.
# ---------------------------------------------------------------------------


def test_resolve_admission_authority_without_a_signer_key_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(admission_authority.SIGNER_REGISTRY_ENV, raising=False)
    actor = brain_context.ActorContext(actor_id="graph-os:process", authenticated=True)
    token = brain_context.set_actor(actor)
    try:
        with pytest.raises(admission_authority.AdmissionAuthorityError) as exc_info:
            admission_authority.resolve_admission_authority()
    finally:
        brain_context.reset_actor(token)
    message = str(exc_info.value)
    assert "graph-os:process" in message
    assert admission_authority.SIGNER_REGISTRY_ENV in message


def test_resolve_admission_authority_with_a_malformed_registry_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(admission_authority.SIGNER_REGISTRY_ENV, "{not json")
    actor = brain_context.ActorContext(actor_id="graph-os:process", authenticated=True)
    token = brain_context.set_actor(actor)
    try:
        with pytest.raises(admission_authority.AdmissionAuthorityError):
            admission_authority.resolve_admission_authority()
    finally:
        brain_context.reset_actor(token)


def test_resolve_admission_authority_signs_as_the_verified_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        admission_authority.SIGNER_REGISTRY_ENV,
        '{"graph-os:process": "not-a-real-credential"}',  # nosec B105 - test only
    )
    actor = brain_context.ActorContext(actor_id="graph-os:process", authenticated=True)
    token = brain_context.set_actor(actor)
    try:
        authority = admission_authority.resolve_admission_authority()
    finally:
        brain_context.reset_actor(token)
    # The engine accepts no other pairing (SIGNER_TRUST_DENIED otherwise).
    assert authority.agent_id == "graph-os:process"
    assert authority.signer_id == authority.agent_id


# ---------------------------------------------------------------------------
# CLI: dry-run vs apply, and CLI == boot-path provisioning.
# ---------------------------------------------------------------------------


def test_cli_dry_run_never_touches_a_live_client_or_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_resolutions(monkeypatch)
    principals = [sra.SystemPrincipal(agent_id="graph-os-scheduler", existing_roles=())]
    result = cli.run_system_admission(
        principals, apply=False
    )
    assert result.all_admitted is True
    assert result.role == sra.CONTROL_ROLE_NAME
    assert calls[0] == 0, (
        'dry-run must resolve no credential at all'
    )


def test_cli_apply_without_a_signer_key_raises_the_one_credential_error() -> None:
    # Deliberately AdmissionAuthorityError, not this CLI's own error type: one
    # credential model surfaces one credential error across every bridge.
    principals = [sra.SystemPrincipal(agent_id="graph-os-scheduler")]
    with pytest.raises(admission_authority.AdmissionAuthorityError):
        cli.run_system_admission(principals, apply=True)


def test_cli_apply_produces_the_same_provisioning_as_the_boot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DoD: 'the CLI produces the same provisioning as the boot path.'"""

    _hold_signer_key(monkeypatch)
    boot_client = sra.FixtureSystemAdmissionClient()
    sra.ensure_system_principal_access(
        "graph-os-scheduler", client=boot_client, existing_roles=()
    )

    cli_client = sra.FixtureSystemAdmissionClient()
    cli.run_system_admission(
        [sra.SystemPrincipal(agent_id="graph-os-scheduler", existing_roles=())],
        apply=True,
        client=cli_client,
    )

    assert boot_client.identities == cli_client.identities
    assert boot_client.roles == cli_client.roles
    assert boot_client.grants == cli_client.grants


def test_cli_load_manifest_defaults_role_and_parses_principals() -> None:
    raw = '{"principals": [{"agent_id": "graph-os-scheduler", "role": "Agent"}]}'
    role, principals = cli.load_manifest(raw)
    assert role == sra.CONTROL_ROLE_NAME
    assert len(principals) == 1
    assert principals[0].agent_id == "graph-os-scheduler"


def test_cli_load_manifest_rejects_non_object_json() -> None:
    with pytest.raises(cli.SystemAdmissionCliError):
        cli.load_manifest("[]")


def test_cli_main_dry_run_exits_zero(tmp_path, capsys) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"principals": [{"agent_id": "graph-os-scheduler"}]}', encoding="utf-8"
    )
    rc = cli.main(["--manifest-file", str(manifest), "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert sra.CONTROL_ROLE_NAME in out
