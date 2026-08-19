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
"""

from __future__ import annotations

import pytest

from agent_utilities.security import system_admission_cli as cli
from agent_utilities.security import system_rbac_admission as sra


def _authority(agent_id: str) -> sra.SystemAdmissionAuthority:
    return sra.SystemAdmissionAuthority(
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
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler")

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
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler")

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
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler", role="Agent")

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
    principal = sra.SystemPrincipal(agent_id="graph-os-scheduler")
    with pytest.raises(RuntimeError):
        sra.provision_system_principal_access(
            client, [principal], admin_authority=_authority("provisioner:deploy")
        )


# ---------------------------------------------------------------------------
# ensure_system_principal_access: idempotent cache + honest degrade + backoff
# ---------------------------------------------------------------------------


class _FakeSecretsClient:
    def __init__(self, value: str | None) -> None:
        self._value = value
        self.calls = 0

    def get(self, key: str) -> str | None:
        self.calls += 1
        return self._value


def test_ensure_admission_is_idempotent_across_repeated_calls() -> None:
    secrets = _FakeSecretsClient(
        '{"agent_id": "p", "signer_id": "p", "signer_key": "k"}'
    )
    client = sra.FixtureSystemAdmissionClient()

    first = sra.ensure_system_principal_access(
        "graph-os-scheduler", client=client, secrets_client=secrets
    )
    assert first.already_held is False

    second = sra.ensure_system_principal_access(
        "graph-os-scheduler", client=client, secrets_client=secrets
    )
    assert second.already_held is True

    # The second call must be a cache hit: no additional secret resolution
    # and no additional register_identity call.
    assert secrets.calls == 1
    register_calls = [c for c, _a in client.calls if c == "register_identity"]
    assert len(register_calls) == 1


def test_ensure_admission_degrades_honestly_on_missing_credential() -> None:
    """NE-021: the provisioner credential does not exist on the target
    deployment. This must raise a clear, actionable, non-secret-leaking
    error — never crash with an unrelated exception, and never return a
    value that looks like success."""

    secrets = _FakeSecretsClient(None)

    with pytest.raises(sra.SystemAdmissionError) as exc_info:
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)

    message = str(exc_info.value)
    assert "engine-admission/provisioner" in message
    # Never leaks a secret VALUE - there is none to leak here (the fixture
    # returned None), but assert the message stays a diagnosis, not a dump.
    assert "signer_key" not in message.lower() or "key.." not in message


def test_ensure_admission_backs_off_rather_than_hammering() -> None:
    secrets = _FakeSecretsClient(None)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)
    assert secrets.calls == 1

    # Immediately retrying within the backoff window must NOT re-resolve
    # the secret (would "hammer" a broken precondition on every call).
    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)
    assert secrets.calls == 1


def test_ensure_admission_retries_after_backoff_window_elapses(monkeypatch) -> None:
    secrets = _FakeSecretsClient(None)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)
    assert secrets.calls == 1

    # Simulate the backoff window having elapsed.
    key = (sra.CONTROL_ROLE_NAME, "graph-os-scheduler")
    attempted_at, cached_exc = sra._FAILURES[key]
    sra._FAILURES[key] = (attempted_at - sra._FAILURE_BACKOFF_SECONDS - 1, cached_exc)

    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)
    assert secrets.calls == 2


def test_ensure_admission_never_crashes_the_process_it_only_raises_a_typed_error() -> (
    None
):
    """The caller (kg_server.py's boot path) is what must never crash; this
    function's contract is a typed, catchable error, never a bare/opaque
    exception a caller cannot reason about."""

    secrets = _FakeSecretsClient("not valid json")
    with pytest.raises(sra.SystemAdmissionError):
        sra.ensure_system_principal_access("graph-os-scheduler", secrets_client=secrets)


def test_ensure_admission_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError):
        sra.ensure_system_principal_access("   ")


# ---------------------------------------------------------------------------
# resolve_provisioner_authority: NE-021 credential resolution.
# ---------------------------------------------------------------------------


def test_resolve_provisioner_authority_missing_key_raises_actionable_error() -> None:
    secrets = _FakeSecretsClient(None)
    with pytest.raises(sra.SystemAdmissionError) as exc_info:
        sra.resolve_provisioner_authority(secrets_client=secrets)
    assert sra.DEFAULT_PROVISIONER_SECRET_KEY in str(exc_info.value)


def test_resolve_provisioner_authority_malformed_json_raises() -> None:
    secrets = _FakeSecretsClient("{not json")
    with pytest.raises(sra.SystemAdmissionError):
        sra.resolve_provisioner_authority(secrets_client=secrets)


def test_resolve_provisioner_authority_succeeds_on_well_formed_secret() -> None:
    secrets = _FakeSecretsClient(
        '{"agent_id": "provisioner", "signer_id": "provisioner", "signer_key": "k"}'
    )
    authority = sra.resolve_provisioner_authority(secrets_client=secrets)
    assert authority.agent_id == "provisioner"
    assert authority.signer_id == "provisioner"


# ---------------------------------------------------------------------------
# CLI: dry-run vs apply, and CLI == boot-path provisioning.
# ---------------------------------------------------------------------------


def test_cli_dry_run_never_touches_a_live_client_or_secrets() -> None:
    class _ExplodingSecretsClient:
        def get(self, key: str) -> str:  # pragma: no cover - must never run
            raise AssertionError("dry-run must never resolve a real credential")

    principals = [sra.SystemPrincipal(agent_id="graph-os-scheduler")]
    result = cli.run_system_admission(
        principals, apply=False, secrets_client=_ExplodingSecretsClient()
    )
    assert result.all_admitted is True
    assert result.role == sra.CONTROL_ROLE_NAME


def test_cli_apply_without_credential_raises_cli_error() -> None:
    principals = [sra.SystemPrincipal(agent_id="graph-os-scheduler")]
    secrets = _FakeSecretsClient(None)
    with pytest.raises(cli.SystemAdmissionCliError):
        cli.run_system_admission(principals, apply=True, secrets_client=secrets)


def test_cli_apply_produces_the_same_provisioning_as_the_boot_path() -> None:
    """DoD: 'the CLI produces the same provisioning as the boot path.'"""

    secrets = _FakeSecretsClient(
        '{"agent_id": "p", "signer_id": "p", "signer_key": "k"}'
    )

    boot_client = sra.FixtureSystemAdmissionClient()
    sra.ensure_system_principal_access(
        "graph-os-scheduler", client=boot_client, secrets_client=secrets
    )

    cli_client = sra.FixtureSystemAdmissionClient()
    cli.run_system_admission(
        [sra.SystemPrincipal(agent_id="graph-os-scheduler")],
        apply=True,
        client=cli_client,
        secrets_client=_FakeSecretsClient(
            '{"agent_id": "p", "signer_id": "p", "signer_key": "k"}'
        ),
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
