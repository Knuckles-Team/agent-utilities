"""Tests for the tenant-access deployment-tooling bridge (Wire-First closure for
``tenant_rbac_admission.provision_tenant_access``, mirroring
``test_tier2_admission_cli.py``'s proof shape for the admin-action sibling).

These tests prove the REAL code path — credential resolution ->
``provision_tenant_access`` -> engine client — end to end, using an injected
:class:`FixtureEngineIdentityClient` and a fake secrets source. They never
construct a ``LiveEngineIdentityClient`` and never touch a live engine or a
real secrets backend.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.security import tenant_admission_cli as cli
from agent_utilities.security import tenant_rbac_admission as tra


class _FakeSecretsClient:
    """A minimal stand-in for ``SecretsClient`` — a plain ``get(key)`` -> str
    | None, exactly the surface :func:`resolve_provisioner_authority` uses."""

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets

    def get(self, key: str) -> str | None:
        return self._secrets.get(key)


class _ExplodingSecretsClient:
    """Fails any call — used to prove the dry-run path never touches it."""

    def get(self, key: str) -> str:
        raise AssertionError("dry-run must never resolve a real secret")


def _principal_manifest() -> list[tra.TenantPrincipal]:
    return [tra.TenantPrincipal(agent_id="webui-user-1", role="Agent")]


def _provisioner_secret_json(agent_id: str = "provisioner:deploy") -> str:
    return json.dumps(
        {
            "agent_id": agent_id,
            "signer_id": agent_id,
            "signer_key": "not-a-real-credential",  # nosec B105 - test only
        }
    )


# ---------------------------------------------------------------------------
# dry-run (apply=False): NEVER touches secrets or a live client
# ---------------------------------------------------------------------------


def test_dry_run_never_resolves_secrets_and_reports_a_real_preview() -> None:
    result = cli.run_tenant_admission(
        "homelab",
        _principal_manifest(),
        apply=False,
        secrets_client=_ExplodingSecretsClient(),
    )
    assert result.all_admitted is True
    assert result.tenant_slug == "homelab"
    assert result.role == "tenant:homelab"


# ---------------------------------------------------------------------------
# apply=True against an injected fixture client -- proves the REAL code path
# end-to-end without ever constructing a LiveEngineIdentityClient.
# ---------------------------------------------------------------------------


def test_apply_resolves_credentials_and_admits_against_injected_client() -> None:
    client = tra.FixtureEngineIdentityClient()
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json()}
    )

    result = cli.run_tenant_admission(
        "homelab",
        _principal_manifest(),
        apply=True,
        client=client,
        secrets_client=secrets,
    )

    assert result.all_admitted is True
    assert client.identities["webui-user-1"]["roles"] == ["tenant:homelab"]


def test_apply_is_idempotent_across_two_runs_with_the_same_client() -> None:
    client = tra.FixtureEngineIdentityClient()
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json()}
    )

    cli.run_tenant_admission(
        "homelab",
        _principal_manifest(),
        apply=True,
        client=client,
        secrets_client=secrets,
    )
    calls_after_first = len(client.calls)

    already_admitted = [
        tra.TenantPrincipal(agent_id="webui-user-1", existing_roles=("tenant:homelab",))
    ]
    second = cli.run_tenant_admission(
        "homelab", already_admitted, apply=True, client=client, secrets_client=secrets
    )

    assert second.outcomes[0].already_held is True
    assert len(client.calls) == calls_after_first, (
        "a re-run for an already-admitted principal must not re-register it"
    )


# ---------------------------------------------------------------------------
# Known-bad-input / fail-loud proofs
# ---------------------------------------------------------------------------


def test_apply_without_a_configured_secret_fails_loud_not_silent() -> None:
    with pytest.raises(cli.TenantAdmissionCliError, match="no provisioner credential"):
        cli.run_tenant_admission(
            "homelab",
            _principal_manifest(),
            apply=True,
            client=tra.FixtureEngineIdentityClient(),
            secrets_client=_FakeSecretsClient({}),
        )


def test_apply_with_malformed_secret_json_fails_loud() -> None:
    secrets = _FakeSecretsClient({cli.DEFAULT_PROVISIONER_SECRET_KEY: "not json"})
    with pytest.raises(cli.TenantAdmissionCliError, match="not valid JSON"):
        cli.run_tenant_admission(
            "homelab",
            _principal_manifest(),
            apply=True,
            client=tra.FixtureEngineIdentityClient(),
            secrets_client=secrets,
        )


def test_apply_with_incomplete_secret_payload_fails_loud() -> None:
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: json.dumps({"agent_id": "x"})}
    )
    with pytest.raises(cli.TenantAdmissionCliError, match="missing or has an invalid"):
        cli.run_tenant_admission(
            "homelab",
            _principal_manifest(),
            apply=True,
            client=tra.FixtureEngineIdentityClient(),
            secrets_client=secrets,
        )


def test_a_tenant_admission_error_is_never_swallowed() -> None:
    # Mirrors the shape `LiveEngineIdentityClient.register_identity` actually
    # raises on an underlying RPC failure (it wraps every exception in
    # `TenantAdmissionError` — see `tenant_rbac_admission.py`), so this proves
    # the CLI bridge's own re-raise-as-`TenantAdmissionCliError` wrapping,
    # not just bare exception propagation (already covered by
    # `test_tenant_rbac_admission.py::test_a_failed_admission_rpc_is_never_swallowed`).
    class FailingClient:
        def register_identity(self, **kwargs: object) -> str:
            raise tra.TenantAdmissionError("engine unreachable")

    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json()}
    )
    with pytest.raises(cli.TenantAdmissionCliError, match="tenant admission failed"):
        cli.run_tenant_admission(
            "homelab",
            _principal_manifest(),
            apply=True,
            client=FailingClient(),  # type: ignore[arg-type]
            secrets_client=secrets,
        )


# ---------------------------------------------------------------------------
# Manifest loading (the CLI's own parsing)
# ---------------------------------------------------------------------------


def test_load_principal_manifest_round_trips_every_field() -> None:
    raw = json.dumps(
        {
            "tenant_slug": "homelab",
            "principals": [
                {
                    "agent_id": "webui-user-1",
                    "role": "Agent",
                    "teams": ["support"],
                    "existing_roles": ["code-reader"],
                }
            ],
        }
    )
    tenant_slug, principals = cli.load_principal_manifest(raw)
    assert tenant_slug == "homelab"
    assert principals == [
        tra.TenantPrincipal(
            agent_id="webui-user-1",
            role="Agent",
            teams=("support",),
            existing_roles=("code-reader",),
        )
    ]


def test_load_principal_manifest_rejects_a_non_object_payload() -> None:
    with pytest.raises(cli.TenantAdmissionCliError, match="must be an object"):
        cli.load_principal_manifest(json.dumps(["not", "an", "object"]))


def test_load_principal_manifest_rejects_a_system_role_principal() -> None:
    raw = json.dumps(
        {
            "tenant_slug": "homelab",
            "principals": [{"agent_id": "whoever", "role": "System"}],
        }
    )
    with pytest.raises(ValueError, match="System"):
        cli.load_principal_manifest(raw)


# ---------------------------------------------------------------------------
# CLI wiring: a real entrypoint, driven end to end, dry-run only
# ---------------------------------------------------------------------------


def test_cli_main_dry_run_prints_a_preview_and_exits_zero(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tenant_slug": "homelab",
                "principals": [{"agent_id": "webui-user-1", "role": "Agent"}],
            }
        )
    )

    exit_code = cli.main(["--manifest-file", str(manifest_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "webui-user-1" in out
    assert "all_admitted=True" in out
