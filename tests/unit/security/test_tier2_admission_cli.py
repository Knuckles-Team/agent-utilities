"""Tests for the Tier-2 admission deployment-tooling bridge (BUG-068/BUG-038).

``engine_rbac_admission.provision_tier2_admission`` existed, fully unit-tested,
with zero live callers — the exact Wire-First trap this repo's ``AGENTS.md``
warns about. ``tier2_admission_cli.py`` is the missing bridge: it resolves the
provisioner's signer credentials from a secrets backend and gives deployment
tooling (``agent-webui``'s ``provision_identity.py``) ONE function to call.

These tests prove the REAL code path — credential resolution ->
``provision_tier2_admission`` -> engine client — end to end, using an injected
:class:`FixtureEngineAdmissionClient` and a fake secrets source. They never
construct a ``LiveEngineAdmissionClient`` and never touch a live engine or a
real secrets backend; per BUG-068's explicit instruction, no test here (or
anywhere in this change) exercises the admin RPC against a live cluster.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.security import engine_rbac_admission as era
from agent_utilities.security import tier2_admission_cli as cli


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


def _admin_grant_manifest() -> list[era.ServiceAdmissionEntry]:
    """The actual production shape: the webui service admitted for
    ``admin:cluster-read`` via a narrow named role, never full System."""

    return [
        era.ServiceAdmissionEntry(
            agent_id="service:webui",
            tier2_actions=("admin:cluster-read",),
            grant_mode="admin_grant",
            role="webui-cluster-read",
        )
    ]


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
    result = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=False, secrets_client=_ExplodingSecretsClient()
    )
    assert result.all_admitted is True
    assert result.bootstrap_attempted is True
    assert result.bootstrap_succeeded is True


# ---------------------------------------------------------------------------
# apply=True against an injected fixture client -- proves the REAL code path
# end-to-end without ever constructing a LiveEngineAdmissionClient.
# ---------------------------------------------------------------------------


def test_apply_resolves_credentials_and_admits_against_injected_client() -> None:
    client = era.FixtureEngineAdmissionClient()
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json()}
    )

    result = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client, secrets_client=secrets
    )

    assert result.all_admitted is True
    assert result.bootstrap_succeeded is True
    assert client.has_admin_capability("provisioner:deploy") is True
    # The fresh-store proof, mirroring test_engine_rbac_admission.py's own
    # admin_grant case: the grant lands on the ROLE; an agent holding that
    # role then satisfies has_admin_capability.
    client.agents["service:webui"] = "webui-cluster-read"
    assert client.has_admin_capability("service:webui") is True


def test_apply_is_idempotent_across_two_runs_with_the_same_client() -> None:
    client = era.FixtureEngineAdmissionClient()
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json()}
    )

    first = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client, secrets_client=secrets
    )
    second = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client, secrets_client=secrets
    )

    assert first.bootstrap_succeeded is True
    assert second.bootstrap_succeeded is False
    assert second.bootstrap_already_consumed is True
    assert second.all_admitted is True


# ---------------------------------------------------------------------------
# Known-bad-input / fail-loud proofs
# ---------------------------------------------------------------------------


def test_apply_without_a_configured_secret_fails_loud_not_silent() -> None:
    with pytest.raises(cli.Tier2AdmissionError, match="no provisioner credential"):
        cli.run_tier2_admission(
            _admin_grant_manifest(),
            apply=True,
            client=era.FixtureEngineAdmissionClient(),
            secrets_client=_FakeSecretsClient({}),
        )


def test_apply_with_malformed_secret_json_fails_loud() -> None:
    secrets = _FakeSecretsClient({cli.DEFAULT_PROVISIONER_SECRET_KEY: "not json"})
    with pytest.raises(cli.Tier2AdmissionError, match="not valid JSON"):
        cli.run_tier2_admission(
            _admin_grant_manifest(),
            apply=True,
            client=era.FixtureEngineAdmissionClient(),
            secrets_client=secrets,
        )


def test_apply_with_incomplete_secret_payload_fails_loud() -> None:
    secrets = _FakeSecretsClient(
        {cli.DEFAULT_PROVISIONER_SECRET_KEY: json.dumps({"agent_id": "x"})}
    )
    with pytest.raises(cli.Tier2AdmissionError, match="missing or has an invalid"):
        cli.run_tier2_admission(
            _admin_grant_manifest(),
            apply=True,
            client=era.FixtureEngineAdmissionClient(),
            secrets_client=secrets,
        )


def test_an_engine_admission_error_is_never_swallowed() -> None:
    """A genuinely non-admin provisioner authority must fail the pass, not be
    reported as success — restated for the deployment-tooling bridge, mirroring
    ``engine_rbac_admission``'s own equivalent proof. Uses the default
    ``grant_mode="system_role"`` (unlike the admin_grant manifest above) because
    that is the path the fixture actually gates on ``has_admin_capability``."""

    client = era.FixtureEngineAdmissionClient()
    # Pre-consume bootstrap with an unrelated identity so the bad authority
    # below cannot bootstrap its way to admin either.
    client.bootstrap_system_identity(
        agent_id="someone:else", signer_id="someone:else", signer_key="k"
    )
    manifest = [
        era.ServiceAdmissionEntry(
            agent_id="service:webui", tier2_actions=("admin:cluster-read",)
        )
    ]
    secrets = _FakeSecretsClient(
        {
            cli.DEFAULT_PROVISIONER_SECRET_KEY: _provisioner_secret_json(
                "nobody:not-an-admin"
            )
        }
    )

    with pytest.raises(cli.Tier2AdmissionError, match="Tier-2 admission failed"):
        cli.run_tier2_admission(
            manifest, apply=True, client=client, secrets_client=secrets
        )

    # The failed attempt must not have silently admitted the service.
    assert client.has_admin_capability("service:webui") is False


# ---------------------------------------------------------------------------
# Manifest loading (the CLI's own parsing)
# ---------------------------------------------------------------------------


def test_load_manifest_round_trips_every_field() -> None:
    raw = json.dumps(
        [
            {
                "agent_id": "service:webui",
                "tier2_actions": ["admin:cluster-read"],
                "grant_mode": "admin_grant",
                "role": "webui-cluster-read",
            }
        ]
    )
    assert cli.load_manifest(raw) == _admin_grant_manifest()


def test_load_manifest_rejects_a_non_list_payload() -> None:
    with pytest.raises(cli.Tier2AdmissionError, match="must be a list"):
        cli.load_manifest(json.dumps({"agent_id": "x"}))


# ---------------------------------------------------------------------------
# CLI wiring: a real entrypoint, driven end to end, dry-run only
# ---------------------------------------------------------------------------


def test_cli_main_dry_run_prints_a_preview_and_exits_zero(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "agent_id": "service:webui",
                    "tier2_actions": ["admin:cluster-read"],
                    "grant_mode": "admin_grant",
                    "role": "webui-cluster-read",
                }
            ]
        )
    )

    exit_code = cli.main(["--manifest-file", str(manifest_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "service:webui" in out
    assert "all_admitted=True" in out
