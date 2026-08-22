"""Tests for the Tier-2 admission deployment-tooling bridge (BUG-068/BUG-038).

``engine_rbac_admission.provision_tier2_admission`` existed, fully unit-tested,
with zero live callers — the exact Wire-First trap this repo's ``AGENTS.md``
warns about. ``tier2_admission_cli.py`` is the missing bridge: it resolves the
provisioner's signer credentials from a secrets backend and gives deployment
tooling (``agent-webui``'s ``provision_identity.py``) ONE function to call.

These tests prove the REAL code path — credential resolution ->
``provision_tier2_admission`` -> engine client — end to end, using an injected
:class:`FixtureEngineAdmissionClient`. They never
construct a ``LiveEngineAdmissionClient`` and never touch a live engine or a
real secrets backend; per BUG-068's explicit instruction, no test here (or
anywhere in this change) exercises the admin RPC against a live cluster.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.security import admission_authority, brain_context
from agent_utilities.security import engine_rbac_admission as era
from agent_utilities.security import tier2_admission_cli as cli


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


def _provisioner_secret_json(agent_id: str = "graph-os:process") -> str:
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



ADMITTING_PRINCIPAL = "graph-os:process"


@pytest.fixture(autouse=True)
def _bound_principal(monkeypatch: pytest.MonkeyPatch):
    """Bind a verified principal; start from "this process holds no key"."""

    monkeypatch.delenv(admission_authority.SIGNER_REGISTRY_ENV, raising=False)
    actor = brain_context.ActorContext(
        actor_id=ADMITTING_PRINCIPAL, authenticated=True
    )
    token = brain_context.set_actor(actor)
    try:
        yield
    finally:
        brain_context.reset_actor(token)


def _hold_signer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this process the signer key for its own bound principal."""

    monkeypatch.setenv(
        admission_authority.SIGNER_REGISTRY_ENV,
        json.dumps({ADMITTING_PRINCIPAL: "not-a-real-credential"}),  # nosec B105
    )


def test_dry_run_never_resolves_secrets_and_reports_a_real_preview() -> None:
    result = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=False
    )
    assert result.all_admitted is True
    assert result.bootstrap_attempted is True
    assert result.bootstrap_succeeded is True


# ---------------------------------------------------------------------------
# apply=True against an injected fixture client -- proves the REAL code path
# end-to-end without ever constructing a LiveEngineAdmissionClient.
# ---------------------------------------------------------------------------


def test_apply_resolves_credentials_and_admits_against_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hold_signer_key(monkeypatch)
    client = era.FixtureEngineAdmissionClient()

    result = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client
    )

    assert result.all_admitted is True
    assert result.bootstrap_succeeded is True
    assert client.has_admin_capability(ADMITTING_PRINCIPAL) is True
    # The fresh-store proof, mirroring test_engine_rbac_admission.py's own
    # admin_grant case: the grant lands on the ROLE; an agent holding that
    # role then satisfies has_admin_capability.
    client.agents["service:webui"] = "webui-cluster-read"
    assert client.has_admin_capability("service:webui") is True


def test_apply_is_idempotent_across_two_runs_with_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hold_signer_key(monkeypatch)
    client = era.FixtureEngineAdmissionClient()

    first = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client
    )
    second = cli.run_tier2_admission(
        _admin_grant_manifest(), apply=True, client=client
    )

    assert first.bootstrap_succeeded is True
    assert second.bootstrap_succeeded is False
    assert second.bootstrap_already_consumed is True
    assert second.all_admitted is True


# ---------------------------------------------------------------------------
# Known-bad-input / fail-loud proofs
# ---------------------------------------------------------------------------


def test_apply_without_a_signer_key_for_this_principal_fails_loud() -> None:
    with pytest.raises(
        admission_authority.AdmissionAuthorityError, match="no signer key"
    ):
        cli.run_tier2_admission(
            _admin_grant_manifest(),
            apply=True,
            client=era.FixtureEngineAdmissionClient(),
        )


def test_apply_with_a_malformed_signer_registry_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(admission_authority.SIGNER_REGISTRY_ENV, "not json")
    with pytest.raises(admission_authority.AdmissionAuthorityError, match="not valid JSON"):
        cli.run_tier2_admission(
            _admin_grant_manifest(),
            apply=True,
            client=era.FixtureEngineAdmissionClient(),
        )




def test_an_engine_admission_error_is_never_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hold_signer_key(monkeypatch)
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

    with pytest.raises(cli.Tier2AdmissionError, match="Tier-2 admission failed"):
        cli.run_tier2_admission(
            manifest, apply=True, client=client
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
