"""Tests for the tenant-access deployment-tooling bridge (Wire-First closure for
``tenant_rbac_admission.provision_tenant_access``, mirroring
``test_tier2_admission_cli.py``'s proof shape for the admin-action sibling).

These tests prove the REAL code path — credential resolution ->
``provision_tenant_access`` -> engine client — end to end, using an injected
:class:`FixtureEngineIdentityClient`. They never construct a
``LiveEngineIdentityClient`` and never touch a live engine.

Credential resolution no longer reads a secrets backend. There is ONE admission
credential — the caller's own verified principal, signing as itself — because
the engine's ``verify_register_identity_signature`` requires
``signer == context.principal()`` and rejects anything else with
``SIGNER_TRUST_DENIED``. So these tests bind a verified actor and provide that
principal's key through the same ``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`` registry
the engine itself reads.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from agent_utilities.security import admission_authority, brain_context
from agent_utilities.security import tenant_admission_cli as cli
from agent_utilities.security import tenant_rbac_admission as tra

ADMITTING_PRINCIPAL = "graph-os:process"


def _principal_manifest() -> list[tra.TenantPrincipal]:
    # existing_roles=() confirms "known fresh" -- these tests exercise the
    # provisioning round trip, not the unknown-existing_roles fail-loud path
    # (see test_tenant_rbac_admission.py for that).
    return [
        tra.TenantPrincipal(agent_id="webui-user-1", role="Agent", existing_roles=())
    ]


@contextlib.contextmanager
def _verified_principal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    principal: str = ADMITTING_PRINCIPAL,
    registry: dict[str, str] | str | None = "default",
):
    """Bind a verified actor and this process's signer registry.

    This is the whole credential model: an actor to be, and a key for being it.
    ``registry=None`` omits the registry entirely (the "this process holds no
    signer key" case); passing a raw ``str`` injects a malformed one.
    """

    if registry == "default":
        registry = {principal: "not-a-real-credential"}  # nosec B105 - test only
    if registry is None:
        monkeypatch.delenv(admission_authority.SIGNER_REGISTRY_ENV, raising=False)
    else:
        monkeypatch.setenv(
            admission_authority.SIGNER_REGISTRY_ENV,
            registry if isinstance(registry, str) else json.dumps(registry),
        )
    actor = brain_context.ActorContext(actor_id=principal, authenticated=True)
    token = brain_context.set_actor(actor)
    try:
        yield actor
    finally:
        brain_context.reset_actor(token)


# ---------------------------------------------------------------------------
# dry-run (apply=False): NEVER touches secrets or a live client
# ---------------------------------------------------------------------------


def test_dry_run_needs_no_credential_at_all_and_reports_a_real_preview() -> None:
    # No bound actor, no signer registry: the dry-run path must still produce a
    # real preview, proving it resolves no credential of any kind.
    result = cli.run_tenant_admission("homelab", _principal_manifest(), apply=False)
    assert result.all_admitted is True
    assert result.tenant_slug == "homelab"
    assert result.role == "tenant:homelab"


# ---------------------------------------------------------------------------
# apply=True against an injected fixture client -- proves the REAL code path
# end-to-end without ever constructing a LiveEngineIdentityClient.
# ---------------------------------------------------------------------------


def test_apply_signs_as_the_verified_principal_and_admits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = tra.FixtureEngineIdentityClient()
    with _verified_principal(monkeypatch):
        result = cli.run_tenant_admission(
            "homelab", _principal_manifest(), apply=True, client=client
        )

    assert result.all_admitted is True
    assert client.identities["webui-user-1"]["roles"] == ["tenant:homelab"]


def test_the_signer_sent_to_the_engine_is_the_calling_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine refuses ``signer != principal`` with SIGNER_TRUST_DENIED, so
    prove the value actually put on the wire is the admitting principal — not
    the subject being admitted, and not some separately-provisioned identity."""

    seen: dict[str, object] = {}

    class _CapturingClient:
        def register_identity(self, **kwargs: object) -> str:
            seen.update(kwargs)
            return "ok"

    with _verified_principal(monkeypatch):
        cli.run_tenant_admission(
            "homelab",
            _principal_manifest(),
            apply=True,
            client=_CapturingClient(),  # type: ignore[arg-type]
        )

    assert seen["signer_id"] == ADMITTING_PRINCIPAL
    assert seen["agent_id"] == "webui-user-1", (
        "the SUBJECT is the principal being admitted; the SIGNER is the caller"
    )


def test_apply_is_idempotent_across_two_runs_with_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = tra.FixtureEngineIdentityClient()
    with _verified_principal(monkeypatch):
        cli.run_tenant_admission(
            "homelab", _principal_manifest(), apply=True, client=client
        )
        calls_after_first = len(client.calls)

        already_admitted = [
            tra.TenantPrincipal(
                agent_id="webui-user-1", existing_roles=("tenant:homelab",)
            )
        ]
        second = cli.run_tenant_admission(
            "homelab", already_admitted, apply=True, client=client
        )

    assert second.outcomes[0].already_held is True
    assert len(client.calls) == calls_after_first, (
        "a re-run for an already-admitted principal must not re-register it"
    )


# ---------------------------------------------------------------------------
# Known-bad-input / fail-loud proofs
# ---------------------------------------------------------------------------


def test_apply_with_no_bound_actor_fails_loud_not_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The suite binds an ambient actor for every test, so "no actor" has to be
    # produced explicitly rather than by omission.
    monkeypatch.setenv(
        admission_authority.SIGNER_REGISTRY_ENV,
        json.dumps({ADMITTING_PRINCIPAL: "not-a-real-credential"}),  # nosec B105
    )
    token = brain_context._current.set(None)
    try:
        with pytest.raises(
            admission_authority.AdmissionAuthorityError, match="bound verified actor"
        ):
            cli.run_tenant_admission(
                "homelab",
                _principal_manifest(),
                apply=True,
                client=tra.FixtureEngineIdentityClient(),
            )
    finally:
        brain_context._current.reset(token)


def test_apply_without_a_signer_key_for_this_principal_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exact live condition: agent-webui ran as its own deployment, holding a
    # verified identity but no signer entry, so it could not sign admission.
    with _verified_principal(monkeypatch, registry=None):
        with pytest.raises(
            admission_authority.AdmissionAuthorityError, match="no signer key"
        ):
            cli.run_tenant_admission(
                "homelab",
                _principal_manifest(),
                apply=True,
                client=tra.FixtureEngineIdentityClient(),
            )


def test_apply_with_a_malformed_signer_registry_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _verified_principal(monkeypatch, registry="not json"):
        with pytest.raises(
            admission_authority.AdmissionAuthorityError, match="not valid JSON"
        ):
            cli.run_tenant_admission(
                "homelab",
                _principal_manifest(),
                apply=True,
                client=tra.FixtureEngineIdentityClient(),
            )


def test_a_signer_that_is_not_the_principal_cannot_be_constructed() -> None:
    """The engine's rule, enforced locally so a mismatch fails here rather than
    as an opaque SIGNER_TRUST_DENIED after a round trip."""

    with pytest.raises(ValueError, match="signer_id must equal agent_id"):
        admission_authority.AdmissionAuthority(
            agent_id="webui-user-1",
            signer_id="provisioner:deploy",
            signer_key="not-a-real-credential",  # nosec B106 - test only
        )


def test_a_tenant_admission_error_is_never_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mirrors the shape `LiveEngineIdentityClient.register_identity` actually
    # raises on an underlying RPC failure (it wraps every exception in
    # `TenantAdmissionError` — see `tenant_rbac_admission.py`), so this proves
    # the CLI bridge's own re-raise-as-`TenantAdmissionCliError` wrapping.
    class FailingClient:
        def register_identity(self, **kwargs: object) -> str:
            raise tra.TenantAdmissionError("engine unreachable")

    with _verified_principal(monkeypatch):
        with pytest.raises(cli.TenantAdmissionCliError, match="tenant admission failed"):
            cli.run_tenant_admission(
                "homelab",
                _principal_manifest(),
                apply=True,
                client=FailingClient(),  # type: ignore[arg-type]
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
