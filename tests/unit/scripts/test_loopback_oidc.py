"""Focused tests for the lifecycle-owned HTTPS certification authority."""

from __future__ import annotations

import base64
import json
from http import HTTPStatus
from pathlib import Path

import pytest

from agent_utilities.deployment import certification_oidc as authority_module


def _decode_claims(token: str) -> dict[str, object]:
    encoded_claims = token.split(".")[1]
    padding = "=" * (-len(encoded_claims) % 4)
    decoded = base64.urlsafe_b64decode(encoded_claims + padding)
    claims = json.loads(decoded)
    assert isinstance(claims, dict)
    return claims


def _token_request(client_id: str, client_secret: str):
    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode("ascii")
    ).decode("ascii")
    return authority_module._Request(
        method="POST",
        path="/token",
        headers={
            "authorization": f"Basic {credentials}",
            "content-type": "application/x-www-form-urlencoded",
        },
        body=b"grant_type=client_credentials&scope=kg%3Aadmin",
    )


def test_default_token_lifetime_has_certification_headroom() -> None:
    assert authority_module.validated_token_ttl_seconds(None) == 300


@pytest.mark.parametrize("value", [180, 3_600])
def test_token_lifetime_accepts_bounded_endpoints(value: int) -> None:
    assert authority_module.validated_token_ttl_seconds(value) == value


def test_authority_uses_https_issuer_and_configured_lifetime() -> None:
    client_id = "certification-client"
    client_secret = "example-secret-material"
    authority = authority_module._Authority(
        client_id=client_id,
        client_secret=client_secret,
        issuer="https://127.0.0.1:1024",
        token_ttl_seconds=600,
    )
    try:
        status, response = authority.token(_token_request(client_id, client_secret))
        assert status == HTTPStatus.OK
        assert response["expires_in"] == 600
        claims = _decode_claims(response["access_token"])
        assert claims["iss"] == "https://127.0.0.1:1024"
        assert int(claims["exp"]) - int(claims["iat"]) == 600
    finally:
        authority.close()


@pytest.mark.parametrize("value", [True, 179, 3_601])
def test_token_lifetime_rejects_values_outside_bounded_contract(value: object) -> None:
    with pytest.raises(
        authority_module.CertificationAuthorityError, match="certification range"
    ):
        authority_module.validated_token_ttl_seconds(value)


def test_lifecycle_authority_verifies_tls_renews_and_removes_private_work() -> None:
    authority = authority_module.EphemeralLoopbackOidcAuthority()
    work_root: Path | None = None
    try:
        authority.start()
        work_root = authority._work_root
        assert isinstance(work_root, Path)
        assert authority.running is True
        assert authority.tls_verified is True
        assert authority.issuer.startswith("https://127.0.0.1:")
        assert authority.prove_renewable() is True
        assert authority.token_mint_count == 2
        environment = authority.child_environment(
            {
                "OAUTH2_TOKEN_TLS_PROFILE": "model-token-authority",
                "OAUTH2_TOKEN_TLS_PROFILE_REF": "env://MODEL_TOKEN_TLS_PROFILE",
            },
            model_private_hosts=["10.0.0.10"],
        )
        profile = json.loads(environment["GRAPHOS_SKILL_CERT_OIDC_TLS_PROFILE"])
        identity_oauth2 = json.loads(environment["KG_IDENTITY_OAUTH2"])
        assert profile["system_trust"] is False
        assert profile["trust_env"] is False
        assert "ca_bundle_pem" in profile
        assert "verify" not in profile
        assert "allow_insecure" not in profile
        assert identity_oauth2["tls_profile_ref"] == (
            "env://GRAPHOS_SKILL_CERT_OIDC_TLS_PROFILE"
        )
        assert environment["OIDC_TLS_PROFILE_REF"] == (
            "env://GRAPHOS_SKILL_CERT_OIDC_TLS_PROFILE"
        )
        assert environment["OAUTH2_TOKEN_TLS_PROFILE"] == "model-token-authority"
        assert environment["OAUTH2_TOKEN_TLS_PROFILE_REF"] == (
            "env://MODEL_TOKEN_TLS_PROFILE"
        )
    finally:
        authority.stop()
    assert authority.running is False
    assert work_root is not None
    assert not work_root.exists()


def test_source_wrapper_exposes_self_check_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert authority_module.main(["--self-check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert authority_module.main(["--port", "1024"]) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}
