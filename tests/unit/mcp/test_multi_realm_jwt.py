"""Tests for native multi-realm JWT trust (CONCEPT:AU-OS.identity.native-multi-realm-jwt).

A comma-separated FASTMCP_SERVER_AUTH_JWT_ISSUER + _JWKS_URI builds one JWTVerifier per
realm; a token is accepted if ANY realm validates it. This enables a zero-downtime Keycloak
realm migration (trust both realms, flip the minter, drop the old realm).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_utilities.mcp.server_factory import _configure_jwt_auth


def _args(**kw: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "token_jwks_uri": None,
        "token_issuer": None,
        "token_audience": "agent-services",
        "token_algorithm": None,
        "token_secret": None,
        "token_public_key": None,
        "required_scopes": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_single_realm_returns_plain_jwtverifier() -> None:
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = _configure_jwt_auth(
        _args(
            token_jwks_uri="http://kc.test/realms/master/protocol/openid-connect/certs",
            token_issuer="http://kc.test/realms/master",
        )
    )
    assert isinstance(verifier, JWTVerifier)


def test_multi_realm_builds_one_verifier_per_realm() -> None:
    verifier = _configure_jwt_auth(
        _args(
            token_jwks_uri=(
                "http://kc.test/realms/master/protocol/openid-connect/certs,"
                "http://kc.test/realms/homelab/protocol/openid-connect/certs"
            ),
            token_issuer="http://kc.test/realms/master,http://kc.test/realms/homelab",
        )
    )
    assert hasattr(verifier, "_verifiers")
    assert len(verifier._verifiers) == 2


def test_multi_realm_mismatched_list_lengths_exit() -> None:
    with pytest.raises(SystemExit):
        _configure_jwt_auth(
            _args(
                token_jwks_uri="http://kc.test/realms/master/protocol/openid-connect/certs",
                token_issuer="http://kc.test/realms/master,http://kc.test/realms/homelab",
            )
        )


def _multi() -> object:
    return _configure_jwt_auth(
        _args(
            token_jwks_uri=(
                "http://kc.test/realms/master/protocol/openid-connect/certs,"
                "http://kc.test/realms/homelab/protocol/openid-connect/certs"
            ),
            token_issuer="http://kc.test/realms/master,http://kc.test/realms/homelab",
        )
    )


def test_multi_realm_accepts_when_any_realm_validates() -> None:
    verifier = _multi()

    class _Fake:
        def __init__(self, result: object) -> None:
            self.result = result
            self.called = False

        async def verify_token(self, token: str) -> object:
            self.called = True
            return self.result

    rejects, accepts = _Fake(None), _Fake("ACCESS")
    verifier._verifiers = [rejects, accepts]  # type: ignore[attr-defined]
    assert asyncio.run(verifier.verify_token("tok")) == "ACCESS"  # type: ignore[attr-defined]
    assert rejects.called and accepts.called  # tried the first, then the second


def test_multi_realm_rejects_when_no_realm_validates() -> None:
    verifier = _multi()

    class _Reject:
        async def verify_token(self, token: str) -> None:
            return None

    verifier._verifiers = [_Reject(), _Reject()]  # type: ignore[attr-defined]
    assert asyncio.run(verifier.verify_token("tok")) is None  # type: ignore[attr-defined]


def test_multi_realm_survives_a_verifier_raising() -> None:
    verifier = _multi()

    class _Boom:
        async def verify_token(self, token: str) -> None:
            raise RuntimeError("jwks fetch failed")

    class _Ok:
        async def verify_token(self, token: str) -> str:
            return "ACCESS"

    verifier._verifiers = [_Boom(), _Ok()]  # type: ignore[attr-defined]
    assert asyncio.run(verifier.verify_token("tok")) == "ACCESS"  # type: ignore[attr-defined]
