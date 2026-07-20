"""Tests for JWT bearer authentication.

CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

Covers JWT decoding, bearer-policy enforcement, JWKS caching, CORS, and
TrustedHost configuration.
"""

import time
from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from agent_utilities.security.auth import (
    _decode_jwt,
    _fetch_jwks,
    authenticate_header_values,
    parse_bearer_authorization,
    verify_credentials,
)

# ---------------------------------------------------------------------------
# Helper: Build a config mock
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a config-like mock with sensible defaults."""
    cfg = MagicMock()
    cfg.auth_jwt_jwks_uri = overrides.get("auth_jwt_jwks_uri", None)
    cfg.auth_jwt_issuer = overrides.get(
        "auth_jwt_issuer",
        "https://identity.example.test" if cfg.auth_jwt_jwks_uri else None,
    )
    cfg.auth_jwt_audience = overrides.get(
        "auth_jwt_audience", "agent-services" if cfg.auth_jwt_jwks_uri else None
    )
    cfg.allowed_origins = overrides.get("allowed_origins", None)
    cfg.allowed_hosts = overrides.get("allowed_hosts", None)
    cfg.cors_allow_credentials = overrides.get("cors_allow_credentials", False)
    cfg.server_tls_certfile = overrides.get("server_tls_certfile", None)
    cfg.server_tls_keyfile = overrides.get("server_tls_keyfile", None)
    cfg.server_tls_terminated = overrides.get("server_tls_terminated", False)
    cfg.server_trusted_proxy_cidrs = overrides.get(
        "server_trusted_proxy_cidrs", None
    )
    return cfg


# ---------------------------------------------------------------------------
# JWT Decode Tests
# ---------------------------------------------------------------------------


class TestDecodeJWT:
    """Tests for JWT token decoding logic."""

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_valid_token(self):
        """Decoding a valid JWT with matching claims succeeds."""
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key)

        claims = _decode_jwt(
            token, jwks, issuer="https://auth.example.com", audience="my-api"
        )
        assert claims["sub"] == "user123"
        assert claims["iss"] == "https://auth.example.com"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_expired_token_raises(self):
        """Expired tokens raise HTTPException with 401."""
        from fastapi import HTTPException
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) - 3600,
            "iat": int(time.time()) - 7200,
        }
        token = joserfc_jwt.encode(header, payload, key)

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token, jwks, issuer="https://auth.example.com", audience="my-api"
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_wrong_issuer_raises(self):
        """Token with wrong issuer raises HTTPException."""
        from fastapi import HTTPException
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://wrong-issuer.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key)

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token,
                jwks,
                issuer="https://correct-issuer.com",
                audience="my-api",
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_wrong_audience_raises(self):
        """Token with wrong audience raises HTTPException."""
        from fastapi import HTTPException
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "wrong-audience",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key)

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token,
                jwks,
                issuer="https://auth.example.com",
                audience="correct-audience",
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_bad_signature_raises(self):
        """Token signed with a different key raises HTTPException."""
        from fastapi import HTTPException
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key1 = RSAKey.generate_key(2048)
        key2 = RSAKey.generate_key(2048)
        jwks = {"keys": [key2.as_dict(is_private=False)]}  # Different key

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key1)

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token, jwks, issuer="https://auth.example.com", audience="my-api"
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_no_issuer_audience_check(self):
        """Decoding without issuer/audience validation succeeds."""
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key)

        claims = _decode_jwt(token, jwks, issuer=None, audience=None)
        assert claims["sub"] == "user123"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_decode_audience_list_valid(self):
        """Token with audience as list containing expected value succeeds."""
        from joserfc import jwt as joserfc_jwt
        from joserfc.jwk import RSAKey

        key = RSAKey.generate_key(2048)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": ["my-api", "other-api"],
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = joserfc_jwt.encode(header, payload, key)

        claims = _decode_jwt(
            token, jwks, issuer="https://auth.example.com", audience="my-api"
        )
        assert claims["sub"] == "user123"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.asyncio
    async def test_plaintext_jwks_outside_loopback_is_rejected(self):
        with pytest.raises(RuntimeError, match="requires HTTPS"):
            await _fetch_jwks("http://identity.example.test/jwks")


# ---------------------------------------------------------------------------
# Combined Credential Verification
# ---------------------------------------------------------------------------


class TestVerifyCredentials:
    """Tests for bearer-policy verification."""

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.asyncio
    async def test_no_auth_configured_allows_through(self):
        """When no auth is configured, all requests are allowed."""
        cfg = _make_config()
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            result = await verify_credentials(request=request, bearer=None)
            assert result is None

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.asyncio
    async def test_no_credentials_when_auth_required_raises(self):
        """No credentials at all when auth is configured raises 401."""
        from fastapi import HTTPException

        cfg = _make_config(
            auth_jwt_jwks_uri="https://identity.example.test/jwks",
            server_tls_terminated=True,
            server_trusted_proxy_cidrs="127.0.0.1/32",
        )
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            with pytest.raises(HTTPException) as exc_info:
                await verify_credentials(request=request, bearer=None)
            assert exc_info.value.status_code == 401

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.asyncio
    async def test_valid_bearer_identity_is_propagated(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://identity.example.test/jwks")
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.security.auth._fetch_jwks",
                new=mock.AsyncMock(return_value={"keys": []}),
            ),
            mock.patch(
                "agent_utilities.security.auth._decode_jwt",
                return_value={"sub": "subject"},
            ),
        ):
            result = await verify_credentials(request=request, bearer=bearer)
        assert result == {"auth_type": "jwt", "sub": "subject"}
        assert request.state.user_claims == result


class TestBearerHeaderParser:
    def test_absent_header_returns_none(self):
        assert parse_bearer_authorization([]) is None

    def test_one_bearer_header_returns_opaque_token(self):
        assert parse_bearer_authorization([b"bEaReR opaque-token"]) == "opaque-token"

    @pytest.mark.parametrize(
        "authorization",
        [
            [b"Bearer first", b"Bearer second"],
            [b"Basic opaque"],
            [b"Bearer"],
            [b"Bearer two tokens"],
            [b"Bearer opaque\x00suffix"],
            [b"Bearer \xff"],
        ],
    )
    def test_ambiguous_or_malformed_header_fails_closed(self, authorization):
        with pytest.raises(PermissionError):
            parse_bearer_authorization(authorization)

    @pytest.mark.asyncio
    async def test_malformed_header_is_rejected_even_without_jwt_configuration(self):
        with mock.patch("agent_utilities.core.config.config", _make_config()):
            with pytest.raises(PermissionError):
                await authenticate_header_values(authorization=[b"Basic opaque"])

    @pytest.mark.asyncio
    async def test_token_claim_cannot_override_server_minted_auth_type(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://identity.example.test/jwks")
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.security.auth._fetch_jwks",
                new=mock.AsyncMock(return_value={"keys": []}),
            ),
            mock.patch(
                "agent_utilities.security.auth._decode_jwt",
                return_value={"sub": "principal:verified", "auth_type": "api_key"},
            ),
        ):
            identity = await authenticate_header_values(
                authorization=[b"Bearer opaque-token"]
            )
        assert identity == {
            "sub": "principal:verified",
            "auth_type": "jwt",
        }


# ---------------------------------------------------------------------------
# CORS & Host Configuration
# ---------------------------------------------------------------------------


class TestCORSConfig:
    """Tests for CORS and TrustedHost configuration reading."""

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_allowed_origins_parsed_from_config(self):
        """ALLOWED_ORIGINS config is split into a list."""
        origins_str = "https://app.example.com,https://admin.example.com"
        parsed = [o.strip() for o in origins_str.split(",")]
        assert parsed == ["https://app.example.com", "https://admin.example.com"]

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_allowed_origins_default_disables_cors(self):
        """When ALLOWED_ORIGINS is unset, cross-origin access is disabled."""
        from agent_utilities.server.app import _csv_values

        assert _csv_values(None) == []

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_allowed_hosts_parsed_from_config(self):
        """ALLOWED_HOSTS config is split into a list."""
        hosts_str = "api.example.com,*.example.com"
        parsed = [h.strip() for h in hosts_str.split(",")]
        assert parsed == ["api.example.com", "*.example.com"]

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_origins_with_whitespace_trimmed(self):
        """Whitespace around origins is trimmed."""
        origins_str = " https://a.com , https://b.com "
        parsed = [o.strip() for o in origins_str.split(",")]
        assert parsed == ["https://a.com", "https://b.com"]

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_remote_listener_requires_auth_and_host_allowlist(self):
        from agent_utilities.server.app import _http_boundary_settings

        cfg = _make_config()
        with mock.patch("agent_utilities.server.app.config", cfg):
            with pytest.raises(RuntimeError, match="requires JWT"):
                _http_boundary_settings("0.0.0.0")

        cfg = _make_config(
            auth_jwt_jwks_uri="https://identity.example.test/jwks",
            server_tls_terminated=True,
            server_trusted_proxy_cidrs="127.0.0.1/32",
        )
        with mock.patch("agent_utilities.server.app.config", cfg):
            with pytest.raises(RuntimeError, match="explicit ALLOWED_HOSTS"):
                _http_boundary_settings("0.0.0.0")

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_credentialed_cors_rejects_wildcard(self):
        from agent_utilities.server.app import _http_boundary_settings

        cfg = _make_config(
            allowed_origins="*",
            cors_allow_credentials=True,
        )
        with mock.patch("agent_utilities.server.app.config", cfg):
            with pytest.raises(RuntimeError, match="requires explicit"):
                _http_boundary_settings("127.0.0.1")

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_remote_listener_accepts_explicit_secure_boundary(self):
        from agent_utilities.server.app import _http_boundary_settings

        cfg = _make_config(
            auth_jwt_jwks_uri="https://identity.example.test/jwks",
            allowed_hosts="api.example.test",
            allowed_origins="https://ui.example.test",
            cors_allow_credentials=True,
            server_tls_terminated=True,
            server_trusted_proxy_cidrs="127.0.0.1/32",
        )
        with mock.patch("agent_utilities.server.app.config", cfg):
            origins, hosts = _http_boundary_settings("0.0.0.0")
        assert origins == ["https://ui.example.test"]
        assert hosts == ["api.example.test"]
