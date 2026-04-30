"""Tests for the JWT and API Key authentication module.

CONCEPT:AU-011 — Secrets & Authentication

Covers:
- API key verification (valid, invalid, disabled)
- JWT token decoding (valid claims, expired, bad signature, wrong issuer/audience)
- Combined credential verification (API key OR JWT)
- JWKS caching behaviour
- CORS and TrustedHost configuration
"""

import time
from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from agent_utilities.security.auth import (
    _decode_jwt,
    verify_api_key_only,
    verify_credentials,
)

# ---------------------------------------------------------------------------
# Helper: Build a config mock
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Create a config-like mock with sensible defaults."""
    cfg = MagicMock()
    cfg.enable_api_auth = overrides.get("enable_api_auth", False)
    cfg.agent_api_key = overrides.get("agent_api_key", None)
    cfg.auth_jwt_jwks_uri = overrides.get("auth_jwt_jwks_uri", None)
    cfg.auth_jwt_issuer = overrides.get("auth_jwt_issuer", None)
    cfg.auth_jwt_audience = overrides.get("auth_jwt_audience", None)
    cfg.allowed_origins = overrides.get("allowed_origins", None)
    cfg.allowed_hosts = overrides.get("allowed_hosts", None)
    return cfg


# ---------------------------------------------------------------------------
# API Key Tests
# ---------------------------------------------------------------------------


class TestVerifyApiKey:
    """Tests for API key verification."""

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_api_key_disabled_returns_none(self):
        """When ENABLE_API_AUTH=False, returns None (allow through)."""
        cfg = _make_config(enable_api_auth=False)
        with mock.patch("agent_utilities.core.config.config", cfg):
            result = await verify_api_key_only(api_key="anything")
            assert result is None

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_api_key_valid(self):
        """Valid API key returns auth info dict."""
        cfg = _make_config(enable_api_auth=True, agent_api_key="test-key-123")
        with mock.patch("agent_utilities.core.config.config", cfg):
            result = await verify_api_key_only(api_key="test-key-123")
            assert result is not None
            assert result["auth_type"] == "api_key"

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_api_key_invalid_returns_none(self):
        """Invalid API key returns None (not authenticated, but doesn't reject yet)."""
        cfg = _make_config(enable_api_auth=True, agent_api_key="correct-key")
        with mock.patch("agent_utilities.core.config.config", cfg):
            result = await verify_api_key_only(api_key="wrong-key")
            assert result is None

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_api_key_none_returns_none(self):
        """No API key provided returns None."""
        cfg = _make_config(enable_api_auth=True, agent_api_key="correct-key")
        with mock.patch("agent_utilities.core.config.config", cfg):
            result = await verify_api_key_only(api_key=None)
            assert result is None


# ---------------------------------------------------------------------------
# JWT Decode Tests
# ---------------------------------------------------------------------------


class TestDecodeJWT:
    """Tests for JWT token decoding logic."""

    @pytest.mark.concept("AU-011")
    def test_decode_valid_token(self):
        """Decoding a valid JWT with matching claims succeeds."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        claims = _decode_jwt(
            token, jwks, issuer="https://auth.example.com", audience="my-api"
        )
        assert claims["sub"] == "user123"
        assert claims["iss"] == "https://auth.example.com"

    @pytest.mark.concept("AU-011")
    def test_decode_expired_token_raises(self):
        """Expired tokens raise HTTPException with 401."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt
        from fastapi import HTTPException

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) - 3600,
            "iat": int(time.time()) - 7200,
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(token, jwks, issuer="https://auth.example.com", audience="my-api")
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("AU-011")
    def test_decode_wrong_issuer_raises(self):
        """Token with wrong issuer raises HTTPException."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt
        from fastapi import HTTPException

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://wrong-issuer.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token, jwks,
                issuer="https://correct-issuer.com",
                audience="my-api",
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("AU-011")
    def test_decode_wrong_audience_raises(self):
        """Token with wrong audience raises HTTPException."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt
        from fastapi import HTTPException

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "wrong-audience",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(
                token, jwks,
                issuer="https://auth.example.com",
                audience="correct-audience",
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("AU-011")
    def test_decode_bad_signature_raises(self):
        """Token signed with a different key raises HTTPException."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt
        from fastapi import HTTPException

        key1 = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        key2 = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key2.as_dict(is_private=False)]}  # Different key

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key1).decode()

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(token, jwks, issuer="https://auth.example.com", audience="my-api")
        assert exc_info.value.status_code == 401

    @pytest.mark.concept("AU-011")
    def test_decode_no_issuer_audience_check(self):
        """Decoding without issuer/audience validation succeeds."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        claims = _decode_jwt(token, jwks, issuer=None, audience=None)
        assert claims["sub"] == "user123"

    @pytest.mark.concept("AU-011")
    def test_decode_audience_list_valid(self):
        """Token with audience as list containing expected value succeeds."""
        from authlib.jose import JsonWebKey
        from authlib.jose import jwt as authlib_jwt

        key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        jwks = {"keys": [key.as_dict(is_private=False)]}

        header = {"alg": "RS256"}
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "aud": ["my-api", "other-api"],
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = authlib_jwt.encode(header, payload, key).decode()

        claims = _decode_jwt(
            token, jwks, issuer="https://auth.example.com", audience="my-api"
        )
        assert claims["sub"] == "user123"


# ---------------------------------------------------------------------------
# Combined Credential Verification
# ---------------------------------------------------------------------------


class TestVerifyCredentials:
    """Tests for the combined API key + JWT verification."""

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_no_auth_configured_allows_through(self):
        """When no auth is configured, all requests are allowed."""
        cfg = _make_config()
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            result = await verify_credentials(
                request=request, api_key=None, bearer=None
            )
            assert result is None

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_valid_api_key_accepted(self):
        """Valid API key is accepted when JWT is not configured."""
        cfg = _make_config(enable_api_auth=True, agent_api_key="valid-key")
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            request.state = MagicMock()
            result = await verify_credentials(
                request=request, api_key="valid-key", bearer=None
            )
            assert result is not None
            assert result["auth_type"] == "api_key"

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_invalid_credentials_raises(self):
        """Invalid credentials raise 401."""
        from fastapi import HTTPException

        cfg = _make_config(enable_api_auth=True, agent_api_key="correct-key")
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            with pytest.raises(HTTPException) as exc_info:
                await verify_credentials(
                    request=request, api_key="wrong-key", bearer=None
                )
            assert exc_info.value.status_code == 401

    @pytest.mark.concept("AU-011")
    @pytest.mark.asyncio
    async def test_no_credentials_when_auth_required_raises(self):
        """No credentials at all when auth is configured raises 401."""
        from fastapi import HTTPException

        cfg = _make_config(enable_api_auth=True, agent_api_key="some-key")
        with mock.patch("agent_utilities.core.config.config", cfg):
            request = MagicMock(spec=Request)
            with pytest.raises(HTTPException) as exc_info:
                await verify_credentials(
                    request=request, api_key=None, bearer=None
                )
            assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# CORS & Host Configuration
# ---------------------------------------------------------------------------


class TestCORSConfig:
    """Tests for CORS and TrustedHost configuration reading."""

    @pytest.mark.concept("AU-011")
    def test_allowed_origins_parsed_from_config(self):
        """ALLOWED_ORIGINS config is split into a list."""
        origins_str = "https://app.example.com,https://admin.example.com"
        parsed = [o.strip() for o in origins_str.split(",")]
        assert parsed == ["https://app.example.com", "https://admin.example.com"]

    @pytest.mark.concept("AU-011")
    def test_allowed_origins_default_to_wildcard(self):
        """When ALLOWED_ORIGINS is None, default to ['*']."""
        origins = None
        result = [o.strip() for o in origins.split(",")] if origins else ["*"]
        assert result == ["*"]

    @pytest.mark.concept("AU-011")
    def test_allowed_hosts_parsed_from_config(self):
        """ALLOWED_HOSTS config is split into a list."""
        hosts_str = "api.example.com,*.example.com"
        parsed = [h.strip() for h in hosts_str.split(",")]
        assert parsed == ["api.example.com", "*.example.com"]

    @pytest.mark.concept("AU-011")
    def test_origins_with_whitespace_trimmed(self):
        """Whitespace around origins is trimmed."""
        origins_str = " https://a.com , https://b.com "
        parsed = [o.strip() for o in origins_str.split(",")]
        assert parsed == ["https://a.com", "https://b.com"]
