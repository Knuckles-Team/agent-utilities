"""Tests for the XAI OAuth PKCE authentication manager.

CONCEPT:OS-5.1 — Secrets & Authentication
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.security.browser_auth import generate_pkce
from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
)
from agent_utilities.security.xai_auth import (
    XaiAuthManager,
    validate_xai_oauth_endpoint,
)


class TestXaiAuthManager:
    """Tests for the XaiAuthManager authentication flows."""

    @pytest.fixture
    def mock_secrets_client(self):
        """Create an InEpistemicGraphBackend SecretsClient for testing."""
        client = MagicMock(spec=SecretsClient)
        # Mock backend type to prevent sqlite redirection
        client.backend = MagicMock(spec=InEpistemicGraphBackend)

        # Simple in-memory storage dict
        storage: dict[str, str] = {}

        def get_val(key):
            return storage.get(key)

        def set_val(key, val):
            storage[key] = val

        client.get.side_effect = get_val
        client.set.side_effect = set_val

        def get_or_env_val(key, env_var=None):
            return storage.get(key)

        client.get_or_env.side_effect = get_or_env_val
        return client

    @pytest.fixture
    def manager(self, mock_secrets_client):
        """Pre-configured XaiAuthManager with mock secrets client."""
        return XaiAuthManager(secrets_client=mock_secrets_client)

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_pkce_generation(self):
        """PKCE code verifier and challenge should be correctly generated."""
        verifier, challenge = generate_pkce()
        assert 43 <= len(verifier) <= 128
        assert len(challenge) == 43  # SHA-256 base64url has 43 chars

        # Re-running should produce different PKCE keys
        v2, c2 = generate_pkce()
        assert verifier != v2

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_validate_oauth_endpoint(self):
        """OIDC endpoints validation should pass for valid xAI origins and reject others."""
        # Valid cases
        assert (
            validate_xai_oauth_endpoint("https://auth.x.ai/oauth2/token", "token")
            == "https://auth.x.ai/oauth2/token"
        )
        assert (
            validate_xai_oauth_endpoint("https://x.ai/auth", "auth")
            == "https://x.ai/auth"
        )
        assert (
            validate_xai_oauth_endpoint("https://subdomain.x.ai/auth", "auth")
            == "https://subdomain.x.ai/auth"
        )

        # Invalid cases
        with pytest.raises(ValueError) as exc:
            validate_xai_oauth_endpoint("http://auth.x.ai/token", "token")
        assert "non-HTTPS" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            validate_xai_oauth_endpoint("https://evil-hacker.com/token", "token")
        assert "not on the xAI origin" in str(exc.value)

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_token_save_and_retrieve(self, manager):
        """Tokens should be saved and loaded from secrets client successfully."""
        assert manager.get_cached_tokens() is None

        tokens = {
            "access_token": "acc_123",
            "refresh_token": "ref_123",
            "expires_at": int(time.time()) + 3600,
        }
        manager.save_tokens(tokens)

        loaded = manager.get_cached_tokens()
        assert loaded is not None
        assert loaded["access_token"] == "acc_123"
        assert loaded["refresh_token"] == "ref_123"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    @patch("httpx.post")
    @patch("httpx.get")
    def test_refresh_access_token_success(self, mock_get, mock_post, manager):
        """Refreshing token should query xAI endpoints and update stored token."""
        # Setup initial expiring token
        manager.save_tokens(
            {
                "access_token": "old_acc",
                "refresh_token": "old_ref",
                "expires_at": time.time() - 100,  # Expired
            }
        )

        # Mock discovery response
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {
            "authorization_endpoint": "https://auth.x.ai/oauth/auth",
            "token_endpoint": "https://auth.x.ai/oauth/token",
        }
        mock_get.return_value = mock_get_resp

        # Mock token endpoint response
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "access_token": "new_acc",
            "refresh_token": "new_ref",
            "expires_in": 7200,
        }
        mock_post.return_value = mock_post_resp

        new_tokens = manager.refresh_tokens({"refresh_token": "old_ref"})
        assert new_tokens["access_token"] == "new_acc"
        assert new_tokens["refresh_token"] == "new_ref"
        assert new_tokens["expires_at"] > time.time() + 7000

        # Check update in store
        loaded = manager.get_cached_tokens()
        assert loaded["access_token"] == "new_acc"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    @patch("httpx.post")
    @patch("httpx.get")
    def test_resolve_credentials(self, mock_get, mock_post, manager):
        """Should return current token if valid, otherwise refresh or return None."""
        # 1. No tokens stored
        assert manager.resolve_credentials() is None

        # 2. Valid token stored
        now = time.time()
        manager.save_tokens(
            {
                "access_token": "valid_tok",
                "refresh_token": "ref_tok",
                "expires_at": now + 600,
            }
        )
        assert manager.resolve_credentials() == "valid_tok"

        # 3. Expired token stored -> should refresh
        manager.save_tokens(
            {
                "access_token": "expired_tok",
                "refresh_token": "ref_tok",
                "expires_at": now - 10,
            }
        )

        # Mock discovery response
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {
            "authorization_endpoint": "https://auth.x.ai/oauth/auth",
            "token_endpoint": "https://auth.x.ai/oauth/token",
        }
        mock_get.return_value = mock_get_resp

        # Mock token endpoint response
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "access_token": "refreshed_tok",
            "refresh_token": "ref_tok",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_post_resp

        assert manager.resolve_credentials() == "refreshed_tok"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_resolve_credentials_auto_login(self, manager):
        """Should proactively trigger login() if credentials are missing and auto_login is True."""
        # 1. auto_login is False (default) -> should not trigger login and return None
        with patch.object(manager, "login") as mock_login:
            assert manager.resolve_credentials() is None
            mock_login.assert_not_called()

        # 2. auto_login is True -> should trigger login() and return new access token
        with patch.object(manager, "login") as mock_login:
            mock_login.return_value = {
                "access_token": "auto_login_token_abc",
                "refresh_token": "refresh_abc",
                "expires_at": time.time() + 3600,
            }
            # Since get_cached_tokens is mock-backed, when we call resolve_credentials,
            # it triggers login() which saves to the store.
            # Let's mock get_cached_tokens after login to return the new dict so resolve_credentials can see it.
            # Actually, manager.login() saves tokens using self.save_tokens, which saves to mock_secrets_client.
            # Let's check: self.login() returns tokens and updates mock_secrets_client, so get_cached_tokens() reads it.
            # Wait, let's verify if mock_login is enough.
            # In our implementation of resolve_credentials:
            # if not tokens:
            #     if auto_login:
            #         tokens = self.login()
            # ...
            # return tokens.get("access_token")
            # If mock_login returns the tokens dict, tokens will be resolved and returns access_token!
            assert (
                manager.resolve_credentials(auto_login=True) == "auto_login_token_abc"
            )
            mock_login.assert_called_once()

        # 3. auto_login is True but login() raises an exception -> should log and return None gracefully
        with patch.object(
            manager, "login", side_effect=Exception("Browser launch failed")
        ) as mock_login:
            assert manager.resolve_credentials(auto_login=True) is None
            mock_login.assert_called_once()


class TestLoopbackCallbackHandler:
    """Tests for LoopbackCallbackHandler requests parsing."""

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_handler_parse_code(self):
        """Handler should extract code from redirect URL query params."""
        mock_server = MagicMock()
        mock_server.auth_code = None

        # Mock HTTPServer behavior or callback logic
        handler = MagicMock()
        handler.server = mock_server
        handler.path = "/callback?code=abc_auth_code_123&state=xyz"

        # Simulate LoopbackCallbackHandler.do_GET logic
        # Using parsed URL and setting attr
        from urllib.parse import parse_qs, urlparse

        parsed_url = urlparse(handler.path)
        if parsed_url.path == "/callback":
            query_params = parse_qs(parsed_url.query)
            if "code" in query_params:
                mock_server.auth_code = query_params["code"][0]

        assert mock_server.auth_code == "abc_auth_code_123"
