"""Tests for the generic BaseBrowserAuthManager authentication flow.

CONCEPT:OS-5.1 — Secrets & Authentication
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.security.browser_auth import (
    BaseBrowserAuthManager,
    generate_pkce,
)
from agent_utilities.security.secrets_client import InMemoryBackend, SecretsClient


class TestBaseBrowserAuthManager:
    """Tests for the generic BaseBrowserAuthManager core loops."""

    @pytest.fixture
    def mock_secrets_client(self):
        """Create an InMemoryBackend SecretsClient for testing."""
        client = MagicMock(spec=SecretsClient)
        client.backend = MagicMock(spec=InMemoryBackend)

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
        """Pre-configured BaseBrowserAuthManager with mock secrets client."""
        return BaseBrowserAuthManager(
            client_id="test-client-id",
            auth_endpoint="https://auth.example.com/oauth/auth",
            token_endpoint="https://auth.example.com/oauth/token",
            scopes="openid profile offline_access",
            secret_key="test/oauth_tokens",
            secrets_client=mock_secrets_client,
            redirect_port=56122,
        )

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_initialization(self, manager):
        """Should correctly initialize configurations and redirect URIs."""
        assert manager.client_id == "test-client-id"
        assert manager.auth_endpoint == "https://auth.example.com/oauth/auth"
        assert manager.token_endpoint == "https://auth.example.com/oauth/token"
        assert manager.scopes == "openid profile offline_access"
        assert manager.secret_key == "test/oauth_tokens"
        assert manager.redirect_uri == "http://127.0.0.1:56122/callback"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_pkce_generation(self):
        """PKCE generation should produce valid values."""
        verifier, challenge = generate_pkce()
        assert 43 <= len(verifier) <= 128
        assert len(challenge) == 43

        v2, c2 = generate_pkce()
        assert verifier != v2

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
    def test_refresh_access_token_success(self, mock_post, manager):
        """Refreshing token should query configured token endpoint and update cache."""
        manager.save_tokens(
            {
                "access_token": "old_acc",
                "refresh_token": "old_ref",
                "expires_at": time.time() - 100,
            }
        )

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
    def test_resolve_credentials(self, mock_post, manager):
        """Should return current token if valid, otherwise trigger a refresh."""
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
        """Should trigger login() if credentials are missing and auto_login is True."""
        # 1. auto_login is False (default) -> should not trigger login and return None
        with patch.object(manager, "login") as mock_login:
            assert manager.resolve_credentials() is None
            mock_login.assert_not_called()

        # 2. auto_login is True -> should trigger login() and return access token
        with patch.object(manager, "login") as mock_login:
            mock_login.return_value = {
                "access_token": "auto_login_token_abc",
                "refresh_token": "refresh_abc",
                "expires_at": time.time() + 3600,
            }
            assert (
                manager.resolve_credentials(auto_login=True) == "auto_login_token_abc"
            )
            mock_login.assert_called_once()
