"""Tests for the generic BaseBrowserAuthManager authentication flow.

CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
"""

import http.client
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.security.browser_auth import (
    BaseBrowserAuthManager,
    BaseLoopbackCallbackHandler,
    BaseLoopbackCallbackServer,
    generate_pkce,
)
from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
)
from tests.wiring import observe


class TestBaseBrowserAuthManager:
    """Tests for the generic BaseBrowserAuthManager core loops."""

    @pytest.fixture
    def mock_secrets_client(self):
        """Create an InEpistemicGraphBackend SecretsClient for testing."""
        client = MagicMock(spec=SecretsClient)
        client.backend = MagicMock(spec=InEpistemicGraphBackend)

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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_initialization(self, manager):
        """Should correctly initialize configurations and redirect URIs."""
        assert manager.client_id == "test-client-id"
        assert manager.auth_endpoint == "https://auth.example.com/oauth/auth"
        assert manager.token_endpoint == "https://auth.example.com/oauth/token"
        assert manager.scopes == "openid profile offline_access"
        assert manager.secret_key == "test/oauth_tokens"
        assert manager.redirect_uri == "http://127.0.0.1:56122/callback"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_pkce_generation(self):
        """PKCE generation should produce valid values."""
        verifier, challenge = generate_pkce()
        assert 43 <= len(verifier) <= 128
        assert len(challenge) == 43

        v2, c2 = generate_pkce()
        assert verifier != v2

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
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


class _FakeCallbackSelf:
    """Minimal stand-in for a BaseLoopbackCallbackHandler instance.

    Constructing a real BaseHTTPRequestHandler requires a live socket; the
    logging bug (BUG-140) and its fix live entirely in `log_message`, which
    only reads `self.path`/`self.command`, so a lightweight double is
    sufficient and avoids standing up a real loopback server per test.
    """

    # Colons deliberately embedded in the secret values: `agent_utilities`
    # already installs a process-wide LogRecord factory
    # (`core.log_privacy.install_log_privacy_boundary`, active for every
    # `agent_utilities.*` logger, including this one) that incidentally
    # masks *path-shaped* text — but its POSIX-path pattern stops at the
    # first `:`, so anything after one survives untouched. Using that shape
    # here proves this fix is doing real, additional work and is not merely
    # restating protection the ambient factory already provides by accident.
    path = "/callback?code=SUPER:SECRET_AUTH_CODE&state=SUPER:SECRET_CSRF_STATE"
    command = "GET"


def _pre_fix_log_message(self_obj, format: str, *args) -> None:  # noqa: A002
    """The exact pre-fix implementation, kept only to prove BUG-140 was real."""
    logging.getLogger("agent_utilities.security.browser_auth").debug(format, *args)


class TestLoopbackCallbackHandlerLogHygiene:
    """BUG-140: the loopback OAuth callback server must never log the raw
    request line/args — for this handler, the request line IS the OAuth
    redirect URI, and its query string carries the authorization `code` and
    CSRF `state` in the clear. The ambient process-wide path-privacy factory
    (`agent_utilities.core.log_privacy`) masks *some* shapes of this by
    accident but is not a substitute for not logging the secret at all (see
    the colon-embedding note on `_FakeCallbackSelf.path` above)."""

    def test_pre_fix_pattern_would_have_leaked_code_and_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failing-without baseline: forwarding BaseHTTPRequestHandler's
        default log_request() call (`format='"%s" %s %s'`,
        `args=(self.requestline, code, size)`) straight into `logger.debug`
        — what this file's `log_message` did before the fix — leaks both
        secrets the moment DEBUG logging is enabled, even with the ambient
        path-privacy factory active (proven by running this test the same
        way the rest of the suite does: `agent_utilities` already imported,
        factory already installed)."""
        fake = _FakeCallbackSelf()
        request_line = f"GET {fake.path} HTTP/1.1"
        logger_name = "agent_utilities.security.browser_auth"
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            _pre_fix_log_message(fake, '"%s" %s %s', request_line, "200", "-")
        assert "SECRET_AUTH_CODE" in caplog.text
        assert "SECRET_CSRF_STATE" in caplog.text

    def test_fixed_log_message_does_not_leak_code_or_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _FakeCallbackSelf()
        request_line = f"GET {fake.path} HTTP/1.1"
        logger_name = "agent_utilities.security.browser_auth"
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            BaseLoopbackCallbackHandler.log_message(
                fake, '"%s" %s %s', request_line, "200", "-"
            )
        assert "SECRET_AUTH_CODE" not in caplog.text
        assert "SECRET_CSRF_STATE" not in caplog.text
        # Still useful for local troubleshooting.
        assert "GET" in caplog.text

    def test_fixed_log_message_survives_an_unparseable_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The redaction path must itself fail closed to a safe placeholder,
        never fall back to logging the raw (possibly still secret-bearing)
        path on a parse error."""

        class _BadPathSelf:
            command = "GET"

            @property
            def path(self) -> str:
                raise ValueError("simulated unparseable path")

        logger_name = "agent_utilities.security.browser_auth"
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            BaseLoopbackCallbackHandler.log_message(_BadPathSelf(), "unused")
        assert "<unparseable>" in caplog.text

    def test_log_message_reached_from_a_real_http_callback_wiring(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Wire-First (D-OB-9): `log_message` has zero non-test callers inside
        `agent_utilities/` because its real caller is CPython's own
        `BaseHTTPRequestHandler.log_request()` (`http/server.py`), invoked
        polymorphically on `self` whenever `send_response()` runs — a
        text-based caller scan can never see an edge whose source lives in the
        stdlib, not this repo. So this test drives the actual live entrypoint
        `login()` relies on instead of calling `log_message` directly: a real
        socket-bound `BaseLoopbackCallbackServer` + the real, undoubled
        `BaseLoopbackCallbackHandler`, hit with a genuine HTTP GET carrying a
        code/state pair, exactly like a real browser OAuth redirect would.
        `observe()` proves the seam was reached without replacing it, and the
        response asserts BUG-140's fix held under the real dispatch path (not
        just the direct-call unit tests above)."""
        server = BaseLoopbackCallbackServer(
            ("127.0.0.1", 0), BaseLoopbackCallbackHandler
        )
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        logger_name = "agent_utilities.security.browser_auth"
        try:
            with caplog.at_level(logging.DEBUG, logger=logger_name):
                with observe(BaseLoopbackCallbackHandler, "log_message") as scan:
                    host, port = server.server_address[0], server.server_address[1]
                    conn = http.client.HTTPConnection(host, port, timeout=10)
                    try:
                        conn.request(
                            "GET",
                            "/callback?code=SUPER:SECRET_AUTH_CODE&"
                            "state=SUPER:SECRET_CSRF_STATE",
                        )
                        response = conn.getresponse()
                        response.read()
                    finally:
                        conn.close()
                    thread.join(timeout=10)
                    scan.assert_called(
                        why="a real OAuth callback request must reach "
                        "log_message through BaseHTTPRequestHandler's own "
                        "log_request() dispatch, the same path login() relies "
                        "on for every genuine browser redirect"
                    )
        finally:
            server.server_close()
        assert server.auth_code == "SUPER:SECRET_AUTH_CODE"
        assert "SECRET_AUTH_CODE" not in caplog.text
        assert "SECRET_CSRF_STATE" not in caplog.text
