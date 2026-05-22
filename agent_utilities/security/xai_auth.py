#!/usr/bin/python
from __future__ import annotations

"""Native xAI OAuth PKCE & Token Lifecycle Manager.

CONCEPT:OS-5.1 — Secrets & Authentication
"""

import logging
from typing import Any
from urllib.parse import urlparse

from agent_utilities.security.browser_auth import (
    BaseBrowserAuthManager,
    BaseLoopbackCallbackHandler,
    BaseLoopbackCallbackServer,
    get_secrets_client_persistent,
)
from agent_utilities.security.secrets_client import SecretsClient

logger = logging.getLogger(__name__)

# Constants
XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_REDIRECT_HOST = "127.0.0.1"
XAI_OAUTH_REDIRECT_PORT = 56121
XAI_OAUTH_REDIRECT_PATH = "/callback"
XAI_OAUTH_REDIRECT_URI = f"http://{XAI_OAUTH_REDIRECT_HOST}:{XAI_OAUTH_REDIRECT_PORT}{XAI_OAUTH_REDIRECT_PATH}"
XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120  # Refresh 2 mins before expiry


def get_secrets_client_for_xai() -> SecretsClient:
    """Resolve SecretsClient, defaulting to persistent SQLite if InMemory backend is configured."""
    return get_secrets_client_persistent()


def validate_xai_oauth_endpoint(url: str, field: str) -> str:
    """Validate OIDC discovery endpoints to refuse non-HTTPS or non-xAI hosts."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"xAI OIDC discovery {field} is missing a hostname: {url!r}.")
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise ValueError(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin. "
            "Refusing to use endpoint to protect against MITM attacks."
        )
    return url


# Keep LoopbackCallbackServer and LoopbackCallbackHandler for direct backwards compatibility
class LoopbackCallbackServer(BaseLoopbackCallbackServer):
    """Subclassed HTTPServer to store authorization code cleanly."""

    pass


class LoopbackCallbackHandler(BaseLoopbackCallbackHandler):
    """Callback request handler for standard OIDC code flow."""

    # Keep the custom old xAI styles for full regression safety
    redirect_path = XAI_OAUTH_REDIRECT_PATH
    success_html = (
        b"<html><head><style>"
        b"body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f1419; color: #fff; text-align: center; padding: 50px; }"
        b"h1 { color: #1d9bf0; }"
        b"</style></head><body>"
        b"<h1>Authentication Successful</h1>"
        b"<p>xAI integration has been successfully authorized. You may close this window and return to the console.</p>"
        b"</body></html>"
    )


class XaiAuthManager(BaseBrowserAuthManager):
    """Manages xAI OAuth authentication flow, token storage, and refresh lifecycle."""

    def __init__(self, secrets_client: SecretsClient | None = None):
        super().__init__(
            client_id=XAI_OAUTH_CLIENT_ID,
            auth_endpoint="https://auth.x.ai/oauth2/auth",
            token_endpoint="https://auth.x.ai/oauth2/token",  # nosec B106
            scopes=XAI_OAUTH_SCOPE,
            secret_key="xai/oauth_tokens",
            redirect_host=XAI_OAUTH_REDIRECT_HOST,
            redirect_port=XAI_OAUTH_REDIRECT_PORT,
            redirect_path=XAI_OAUTH_REDIRECT_PATH,
            secrets_client=secrets_client,
            refresh_skew_seconds=XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
            oidc_discovery_url=XAI_OAUTH_DISCOVERY_URL,
            validate_endpoint_fn=validate_xai_oauth_endpoint,
            extra_auth_params={
                "plan": "generic",
                "referrer": "agent-utilities",
            },
            api_key_secret_key="xai/api_key",
            api_key_env_var="XAI_API_KEY",
        )

    # Maintain public interfaces exactly for tests
    def get_cached_tokens(self) -> dict[str, Any] | None:
        return super().get_cached_tokens()

    def save_tokens(self, tokens: dict[str, Any]) -> None:
        super().save_tokens(tokens)

    def refresh_tokens(self, tokens: dict[str, Any]) -> dict[str, Any]:
        # Handle custom HTTP 403 / tier restriction mapping from original implementation
        try:
            return super().refresh_tokens(tokens)
        except Exception as exc:
            # Check if 403 was raised in parent call
            if "403" in str(exc):
                raise PermissionError(
                    "xAI token refresh returned HTTP 403. Your SuperGrok tier may be restricted. "
                    "Set XAI_API_KEY to switch to direct API key access."
                ) from exc
            raise
