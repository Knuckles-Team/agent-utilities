#!/usr/bin/env python
from __future__ import annotations

"""Universal Browser-based OAuth PKCE & Token Lifecycle Manager.

CONCEPT:OS-5.1 — Secrets & Authentication
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
    SecretsConfig,
    create_secrets_client,
)

logger = logging.getLogger(__name__)


def get_secrets_client_persistent() -> SecretsClient:
    """Resolve SecretsClient, defaulting to persistent SQLite if InMemory backend is configured."""
    client = create_secrets_client()
    if isinstance(client.backend, InEpistemicGraphBackend):
        config = SecretsConfig(
            backend="sqlite",
            sqlite_path=os.path.expanduser("~/.agent-utilities/secrets.db"),
        )
        return create_secrets_client(config)
    return client


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge using SHA-256 (S256)."""
    verifier = secrets.token_urlsafe(64)
    sha256_hash = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(sha256_hash).decode("utf-8").rstrip("=")
    return verifier, challenge


class BaseLoopbackCallbackServer(HTTPServer):
    """Configurable HTTPServer to store authorization code cleanly."""

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.auth_code: str | None = None


class BaseLoopbackCallbackHandler(BaseHTTPRequestHandler):
    """Callback request handler for standard OIDC code flow."""

    # Dynamic styling customization options
    redirect_path: str = "/callback"
    success_html: bytes = (
        b"<html><head><style>"
        b"body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        b"background-color: #0f1419; color: #fff; text-align: center; display: flex; "
        b"flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }"
        b".card { background-color: #15181c; border: 1px solid #2f3336; border-radius: 16px; "
        b"padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-width: 480px; }"
        b"h1 { color: #1d9bf0; margin-top: 0; font-size: 28px; }"
        b"p { color: #8899a6; font-size: 16px; line-height: 1.5; }"
        b".icon { font-size: 48px; margin-bottom: 20px; color: #00ba7c; }"
        b"</style></head><body>"
        b"<div class='card'>"
        b"<div class='icon'>&#10004;</div>"
        b"<h1>Authentication Successful</h1>"
        b"<p>Your application has been authorized successfully. You can now close this tab and return to the terminal.</p>"
        b"</div>"
        b"</body></html>"
    )
    error_html: bytes = (
        b"<html><head><style>"
        b"body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        b"background-color: #0f1419; color: #fff; text-align: center; display: flex; "
        b"flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }"
        b".card { background-color: #15181c; border: 1px solid #2f3336; border-radius: 16px; "
        b"padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-width: 480px; }"
        b"h1 { color: #f4212e; margin-top: 0; font-size: 28px; }"
        b"p { color: #8899a6; font-size: 16px; line-height: 1.5; }"
        b".icon { font-size: 48px; margin-bottom: 20px; color: #f4212e; }"
        b"</style></head><body>"
        b"<div class='card'>"
        b"<div class='icon'>&#10006;</div>"
        b"<h1>Authentication Failed</h1>"
        b"<p>Missing or invalid authorization parameters. Please try logging in again.</p>"
        b"</div>"
        b"</body></html>"
    )

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress logging request details to standard out to maintain clean CLI
        pass

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == self.redirect_path:
            query_params = parse_qs(parsed_url.query)
            if "code" in query_params:
                if isinstance(self.server, BaseLoopbackCallbackServer):
                    self.server.auth_code = query_params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(self.success_html)
                return

            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self.error_html)
            return

        self.send_response(404)
        self.end_headers()


class BaseBrowserAuthManager:
    """Generic base class managing browser-based OAuth PKCE authorization, storage, and refresh lifecycle."""

    def __init__(
        self,
        client_id: str,
        auth_endpoint: str,
        token_endpoint: str,
        scopes: str,
        secret_key: str,
        redirect_host: str = "127.0.0.1",
        redirect_port: int = 56122,
        redirect_path: str = "/callback",
        secrets_client: SecretsClient | None = None,
        refresh_skew_seconds: int = 120,
        oidc_discovery_url: str | None = None,
        validate_endpoint_fn: Callable[[str, str], str] | None = None,
        extra_auth_params: dict[str, str] | None = None,
        extra_token_params: dict[str, str] | None = None,
        api_key_secret_key: str | None = None,
        api_key_env_var: str | None = None,
    ):
        self.client_id = client_id
        self.auth_endpoint = auth_endpoint
        self.token_endpoint = token_endpoint
        self.scopes = scopes
        self.secret_key = secret_key
        self.redirect_host = redirect_host
        self.redirect_port = redirect_port
        self.redirect_path = redirect_path
        self.secrets_client = secrets_client or get_secrets_client_persistent()
        self.refresh_skew_seconds = refresh_skew_seconds
        self.oidc_discovery_url = oidc_discovery_url
        self.validate_endpoint_fn = validate_endpoint_fn
        self.extra_auth_params = extra_auth_params or {}
        self.extra_token_params = extra_token_params or {}
        self.api_key_secret_key = api_key_secret_key
        self.api_key_env_var = api_key_env_var

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.redirect_host}:{self.redirect_port}{self.redirect_path}"

    def get_cached_tokens(self) -> dict[str, Any] | None:
        """Retrieve stored tokens from the encrypted secrets client."""
        raw = self.secrets_client.get(self.secret_key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Failed to parse cached OAuth tokens for %r: %s", self.secret_key, exc
            )
            return None

    def save_tokens(self, tokens: dict[str, Any]) -> None:
        """Persist tokens to the encrypted secrets client."""
        self.secrets_client.set(self.secret_key, json.dumps(tokens))

    def _discover_endpoints(self) -> tuple[str, str]:
        """Perform OIDC Discovery to dynamically resolve auth and token endpoints if URL is configured."""
        if not self.oidc_discovery_url:
            return self.auth_endpoint, self.token_endpoint

        try:
            response = httpx.get(
                self.oidc_discovery_url,
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
            payload = response.json()
            auth_ep = payload.get("authorization_endpoint", self.auth_endpoint)
            token_ep = payload.get("token_endpoint", self.token_endpoint)

            if self.validate_endpoint_fn:
                auth_ep = self.validate_endpoint_fn(auth_ep, "authorization_endpoint")
                token_ep = self.validate_endpoint_fn(token_ep, "token_endpoint")

            return auth_ep, token_ep
        except Exception as exc:
            logger.error(
                "OIDC discovery failed for %s: %s", self.oidc_discovery_url, exc
            )
            # Graceful fallback to initial values
            return self.auth_endpoint, self.token_endpoint

    def refresh_tokens(self, tokens: dict[str, Any]) -> dict[str, Any]:
        """Refresh an expiring or expired access token using refresh_token."""
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            raise ValueError("No refresh_token found in credentials.")

        _, token_endpoint = self._discover_endpoints()
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh_token,
        }
        if self.extra_token_params:
            data.update(self.extra_token_params)

        try:
            response = httpx.post(
                token_endpoint,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=data,
                timeout=20.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error(
                "OAuth token refresh failed for endpoint %r: %s", token_endpoint, exc
            )
            raise ValueError(f"Failed to refresh tokens: {exc}") from exc

        new_tokens = {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token", refresh_token),
            "expires_at": time.time() + float(payload.get("expires_in", 3600)),
        }
        self.save_tokens(new_tokens)
        return new_tokens

    def resolve_credentials(self, auto_login: bool = False) -> str | None:
        """Master credential resolver.

        Checks direct API Key overrides first. Otherwise resolves, validates, and
        proactively refreshes stored OAuth tokens.
        """
        # 1. API Key Override Check
        if self.api_key_secret_key or self.api_key_env_var:
            api_key = self.secrets_client.get_or_env(
                self.api_key_secret_key or "", self.api_key_env_var or ""
            )
            if api_key:
                return api_key

        # 2. OAuth Token resolution
        tokens = self.get_cached_tokens()
        if not tokens:
            if auto_login:
                logger.info(
                    "Cached tokens missing for %r. Triggering interactive login...",
                    self.secret_key,
                )
                try:
                    tokens = self.login()
                except Exception as exc:
                    logger.error("Interactive login flow auto-trigger failed: %s", exc)
                    return None
            else:
                return None

        expires_at = tokens.get("expires_at", 0.0)
        # Check if expired or within refresh skew window
        if time.time() + self.refresh_skew_seconds >= expires_at:
            try:
                tokens = self.refresh_tokens(tokens)
            except Exception as exc:
                logger.error("Proactive token refresh failed: %s", exc)
                return None

        return tokens.get("access_token")

    def login(self) -> dict[str, Any]:
        """Execute full interactive OAuth PKCE Flow.

        Binds a local loopback server to capture authorization code.
        If loopback server fails, prompts user to copy-paste (if TTY is available).
        """
        auth_endpoint, token_endpoint = self._discover_endpoints()
        verifier, challenge = generate_pkce()

        # Start loopback callback server
        server: BaseLoopbackCallbackServer | None = None
        server_thread: threading.Thread | None = None

        # Build dynamic custom handler class
        class CustomHandler(BaseLoopbackCallbackHandler):
            redirect_path = self.redirect_path

        try:
            server = BaseLoopbackCallbackServer(
                (self.redirect_host, self.redirect_port),
                CustomHandler,
            )
            server_thread = threading.Thread(target=server.handle_request, daemon=True)
            server_thread.start()
            logger.info("Started loopback HTTP server on port %d", self.redirect_port)
        except OSError as exc:
            logger.warning(
                "Unable to start loopback server on port %d: %s. Falling back to headless manual paste mode.",
                self.redirect_port,
                exc,
            )

        # Build Authorization URL
        authorize_params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if self.extra_auth_params:
            authorize_params.update(self.extra_auth_params)

        auth_url = f"{auth_endpoint}?{urlencode(authorize_params)}"

        # Prompt user & attempt browser launch
        print("\n" + "=" * 80)
        print("Interactive OAuth Authentication Setup")
        print("=" * 80)
        print(f"Please navigate to the following link to authenticate:\n\n{auth_url}\n")
        print("=" * 80 + "\n")

        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        auth_code: str | None = None
        if server:
            # Wait up to 60 seconds for loopback thread callback to get code
            wait_limit = 60.0
            start_time = time.time()
            print("Waiting for browser callback authorization...")
            while time.time() - start_time < wait_limit:
                if server.auth_code:
                    auth_code = server.auth_code
                    break
                time.sleep(0.5)

        # Headless manual fallback
        if not auth_code:
            import sys

            print("Loopback server timed out or is unavailable.")
            if not sys.stdin.isatty():
                auth_file_path = os.path.expanduser(
                    "~/.agent-utilities/xai_auth_code.txt"
                )
                print(
                    f"Non-TTY environment detected. Please write the authorization code to {auth_file_path}"
                )
                print(f"Example: echo 'your_code_here' > {auth_file_path}")

                wait_limit = 300.0
                start_time = time.time()
                while time.time() - start_time < wait_limit:
                    if os.path.exists(auth_file_path):
                        with open(auth_file_path) as f:
                            code = f.read().strip()
                        if code:
                            auth_code = code
                            try:
                                os.remove(auth_file_path)
                            except OSError:
                                pass
                            break
                    time.sleep(1.0)

                if not auth_code:
                    raise TimeoutError(
                        f"Browser callback timed out. Authorization code was not provided in {auth_file_path}."
                    )
            else:
                print(
                    "Please open the URL above in your local browser, authorize, and then:"
                )
                paste_val = input(
                    "Paste the redirected URL (containing '?code=...') or the raw code: "
                ).strip()
                if "code=" in paste_val:
                    try:
                        parsed = urlparse(paste_val)
                        auth_code = parse_qs(parsed.query).get("code", [None])[0]
                    except Exception:
                        auth_code = paste_val
                else:
                    auth_code = paste_val

        if not auth_code:
            raise ValueError("Authentication cancelled or code was not provided.")

        # Exchange authorization code for access & refresh tokens
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": verifier,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if self.extra_token_params:
            data.update(self.extra_token_params)

        try:
            response = httpx.post(
                token_endpoint,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=data,
                timeout=20.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error(
                "Token exchange failed for endpoint %r: %s", token_endpoint, exc
            )
            raise ValueError(
                f"Failed to exchange authorization code for tokens: {exc}"
            ) from exc

        tokens = {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token", ""),
            "expires_at": time.time() + float(payload.get("expires_in", 3600)),
        }
        self.save_tokens(tokens)
        print("Successfully authenticated and stored credentials!\n")
        return tokens
