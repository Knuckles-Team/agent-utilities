#!/usr/bin/python
"""OIDC client-credentials token provider for the MCP multiplexer (CONCEPT:OS-5.32).

The multiplexer aggregates many child MCP servers. When a child enforces JWT auth
(``AUTH_TYPE=jwt``), the multiplexer must present a valid bearer token or its calls
are rejected (401). Children are configured per-entry in ``mcp_config.json`` and
historically carried no ``Authorization`` header, so flipping a child to jwt made
it unreachable through the multiplexer.

This module gives the multiplexer ONE service identity: it mints a Keycloak
service-account token via the OAuth2 ``client_credentials`` grant (audience
``agent-services`` — the same audience the children's ``JWTVerifier`` checks),
caches it, refreshes it before expiry, and the multiplexer attaches it to every
remote child that does not declare its own ``Authorization`` header.

Opt-in via ``MCP_CLIENT_AUTH=oidc-client-credentials``; otherwise every helper is
an inert no-op (returns ``None`` / ``{}``), so behaviour is unchanged when unset.

Configuration (env):
  MCP_CLIENT_AUTH        = oidc-client-credentials   # enable
  OIDC_CLIENT_ID         = mcp-multiplexer
  OIDC_CLIENT_SECRET     = <secret>                  # injected from OpenBao at deploy
  OIDC_AUDIENCE          = agent-services            # default
  OIDC_TOKEN_URL         = http://keycloak.arpa/realms/master/protocol/openid-connect/token
                           # default derived from FASTMCP_SERVER_AUTH_JWT_ISSUER
  OIDC_SCOPE             = <space-separated>          # optional
  OIDC_TLS_VERIFY        = true|false                # default true
"""

from __future__ import annotations

import os
import threading
import time

import requests
from fastmcp.utilities.logging import get_logger

logger = get_logger(name="MultiplexerClientAuth")

# Refresh this many seconds before the token actually expires.
_EXPIRY_SKEW_S = 30.0


def _enabled() -> bool:
    return os.getenv("MCP_CLIENT_AUTH", "none").strip().lower() == (
        "oidc-client-credentials"
    )


def _derive_token_url() -> str | None:
    explicit = os.getenv("OIDC_TOKEN_URL")
    if explicit:
        return explicit
    issuer = os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
    if issuer:
        return issuer.rstrip("/") + "/protocol/openid-connect/token"
    return None


class ClientCredentialsTokenProvider:
    """Thread-safe, self-refreshing OAuth2 client-credentials token cache."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        audience: str | None = None,
        scope: str | None = None,
        verify: bool = True,
        timeout: int = 15,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.scope = scope
        self.verify = verify
        self.timeout = timeout
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def get_token(self) -> str:
        """Return a cached access token, refreshing it if missing or near expiry."""
        with self._lock:
            now = time.monotonic()
            if self._token and now < self._expires_at - _EXPIRY_SKEW_S:
                return self._token
            data = {"grant_type": "client_credentials"}
            if self.audience:
                data["audience"] = self.audience
            if self.scope:
                data["scope"] = self.scope
            resp = requests.post(
                self.token_url,
                data=data,
                auth=(self.client_id, self.client_secret),
                verify=self.verify,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._expires_at = now + float(payload.get("expires_in", 300))
            logger.info(
                "Minted multiplexer service token (audience=%s, ttl=%ss)",
                self.audience,
                int(payload.get("expires_in", 300)),
            )
            return self._token


_provider: ClientCredentialsTokenProvider | None = None
_provider_lock = threading.Lock()
_provider_failed = False


def get_provider() -> ClientCredentialsTokenProvider | None:
    """Return the configured provider, or ``None`` when disabled/misconfigured."""
    global _provider, _provider_failed
    if not _enabled() or _provider_failed:
        return None
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is not None:
            return _provider
        token_url = _derive_token_url()
        client_id = os.getenv("OIDC_CLIENT_ID")
        client_secret = os.getenv("OIDC_CLIENT_SECRET")
        if not (token_url and client_id and client_secret):
            logger.error(
                "MCP_CLIENT_AUTH=oidc-client-credentials but OIDC_TOKEN_URL/"
                "FASTMCP_SERVER_AUTH_JWT_ISSUER, OIDC_CLIENT_ID or "
                "OIDC_CLIENT_SECRET is missing — children stay unauthenticated."
            )
            _provider_failed = True
            return None
        verify = os.getenv("OIDC_TLS_VERIFY", "true").strip().lower() not in (
            "false",
            "0",
            "no",
        )
        _provider = ClientCredentialsTokenProvider(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            audience=os.getenv("OIDC_AUDIENCE", "agent-services"),
            scope=os.getenv("OIDC_SCOPE") or None,
            verify=verify,
        )
        return _provider


def bearer_header(existing: dict | None) -> dict:
    """Authorization header for a remote child, or ``{}`` if not applicable.

    Never overrides an explicitly-configured ``Authorization`` header, and never
    raises — a token-mint failure degrades to no header (the child then 401s,
    which is visible in metrics/logs rather than crashing the multiplexer).
    """
    if existing and any(k.lower() == "authorization" for k in existing):
        return {}
    provider = get_provider()
    if provider is None:
        return {}
    try:
        return {"Authorization": f"Bearer {provider.get_token()}"}
    except Exception as exc:  # pragma: no cover - network/permission failure
        logger.warning("Could not mint multiplexer service token: %s", exc)
        return {}
