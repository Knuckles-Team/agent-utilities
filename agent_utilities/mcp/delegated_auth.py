"""Delegated Authentication Module.

Provides vendor-neutral OAuth 2.0 token delegation helpers used by all
agent-package ``auth.py`` modules.  Two flows are supported:

1. **RFC 8693 Token Exchange** — exchanges an IdP-issued token (stored by
   ``UserTokenMiddleware``) for a downstream service-specific token.
2. **Three-Legged OAuth (3LO)** — Authorization Code Grant flow for services
   (like Atlassian Cloud) that require explicit user consent rather than
   direct token exchange.

All IdP references are vendor-neutral (OIDC ``.well-known`` endpoints).
Users configure their Identity Provider (any OIDC-compliant IdP) via
``OIDC_CONFIG_URL``.

Identity Passthrough
~~~~~~~~~~~~~~~~~~~~

When the downstream service does **not** support OIDC token exchange
(e.g. Langfuse, some SaaS APIs), the SSO token still secures the MCP
server layer.  The downstream call uses a service-account credential,
and the user's identity is logged for auditing.  Use
:func:`get_user_identity` for this pattern.

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)

local = threading.local()


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _get_default_config() -> dict[str, Any]:
    """Return the shared ``mcp_auth_config`` from server_factory.

    Deferred import to avoid circular dependencies during startup.
    """
    from agent_utilities.mcp.server_factory import mcp_auth_config

    return mcp_auth_config


# ---------------------------------------------------------------------------
# User-token accessors
# ---------------------------------------------------------------------------


def get_user_token() -> str | None:
    """Retrieve the OIDC Bearer token stored by ``UserTokenMiddleware``.

    Returns ``None`` when delegation is disabled or no token was captured.
    """
    return getattr(local, "user_token", None)


def get_user_claims() -> dict[str, Any] | None:
    """Retrieve decoded JWT claims stored by ``UserTokenMiddleware``."""
    return getattr(local, "user_claims", None)


def get_user_identity() -> dict[str, Any]:
    """Return a structured identity dict for audit / identity passthrough.

    Extracts ``sub``, ``email``, ``name``, and ``preferred_username`` from
    the JWT claims (if available).  Agents that cannot perform token
    exchange should call this to log who is making the downstream call.
    """
    claims = get_user_claims() or {}
    return {
        "subject": claims.get("sub"),
        "email": claims.get("email"),
        "name": claims.get("name"),
        "username": claims.get("preferred_username"),
        "has_claims": bool(claims),
    }


# ---------------------------------------------------------------------------
# RFC 8693 — OAuth 2.0 Token Exchange
# ---------------------------------------------------------------------------


def get_delegated_token(
    config: dict[str, Any] | None = None,
    audience: str | None = None,
    scopes: str | None = None,
    verify: bool = True,
    timeout: int = 30,
) -> str:
    """Exchange the MCP-layer user token for a downstream service token.

    Implements `RFC 8693 <https://tools.ietf.org/html/rfc8693>`_ OAuth 2.0
    Token Exchange.  The subject token is the IdP-issued JWT stored by
    ``UserTokenMiddleware`` in ``threading.local().user_token``.

    Parameters
    ----------
    config:
        Override dict with OIDC configuration keys.  Defaults to the
        shared ``mcp_auth_config`` from ``server_factory``.
    audience:
        Override for the target service audience (e.g.
        ``https://gitlab.example.com``).  Falls back to
        ``config["audience"]``.
    scopes:
        Override for requested scopes (space-separated).  Falls back to
        ``config["delegated_scopes"]``.
    verify:
        SSL verification for the token endpoint call.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    str
        The delegated access token for the downstream service.

    Raises
    ------
    ValueError
        If no user token is available or configuration is incomplete.
    RuntimeError
        If the token exchange request fails.
    """
    cfg = config or _get_default_config()

    user_token = get_user_token()
    if not user_token:
        raise ValueError(
            "No user token available for delegation.  Ensure the MCP server "
            "is configured with --auth-type oidc-proxy and "
            "--enable-delegation, and the request carries a Bearer token."
        )

    token_endpoint = cfg.get("token_endpoint")
    if not token_endpoint:
        raise ValueError(
            "No token_endpoint configured.  Ensure OIDC_CONFIG_URL is set "
            "and --enable-delegation was passed at MCP startup."
        )

    oidc_client_id = cfg.get("oidc_client_id")
    oidc_client_secret = cfg.get("oidc_client_secret")
    if not oidc_client_id or not oidc_client_secret:
        raise ValueError(
            "OIDC client credentials missing.  Set OIDC_CLIENT_ID and "
            "OIDC_CLIENT_SECRET environment variables."
        )

    target_audience = audience or cfg.get("audience")
    target_scopes = scopes or cfg.get("delegated_scopes", "api")

    exchange_data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": user_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",  # nosec B105
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",  # nosec B105
        "audience": target_audience,
        "scope": target_scopes,
    }

    logger.info(
        "Initiating RFC 8693 token exchange",
        extra={
            "audience": target_audience,
            "scopes": target_scopes,
            "token_endpoint": token_endpoint,
        },
    )

    try:
        response = requests.post(
            token_endpoint,
            data=exchange_data,
            auth=(oidc_client_id, oidc_client_secret),
            verify=verify,
            timeout=timeout,
        )
        response.raise_for_status()
        new_token = response.json()["access_token"]
        logger.info(
            "Token exchange successful",
            extra={"new_token_length": len(new_token)},
        )
        return new_token
    except requests.exceptions.HTTPError as e:
        logger.error(
            "Token exchange HTTP error",
            extra={
                "status_code": e.response.status_code
                if e.response is not None
                else None,
                "response_body": (
                    e.response.text[:500] if e.response is not None else ""
                ),
            },
        )
        raise RuntimeError(
            f"Token exchange failed (HTTP {e.response.status_code if e.response is not None else '?'}): "
            f"{e.response.text[:200] if e.response is not None else str(e)}"
        ) from e
    except KeyError:
        raise RuntimeError(
            "Token exchange response missing 'access_token' field."
        ) from None
    except Exception as e:
        logger.error("Token exchange failed", extra={"error": str(e)})
        raise RuntimeError(f"Token exchange failed: {str(e)}") from e


# ---------------------------------------------------------------------------
# Three-Legged OAuth (3LO) — Authorization Code Grant
# ---------------------------------------------------------------------------


def get_3lo_authorization_url(
    authorization_endpoint: str,
    client_id: str,
    redirect_uri: str,
    scopes: list[str],
    state: str | None = None,
) -> str:
    """Build the authorization URL for the 3-Legged OAuth flow.

    The caller should redirect the user to this URL.  After consent, the
    IdP redirects back to ``redirect_uri`` with an authorization code.

    Parameters
    ----------
    authorization_endpoint:
        The OIDC/OAuth authorization endpoint URL.
    client_id:
        The OAuth application client ID registered with the service.
    redirect_uri:
        The callback URL registered with the OAuth application.
    scopes:
        List of OAuth scopes to request.
    state:
        Optional CSRF state parameter.

    Returns
    -------
    str
        The full authorization URL to redirect the user to.
    """
    from urllib.parse import urlencode

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "prompt": "consent",
    }
    if state:
        params["state"] = state

    return f"{authorization_endpoint}?{urlencode(params)}"


def exchange_authorization_code(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    verify: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    """Exchange an authorization code for access and refresh tokens.

    This is the second step of the 3-Legged OAuth flow, executed after
    the user grants consent and the IdP returns an authorization code.

    Parameters
    ----------
    token_endpoint:
        The OIDC/OAuth token endpoint URL.
    client_id:
        The OAuth application client ID.
    client_secret:
        The OAuth application client secret.
    code:
        The authorization code received from the callback.
    redirect_uri:
        Must match the redirect_uri used in the authorization request.
    verify:
        SSL verification.
    timeout:
        HTTP timeout.

    Returns
    -------
    dict
        Token response containing ``access_token``, ``refresh_token``
        (if available), ``expires_in``, ``token_type``, and ``scope``.

    Raises
    ------
    RuntimeError
        If the code exchange fails.
    """
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }

    logger.info("Exchanging authorization code for tokens")

    try:
        response = requests.post(
            token_endpoint,
            data=data,
            verify=verify,
            timeout=timeout,
        )
        response.raise_for_status()
        token_data = response.json()
        logger.info("Authorization code exchange successful")
        return token_data
    except Exception as e:
        logger.error(f"Authorization code exchange failed: {e}")
        raise RuntimeError(f"3LO authorization code exchange failed: {e}") from e


def refresh_access_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    verify: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    """Use a refresh token to obtain a new access token.

    Parameters
    ----------
    token_endpoint:
        The OIDC/OAuth token endpoint URL.
    client_id:
        The OAuth application client ID.
    client_secret:
        The OAuth application client secret.
    refresh_token:
        The refresh token from a previous authorization.
    verify:
        SSL verification.
    timeout:
        HTTP timeout.

    Returns
    -------
    dict
        Token response containing a new ``access_token`` and potentially
        a new ``refresh_token``.
    """
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }

    try:
        response = requests.post(
            token_endpoint,
            data=data,
            verify=verify,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise RuntimeError(f"Token refresh failed: {e}") from e


# ---------------------------------------------------------------------------
# Delegation check helper
# ---------------------------------------------------------------------------


def is_delegation_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return True if OIDC token delegation is active."""
    cfg = config or _get_default_config()
    return bool(cfg.get("enable_delegation", False))
