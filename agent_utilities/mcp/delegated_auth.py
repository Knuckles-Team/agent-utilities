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

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TOKEN_RESPONSE_BYTES = 1024 * 1024

_user_token_ctx: ContextVar[str | None] = ContextVar(
    "agent_utilities_delegated_user_token", default=None
)
_user_claims_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_utilities_delegated_user_claims", default=None
)


def _set_delegated_identity(
    token: str, claims: dict[str, Any]
) -> tuple[Token[str | None], Token[dict[str, Any] | None]]:
    """Scope validated delegation material to the current async task."""
    return _user_token_ctx.set(token), _user_claims_ctx.set(dict(claims))


def _reset_delegated_identity(
    tokens: tuple[Token[str | None], Token[dict[str, Any] | None]],
) -> None:
    _user_token_ctx.reset(tokens[0])
    _user_claims_ctx.reset(tokens[1])


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _get_default_config() -> dict[str, Any]:
    """Return the shared ``mcp_auth_config`` from server_factory.

    Deferred import to avoid circular dependencies during startup.
    """
    from agent_utilities.mcp.server_factory import mcp_auth_config

    return mcp_auth_config


def _post_token(
    endpoint: str,
    data: dict[str, Any],
    *,
    auth: tuple[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """POST one bounded token request through the canonical OIDC transport."""
    from agent_utilities.security.oidc_discovery import (
        canonical_oidc_endpoint,
        oidc_http_client,
    )

    endpoint = canonical_oidc_endpoint(endpoint, field="token endpoint")
    if not 1 <= int(timeout) <= 60:
        raise ValueError("Token request timeout is outside the safe range")

    with oidc_http_client(timeout=float(timeout)) as client:
        with client.stream("POST", endpoint, data=data, auth=auth) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_TOKEN_RESPONSE_BYTES:
                    raise RuntimeError("Token response exceeded its safety boundary")
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        raise RuntimeError("Token response was not valid JSON") from None
    if not isinstance(payload, dict):
        raise RuntimeError("Token response had an invalid shape")
    return payload


def _bounded_text(
    value: Any,
    *,
    field: str,
    minimum: int = 1,
    maximum: int = 4_096,
) -> str:
    rendered = str(value or "")
    if (
        not minimum <= len(rendered.encode("utf-8")) <= maximum
        or any(character in rendered for character in "\r\n\x00")
    ):
        raise ValueError(f"{field} is invalid")
    return rendered


# ---------------------------------------------------------------------------
# User-token accessors
# ---------------------------------------------------------------------------


def get_user_token() -> str | None:
    """Retrieve the OIDC Bearer token stored by ``UserTokenMiddleware``.

    Returns ``None`` when delegation is disabled or no token was captured.
    """
    return _user_token_ctx.get()


def get_user_claims() -> dict[str, Any] | None:
    """Retrieve decoded JWT claims stored by ``UserTokenMiddleware``."""
    claims = _user_claims_ctx.get()
    return dict(claims) if claims is not None else None


def get_user_identity() -> dict[str, Any]:
    """Return a structured identity dict for audit / identity passthrough.

    Returns only an opaque, non-reversible reference. Raw subject, email,
    display name and username claims must not cross a log/trace/persistence
    boundary.
    """
    claims = get_user_claims() or {}
    subject = claims.get("sub") or claims.get("client_id") or claims.get("azp")
    if not subject:
        return {"identity_ref": "", "has_claims": bool(claims)}
    from agent_utilities.security.persistence_privacy import persistence_reference

    return {
        "identity_ref": persistence_reference("delegated_actor", subject),
        "has_claims": True,
    }


# ---------------------------------------------------------------------------
# RFC 8693 — OAuth 2.0 Token Exchange
# ---------------------------------------------------------------------------


def get_delegated_token(
    config: dict[str, Any] | None = None,
    audience: str | None = None,
    scopes: str | None = None,
    timeout: int = 30,
) -> str:
    """Exchange the MCP-layer user token for a downstream service token.

    Implements `RFC 8693 <https://tools.ietf.org/html/rfc8693>`_ OAuth 2.0
    Token Exchange.  The subject token is the IdP-issued JWT stored by
    ``UserTokenMiddleware`` in request-scoped context-variable state.

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

    target_audience = _bounded_text(
        audience or cfg.get("audience"), field="delegation audience", maximum=2_048
    )
    target_scopes = _bounded_text(
        scopes or cfg.get("delegated_scopes", "api"),
        field="delegation scopes",
        maximum=4_096,
    )

    exchange_data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": user_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "audience": target_audience,
        "scope": target_scopes,
    }

    logger.info("Initiating RFC 8693 token exchange")

    try:
        payload = _post_token(
            token_endpoint,
            exchange_data,
            auth=(oidc_client_id, oidc_client_secret),
            timeout=timeout,
        )
        new_token = payload["access_token"]
        if not isinstance(new_token, str) or not 1 <= len(new_token) <= 65_536:
            raise RuntimeError("Token exchange response had an invalid token")
        logger.info("Token exchange successful")
        return new_token
    except KeyError:
        raise RuntimeError(
            "Token exchange response missing 'access_token' field."
        ) from None
    except Exception as e:
        logger.error("Token exchange failed (exception_type=%s)", type(e).__name__)
        raise RuntimeError("Token exchange failed") from None


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
        Required caller-bound CSRF state parameter.

    Returns
    -------
    str
        The full authorization URL to redirect the user to.
    """
    from urllib.parse import urlencode

    from agent_utilities.security.oidc_discovery import canonical_oidc_endpoint

    endpoint = canonical_oidc_endpoint(
        authorization_endpoint, field="authorization endpoint"
    )
    redirect = canonical_oidc_endpoint(redirect_uri, field="redirect URI")
    client = _bounded_text(client_id, field="OAuth client ID", maximum=512)
    if not isinstance(scopes, list) or not 1 <= len(scopes) <= 64:
        raise ValueError("OAuth scopes are invalid")
    requested_scopes = [
        _bounded_text(scope, field="OAuth scope", maximum=256) for scope in scopes
    ]
    csrf_state = _bounded_text(
        state, field="OAuth state", minimum=16, maximum=1_024
    )

    params = {
        "client_id": client,
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": " ".join(requested_scopes),
        "prompt": "consent",
        "state": csrf_state,
    }
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}{urlencode(params)}"


def exchange_authorization_code(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
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
    from agent_utilities.security.oidc_discovery import canonical_oidc_endpoint

    token_endpoint = canonical_oidc_endpoint(token_endpoint, field="token endpoint")
    redirect_uri = canonical_oidc_endpoint(redirect_uri, field="redirect URI")
    client_id = _bounded_text(client_id, field="OAuth client ID", maximum=512)
    client_secret = _bounded_text(
        client_secret, field="OAuth client secret", maximum=16_384
    )
    code = _bounded_text(code, field="authorization code", maximum=16_384)
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }

    logger.info("Exchanging authorization code for tokens")

    try:
        token_data = _post_token(token_endpoint, data, timeout=timeout)
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not 1 <= len(access_token) <= 65_536:
            raise RuntimeError("Authorization response had an invalid token")
        logger.info("Authorization code exchange successful")
        return token_data
    except Exception as e:
        logger.error(
            "Authorization code exchange failed (exception_type=%s)",
            type(e).__name__,
        )
        raise RuntimeError("Authorization code exchange failed") from None


def refresh_access_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
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
    timeout:
        HTTP timeout.

    Returns
    -------
    dict
        Token response containing a new ``access_token`` and potentially
        a new ``refresh_token``.
    """
    from agent_utilities.security.oidc_discovery import canonical_oidc_endpoint

    token_endpoint = canonical_oidc_endpoint(token_endpoint, field="token endpoint")
    client_id = _bounded_text(client_id, field="OAuth client ID", maximum=512)
    client_secret = _bounded_text(
        client_secret, field="OAuth client secret", maximum=16_384
    )
    refresh_token = _bounded_text(
        refresh_token, field="refresh token", maximum=65_536
    )
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }

    try:
        payload = _post_token(token_endpoint, data, timeout=timeout)
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not 1 <= len(access_token) <= 65_536:
            raise RuntimeError("Token refresh response had an invalid token")
        return payload
    except Exception as e:
        logger.error("Token refresh failed (exception_type=%s)", type(e).__name__)
        raise RuntimeError("Token refresh failed") from None


# ---------------------------------------------------------------------------
# Delegation check helper
# ---------------------------------------------------------------------------


def is_delegation_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return True if OIDC token delegation is active."""
    cfg = config or _get_default_config()
    return bool(cfg.get("enable_delegation", False))
