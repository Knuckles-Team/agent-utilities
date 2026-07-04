#!/usr/bin/python
"""Provider-agnostic OIDC discovery (CONCEPT:AU-OS.identity.resolve-token-endpoint-from).

Fleet auth is configured by **issuer URL only** — never vendor-specific endpoint
paths. Given an OIDC issuer (a Keycloak realm ``http://keycloak.arpa/realms/homelab``,
an Okta org ``https://ORG.okta.com/oauth2/default``, Auth0, Entra ID, …), fetch its
RFC 8414 / OIDC discovery document (``<issuer>/.well-known/openid-configuration``) and
read ``jwks_uri`` and ``token_endpoint`` from it. This keeps the identity provider
abstracted away: the only thing a deployment sets is ``OIDC_ISSUER`` (plus client id /
secret), and the provider-specific paths (Keycloak ``/protocol/openid-connect/certs``
vs Okta ``/v1/keys``) are resolved at runtime rather than hardcoded.

Used by:
- ``mcp/server_factory.py`` — derive the inbound JWT ``jwks_uri`` from the issuer.
- ``mcp/client_credentials.py`` — derive the outbound ``token_endpoint`` for minting.

Both keep an explicit override (``FASTMCP_SERVER_AUTH_JWT_JWKS_URI`` / ``OIDC_TOKEN_URL``)
for providers without reachable discovery or for pinning; discovery is the default.
"""

from __future__ import annotations

import time

import httpx

from agent_utilities.core._env import setting

# issuer -> (expiry_monotonic, discovery_doc)
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_S = 3600.0


def _verify_tls() -> bool:
    return setting("OIDC_TLS_VERIFY", "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )


def discover(issuer: str, *, verify: bool | None = None) -> dict:
    """Return the OIDC discovery document for ``issuer`` (cached, TTL 1h).

    Raises ``httpx.HTTPError`` if the document cannot be fetched/parsed.
    """
    issuer = issuer.rstrip("/")
    now = time.monotonic()
    cached = _cache.get(issuer)
    if cached and now < cached[0]:
        return cached[1]

    if verify is None:
        verify = _verify_tls()
    url = f"{issuer}/.well-known/openid-configuration"
    with httpx.Client(timeout=10.0, verify=verify) as client:
        resp = client.get(url)
        resp.raise_for_status()
        doc = resp.json()
    _cache[issuer] = (now + _CACHE_TTL_S, doc)
    return doc


def jwks_uri_for(issuer: str, *, verify: bool | None = None) -> str | None:
    """Discover the ``jwks_uri`` for ``issuer`` (``None`` if discovery fails)."""
    try:
        return discover(issuer, verify=verify).get("jwks_uri")
    except Exception:  # noqa: BLE001 - discovery is best-effort; caller falls back
        return None


def token_endpoint_for(issuer: str, *, verify: bool | None = None) -> str | None:
    """Discover the ``token_endpoint`` for ``issuer`` (``None`` if discovery fails)."""
    try:
        return discover(issuer, verify=verify).get("token_endpoint")
    except Exception:  # noqa: BLE001 - discovery is best-effort; caller falls back
        return None
