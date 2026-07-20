#!/usr/bin/python
"""Provider-neutral, DNS-pinned OIDC discovery and transport construction."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from agent_utilities.core.http_client import (
    create_async_http_client,
    create_http_client,
)
from agent_utilities.core.transport_security import resolve_configured_tls_profile

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_S = 3600.0
_MAX_DISCOVERY_BYTES = 1024 * 1024


def _canonical_endpoint(value: str, *, field: str) -> str:
    rendered = str(value or "").strip()
    if len(rendered) > 8_192:
        raise ValueError(f"{field} is too large")
    parsed = urlsplit(rendered)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    if parsed.scheme.lower() == "http" and parsed.hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError(f"{field} requires HTTPS outside loopback")
    return rendered


def canonical_oidc_endpoint(value: str, *, field: str = "OIDC endpoint") -> str:
    """Validate an operator/configuration supplied OAuth/OIDC endpoint."""
    return _canonical_endpoint(value, field=field)


def _trust() -> tuple[Any, list[str]]:
    from agent_utilities.core.config import config

    trust = resolve_configured_tls_profile(
        "OIDC",
        profile_name=config.oidc_tls_profile,
        profile_ref=config.oidc_tls_profile_ref,
        config=config,
    )
    if trust.proxy_url:
        # DNS pinning cannot prove the origin peer when a generic forward proxy
        # owns the TCP connection. Use a private egress gateway outside process.
        raise RuntimeError("OIDC TLS profile cannot configure an inline proxy")
    return trust.ssl_context, list(config.oidc_http_allowed_private_hosts)


def oidc_http_client(*, timeout: float = 15.0) -> httpx.Client:
    """Build the canonical synchronous OIDC/JWKS/token client."""
    verify, private_hosts = _trust()
    return create_http_client(
        timeout=timeout,
        verify=verify,
        trust_env=False,
        follow_redirects=False,
        pin_egress=True,
        allowed_private_hosts=private_hosts,
        allow_loopback=True,
    )


def oidc_async_http_client(*, timeout: float = 15.0) -> httpx.AsyncClient:
    """Build the canonical asynchronous OIDC/JWKS/token client."""
    verify, private_hosts = _trust()
    return create_async_http_client(
        timeout=timeout,
        verify=verify,
        trust_env=False,
        follow_redirects=False,
        pin_egress=True,
        allowed_private_hosts=private_hosts,
        allow_loopback=True,
    )


def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            raise RuntimeError("OIDC response exceeded its safety boundary")
    return bytes(body)


def discover(issuer: str) -> dict[str, Any]:
    """Return a bounded, validated OIDC discovery document (cached for one hour)."""
    canonical_issuer = _canonical_endpoint(issuer, field="OIDC issuer").rstrip("/")
    now = time.monotonic()
    cached = _cache.get(canonical_issuer)
    if cached and now < cached[0]:
        return dict(cached[1])

    url = f"{canonical_issuer}/.well-known/openid-configuration"
    with oidc_http_client(timeout=10.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            body = _read_bounded(response, _MAX_DISCOVERY_BYTES)
    try:
        document = json.loads(body)
    except (TypeError, ValueError):
        raise RuntimeError("OIDC discovery response was not valid JSON") from None
    if not isinstance(document, dict):
        raise RuntimeError("OIDC discovery response had an invalid shape")
    discovered_issuer = _canonical_endpoint(
        str(document.get("issuer") or ""), field="discovered issuer"
    ).rstrip("/")
    if discovered_issuer != canonical_issuer:
        raise RuntimeError("OIDC discovery issuer did not match configuration")
    for field in ("jwks_uri", "token_endpoint"):
        if document.get(field):
            document[field] = _canonical_endpoint(str(document[field]), field=field)
    _cache[canonical_issuer] = (now + _CACHE_TTL_S, dict(document))
    return dict(document)


def jwks_uri_for(issuer: str) -> str | None:
    """Discover the ``jwks_uri``; return ``None`` on a fail-closed miss."""
    try:
        value = discover(issuer).get("jwks_uri")
        return str(value) if value else None
    except Exception:
        return None


def token_endpoint_for(issuer: str) -> str | None:
    """Discover the ``token_endpoint``; return ``None`` on a fail-closed miss."""
    try:
        value = discover(issuer).get("token_endpoint")
        return str(value) if value else None
    except Exception:
        return None


__all__ = [
    "canonical_oidc_endpoint",
    "discover",
    "jwks_uri_for",
    "oidc_async_http_client",
    "oidc_http_client",
    "token_endpoint_for",
]
