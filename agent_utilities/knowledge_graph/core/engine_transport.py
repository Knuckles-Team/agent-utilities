"""Secure native-engine client transport resolution.

``tls://`` endpoints opt into verified TLS (and optional mTLS) using the same
runtime-only transport profiles as HTTP, GraphQL, Neo4j, and Langfuse. CA and
client-key material is loaded into an in-memory ``SSLContext`` and temporary
secret material is removed before the connection is returned.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse


class EngineTransportError(ConnectionError):
    """Stable engine transport configuration failure."""


def _is_loopback_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    host = str(parsed.hostname or "").strip().lower()
    if host in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def engine_client_transport_kwargs(
    endpoint: str,
    *,
    config: Any = None,
    profile_name: str | None = None,
    profile_ref: str | None = None,
    server_hostname: str | None = None,
) -> dict[str, Any]:
    """Return safe kwargs for ``EpistemicGraphClient.connect``.

    Unix endpoints need no additions. Plain ``tcp://`` is limited to loopback.
    Remote deployments use ``tls://`` and a named/ref-backed ``ENGINE`` TLS
    profile.
    """
    rendered = str(endpoint or "").strip()
    if rendered.startswith("unix://") or not rendered:
        return {}
    if config is None:
        from agent_utilities.core.config import config as runtime_config

        config = runtime_config
    if rendered.startswith("tcp://"):
        if _is_loopback_endpoint(rendered):
            return {}
        raise EngineTransportError("remote native-engine transport requires tls://")
    if not rendered.startswith("tls://"):
        return {}

    from agent_utilities.core.transport_security import (
        TransportSecurityError,
        resolve_configured_tls_profile,
    )

    try:
        trust = resolve_configured_tls_profile(
            "ENGINE",
            profile_name=(
                profile_name
                if profile_name is not None
                else getattr(config, "engine_tls_profile", None)
            ),
            profile_ref=(
                profile_ref
                if profile_ref is not None
                else getattr(config, "engine_tls_profile_ref", None)
            ),
            config=config,
        )
    except TransportSecurityError:
        raise EngineTransportError("engine TLS profile is invalid") from None
    try:
        return {
            "tls": trust.ssl_context,
            "tls_server_hostname": (
                str(
                    server_hostname
                    if server_hostname is not None
                    else getattr(config, "engine_tls_server_name", "") or ""
                ).strip()
                or None
            ),
        }
    finally:
        trust.cleanup()


def native_endpoint_address(endpoint: str) -> tuple[str, bool]:
    """Return ``(host:port, uses_tls)`` for a native TCP endpoint."""
    rendered = str(endpoint or "").strip()
    if rendered.startswith("tls://"):
        return rendered[len("tls://") :], True
    if rendered.startswith("tcp://"):
        return rendered[len("tcp://") :], False
    raise EngineTransportError("native engine endpoint must use tcp:// or tls://")


__all__ = [
    "EngineTransportError",
    "engine_client_transport_kwargs",
    "native_endpoint_address",
]
