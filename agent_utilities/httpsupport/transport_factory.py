"""Hardened construction seam for implementation-neutral HTTP clients.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge — GOC-87 staged ``httpx`` ->
``httpx2`` migration.

This is the ONE place a call family is switched from the ``httpx``-backed
adapter to the ``httpx2``-backed one. Every other file constructs a client
through :func:`create_http_client` / :func:`create_async_http_client` here
— never ``httpx2.Client``/``httpx2.AsyncClient`` directly —
:mod:`scripts.check_http_egress_boundary` (the CI/pre-commit egress-boundary
gate) enforces this by rejecting direct ``httpx2.Client``/``httpx2.AsyncClient``
construction outside this module and
:mod:`agent_utilities.httpsupport.httpx2_adapter` itself, exactly as it
already does for raw ``httpx.Client``/``httpx.AsyncClient`` outside
:mod:`agent_utilities.core.http_client`.

A family absent from :data:`MIGRATED_HTTPX2_FAMILIES` gets
:class:`~agent_utilities.httpsupport.httpx_adapter.HttpxAdapter` — a thin
wrapper over the existing, unchanged ``core.http_client`` factory — so
**every** call family not explicitly listed here sees zero behavior change
from before this module existed. Adding a family is a one-line addition to
the set below, made only after its timeout/TLS/redirect/retry parity is
verified and documented (see
``docs/architecture/httpx_httpx2_migration.md``) — never a bulk flip.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.httpsupport.client_protocol import AsyncHttpClient, HttpClient
from agent_utilities.httpsupport.httpx2_adapter import (
    AsyncHttpx2Adapter,
    Httpx2Adapter,
)
from agent_utilities.httpsupport.httpx_adapter import AsyncHttpxAdapter, HttpxAdapter

__all__ = [
    "MIGRATED_HTTPX2_FAMILIES",
    "create_async_http_client",
    "create_http_client",
]

#: Call families ported to the httpx2-backed adapter. GOC-87 W05: a family is
#: added here ONLY once verified to need none of DNS pinning, air-gap
#: enforcement, or transport-level retry (httpx2_adapter.py does not
#: reproduce those httpx-BaseTransport-typed wrappers — see its docstring),
#: and its before/after timeout/TLS/redirect parity is recorded in
#: docs/architecture/httpx_httpx2_migration.md.
#:
#: "gateway-widget-diagnostics" — agent_utilities/gateway/widgets/ollama.py's
#: two unauthenticated, unpinned, non-streaming local-network GET calls
#: (model/process listing for the dashboard). Before this port it used
#: module-level `httpx.get()` directly (not even through core.http_client),
#: so it had no DNS pinning or retry to begin with — porting it to
#: Httpx2Adapter is a net-neutral-to-positive change (adds a mandatory
#: finite timeout check, TLS verification, and standard headers it did not
#: have before), not a regression against any preserved invariant.
MIGRATED_HTTPX2_FAMILIES: frozenset[str] = frozenset({"gateway-widget-diagnostics"})


def create_http_client(*, family: str, **kwargs: Any) -> HttpClient:
    """Build a sync, protocol-typed client for ``family``.

    Args:
        family: The application call-family name (a stable string the
            caller owns, e.g. ``"gateway-widget-diagnostics"``). Selects the
            adapter via :data:`MIGRATED_HTTPX2_FAMILIES`.
        **kwargs: Forwarded to the selected adapter's constructor
            (``timeout``, ``verify``, ``headers``, plus adapter-specific
            passthrough such as ``transport=`` for httpx or ``mounts=`` for
            httpx2).
    """
    if family in MIGRATED_HTTPX2_FAMILIES:
        return Httpx2Adapter(**kwargs)
    return HttpxAdapter(**kwargs)


def create_async_http_client(*, family: str, **kwargs: Any) -> AsyncHttpClient:
    """Async counterpart of :func:`create_http_client`; same selection rule."""
    if family in MIGRATED_HTTPX2_FAMILIES:
        return AsyncHttpx2Adapter(**kwargs)
    return AsyncHttpxAdapter(**kwargs)
