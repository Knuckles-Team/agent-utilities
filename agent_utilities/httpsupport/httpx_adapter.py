"""``httpx``-backed adapter for the implementation-neutral client protocol.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

Wraps :func:`agent_utilities.core.http_client.create_http_client` /
``create_async_http_client`` — the existing hardened factory — rather than
constructing ``httpx.Client``/``httpx.AsyncClient`` itself. This adapter adds
NO new behavior of its own: every safety property the factory already
enforces (mandatory finite timeout, mandatory TLS verification, optional
DNS-pinned egress, optional air-gap guard, optional
:class:`~agent_utilities.orchestration.resilience.ResiliencePolicy` retry,
standard headers) is inherited unchanged. This is the DEFAULT adapter — a
call family stays on it unless explicitly listed in
:data:`agent_utilities.httpsupport.transport_factory.MIGRATED_HTTPX2_FAMILIES`.
"""

from __future__ import annotations

from typing import Any

import httpx

from agent_utilities.core.http_client import (
    create_async_http_client,
    create_http_client,
)
from agent_utilities.httpsupport.client_protocol import (
    HttpResponse,
    map_transport_error,
    normalize_response,
)

__all__ = ["AsyncHttpxAdapter", "HttpxAdapter"]


class HttpxAdapter:
    """Sync ``HttpClient`` implementation over the governed httpx factory."""

    def __init__(self, **factory_kwargs: Any) -> None:
        self._client: httpx.Client = create_http_client(**factory_kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
        try:
            return normalize_response(self._client.request(method, url, **kwargs))
        except httpx.TransportError as exc:
            raise map_transport_error(exc) from exc

    def close(self) -> None:
        self._client.close()


class AsyncHttpxAdapter:
    """Async ``AsyncHttpClient`` implementation over the governed httpx factory."""

    def __init__(self, **factory_kwargs: Any) -> None:
        self._client: httpx.AsyncClient = create_async_http_client(**factory_kwargs)

    async def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
        try:
            return normalize_response(await self._client.request(method, url, **kwargs))
        except httpx.TransportError as exc:
            raise map_transport_error(exc) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
