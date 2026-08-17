"""``httpx2``-backed adapter for the implementation-neutral client protocol.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

``httpx2`` is imported lazily (function-local ``import httpx2``, matching
the D-MTT-1 convention already established in
:mod:`agent_utilities.mcp.httpx_boundary`) so a profile that omits the
``[mcp]`` extra — the only thing that pulls ``httpx2`` into the resolved
lock today — never fails to import this module merely by referencing the
adapter classes; only *constructing* one requires the package.

**This adapter does NOT get the DNS-pinning / air-gap-guard / resilience-
retry transports that** :mod:`agent_utilities.core.http_client` **builds for
httpx.** Those transports are ``httpx.BaseTransport`` subclasses
(:class:`agent_utilities.core.http_client._PinnedEgressTransport` etc.) —
GOC-87 authority #4 forbids passing one package's concrete transport object
into the other's client, so they cannot be reused here as-is. Every call
family selected into
:data:`agent_utilities.httpsupport.transport_factory.MIGRATED_HTTPX2_FAMILIES`
MUST therefore be verified NOT to require DNS pinning, air-gap enforcement,
or transport-level retry before it is ported — see that module's docstring
for the currently-migrated set and why each entry qualifies. A family that
needs those protections stays on :class:`agent_utilities.httpsupport.httpx_adapter.HttpxAdapter`
until an equivalent httpx2 transport wrapper is built and proven (tracked,
not implemented, as a GOC-87 W06 follow-on).

TLS verification and a finite timeout are still mandatory here, mirroring
``core.http_client``'s own invariants exactly (Authority #1: "package-name
preference never justifies breaking a concrete third-party contract" cuts
both ways — httpx2 does not get to be *less* safe than httpx either).
"""

from __future__ import annotations

from typing import Any

from agent_utilities.core.http_client import DEFAULT_HTTP_TIMEOUT_S, standard_headers
from agent_utilities.httpsupport.client_protocol import (
    HttpResponse,
    map_transport_error,
    normalize_response,
)

__all__ = ["AsyncHttpx2Adapter", "Httpx2Adapter"]


def _prepare_httpx2_kwargs(
    *,
    timeout: float,
    verify: Any,
    headers: dict[str, str] | None,
    httpx2_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if timeout is None:
        raise ValueError(
            "Httpx2Adapter requires a finite timeout; an infinite timeout "
            "(None) hides dead endpoints from callers."
        )
    if verify is False:
        raise ValueError("TLS verification cannot be disabled")
    merged_headers = {**standard_headers(), **(headers or {})}
    return {
        "timeout": timeout,
        "verify": verify,
        "headers": merged_headers,
        **httpx2_kwargs,
    }


class Httpx2Adapter:
    """Sync ``HttpClient`` implementation backed by ``httpx2.Client``."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
        verify: Any = True,
        headers: dict[str, str] | None = None,
        **httpx2_kwargs: Any,
    ) -> None:
        import httpx2

        kwargs = _prepare_httpx2_kwargs(
            timeout=timeout, verify=verify, headers=headers, httpx2_kwargs=httpx2_kwargs
        )
        self._httpx2 = httpx2
        self._client = httpx2.Client(**kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
        try:
            return normalize_response(self._client.request(method, url, **kwargs))
        except self._httpx2.TransportError as exc:
            raise map_transport_error(exc) from exc

    def close(self) -> None:
        self._client.close()


class AsyncHttpx2Adapter:
    """Async ``AsyncHttpClient`` implementation backed by ``httpx2.AsyncClient``."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
        verify: Any = True,
        headers: dict[str, str] | None = None,
        **httpx2_kwargs: Any,
    ) -> None:
        import httpx2

        kwargs = _prepare_httpx2_kwargs(
            timeout=timeout, verify=verify, headers=headers, httpx2_kwargs=httpx2_kwargs
        )
        self._httpx2 = httpx2
        self._client = httpx2.AsyncClient(**kwargs)

    async def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
        try:
            return normalize_response(await self._client.request(method, url, **kwargs))
        except self._httpx2.TransportError as exc:
            raise map_transport_error(exc) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
