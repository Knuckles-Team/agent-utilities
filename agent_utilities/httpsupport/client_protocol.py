"""Implementation-neutral HTTP client protocol + exception taxonomy (GOC-87).

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

This module is the neutral seam of the staged ``httpx`` -> ``httpx2``
strangler migration. **Both packages remain concrete runtime contracts** —
17 locked packages require ``httpx`` (anthropic, openai, huggingface-hub,
llama-index-core, ...) and 3 require ``httpx2`` (``fastmcp-slim``, ``mcp``,
``genai-prices``) as of the 2026-08-16 lock — so a wholesale substitution
would break third-party model, tracing, transport, and test types that name
one package's concrete class directly. This module defines what application
code is allowed to depend on instead: a package-neutral request/response
shape and one exception taxonomy, so a call family can migrate from the
``httpx``-backed adapter (:mod:`agent_utilities.httpsupport.httpx_adapter`)
to the ``httpx2``-backed one (:mod:`agent_utilities.httpsupport.httpx2_adapter`)
— selected per family in :mod:`agent_utilities.httpsupport.transport_factory`
— without the caller changing a single line or branching on which package is
behind the boundary.

**No process-wide alias.** This module never does ``httpx = httpx2`` (or the
reverse) — that would conceal which package a consumer actually uses rather
than removing the dependency, and would break every third-party SDK that
does an ``isinstance`` check against the real class. Each adapter imports
its own backing package explicitly and normalizes at the boundary instead.

``httpx`` is removed from the dependency set only once the resolved
production lock and SBOM show zero runtime consumers (GOC-87 W08) — until
then, both packages are an explicit, documented migration state, not silent
drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AsyncHttpClient",
    "HttpClient",
    "HttpConnectError",
    "HttpProtocolError",
    "HttpResponse",
    "HttpTimeoutError",
    "HttpTooManyRedirectsError",
    "HttpTransportError",
    "map_transport_error",
    "normalize_response",
]


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Package-neutral snapshot of an HTTP response.

    Built by :func:`normalize_response` immediately after either adapter's
    backing client returns — so a real ``httpx.Response`` or ``httpx2.Response``
    instance is never handed back across the protocol boundary. ``json()`` is
    lazy (calls the original response's own decoder on first access) so a
    caller that never reads the body pays nothing for decoding it.
    """

    status_code: int
    headers: dict[str, str]
    content: bytes
    text: str
    _json: Callable[[], Any] = field(repr=False)

    def json(self) -> Any:
        return self._json()


def normalize_response(response: Any) -> HttpResponse:
    """Copy the fields application code needs off a package-concrete response.

    ``response`` is either an ``httpx.Response`` or an ``httpx2.Response`` —
    both expose the same structural shape (``status_code``, ``headers``,
    ``content``, ``text``, ``json()``), so one function normalizes both
    without importing either package.
    """
    return HttpResponse(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=response.content,
        text=response.text,
        _json=response.json,
    )


class HttpTransportError(RuntimeError):
    """Base of the AU-owned taxonomy every adapter maps transport failures onto.

    Application code catches THIS taxonomy — never ``httpx.TransportError``
    or ``httpx2.TransportError`` directly — so it never has to know or
    branch on which package is behind the protocol boundary for a given
    call family (GOC-87 design: "an exception taxonomy maps both packages'
    distinct exception hierarchies onto one AU-owned set").
    """


class HttpConnectError(HttpTransportError):
    """The connection could not be established (DNS, refused, TLS handshake)."""


class HttpTimeoutError(HttpTransportError):
    """The request exceeded its configured connect/read/write/pool timeout."""


class HttpTooManyRedirectsError(HttpTransportError):
    """The request exceeded the configured redirect limit."""


class HttpProtocolError(HttpTransportError):
    """The remote peer violated the HTTP protocol (malformed/decoding failure)."""


#: Exception CLASS NAMES (not isinstance checks against either package) that
#: map onto each taxonomy member. Matching by name — instead of importing
#: httpx/httpx2 here to do `isinstance` — is what keeps this module itself
#: free of a concrete dependency on either package: httpx and httpx2 name
#: their transport-failure subclasses identically (both derive them from the
#: same upstream author's exception vocabulary), so one table serves both.
_TIMEOUT_NAMES = frozenset(
    {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"}
)
_CONNECT_NAMES = frozenset({"ConnectError", "ProxyError", "UnsupportedProtocol"})
_REDIRECT_NAMES = frozenset({"TooManyRedirects"})
_PROTOCOL_NAMES = frozenset(
    {"RemoteProtocolError", "LocalProtocolError", "DecodingError", "StreamError"}
)


def map_transport_error(exc: Exception) -> HttpTransportError:
    """Map a raised httpx/httpx2 transport exception onto the AU taxonomy.

    Callers do ``raise map_transport_error(exc) from exc`` so the original
    package-concrete exception stays attached as ``__cause__`` for
    diagnostics, while the *type* application code catches is always one of
    this module's own classes.
    """
    name = type(exc).__name__
    if name in _TIMEOUT_NAMES:
        category: type[HttpTransportError] = HttpTimeoutError
    elif name in _CONNECT_NAMES:
        category = HttpConnectError
    elif name in _REDIRECT_NAMES:
        category = HttpTooManyRedirectsError
    elif name in _PROTOCOL_NAMES:
        category = HttpProtocolError
    else:
        category = HttpTransportError
    return category(f"{name}: {exc}")


@runtime_checkable
class HttpClient(Protocol):
    """Sync, implementation-neutral client contract.

    Defines only what AU application code actually needs — request +
    lifecycle — never either package's concrete ``Client``/``Transport``
    type. A caller holding an ``HttpClient`` cannot accidentally reach
    through it to an ``httpx.Client`` or ``httpx2.Client``.
    """

    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse: ...

    def close(self) -> None: ...


@runtime_checkable
class AsyncHttpClient(Protocol):
    """Async, implementation-neutral client contract (see :class:`HttpClient`)."""

    async def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse: ...

    async def aclose(self) -> None: ...
