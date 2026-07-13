"""Canonical outbound HTTP client factory.

Every outbound ``httpx`` client in agent-utilities is built here instead of
inline, so the safety defaults are uniform and auditable in ONE place:

* **Explicit timeout** — ``DEFAULT_HTTP_TIMEOUT_S`` (30s) unless the caller
  passes one; an infinite timeout (``None``) is rejected outright.
* **TLS verification ON by default** — ``verify=False`` must be an explicit,
  per-site decision (typically gated behind an ``ssl_verify`` flag), never an
  accidental default.
* **Standard headers** — a ``User-Agent`` identifying agent-utilities, merged
  beneath any caller-supplied headers (caller wins on conflict).
* **Optional declarative retry** — pass a
  :class:`~agent_utilities.orchestration.resilience.ResiliencePolicy` and
  transport-level failures are retried under it (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating), instead
  of each call site hand-rolling a retry loop. :func:`http_retry_policy`
  builds a policy whose ``retry_on`` matches httpx's transport errors.

Sync (:func:`create_http_client`) and async (:func:`create_async_http_client`)
variants accept the same parameters and any extra ``httpx`` keyword arguments
(``transport``, ``limits``, ``follow_redirects``, ...).
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from agent_utilities.orchestration.resilience import ResiliencePolicy

__all__ = [
    "DEFAULT_HTTP_TIMEOUT_S",
    "AirgapViolation",
    "airgap_guard_transport",
    "create_async_http_client",
    "create_http_client",
    "http_retry_policy",
    "is_local_host",
    "standard_headers",
]

#: Default request timeout (seconds) — every client gets a finite timeout.
DEFAULT_HTTP_TIMEOUT_S = 30.0


@lru_cache(maxsize=1)
def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("agent-utilities")
    except Exception:  # noqa: BLE001 - version lookup must never break a client
        return "0"


def standard_headers() -> dict[str, str]:
    """Headers every outbound agent-utilities HTTP call carries by default.

    Callers may override any of these via the factory's ``headers`` parameter
    (caller-supplied headers win on conflict).
    """
    return {"User-Agent": f"agent-utilities/{_package_version()}"}


class AirgapViolation(RuntimeError):
    """Raised when ``AIRGAP_MODE`` is on and a request targets a non-local host.

    CONCEPT:AU-OS.deployment.airgap-mode. Fail-closed: the request is never
    sent — this is raised *before* the wrapped transport is invoked.
    """


def _airgap_enabled() -> bool:
    """Live read of the one air-gap flag (config discipline: no direct
    ``os.environ`` reads outside ``core/_env.py``/``core/config.py``)."""
    from agent_utilities.core._env import setting

    return bool(setting("AIRGAP_MODE", False))


def is_local_host(host: str) -> bool:
    """Is ``host`` loopback / RFC1918-private / link-local?

    Deliberately does **not** perform a DNS lookup — resolving a hostname
    would itself be network activity, and would make the air-gap check
    non-deterministic (test-hostile, resolver-dependent). A bare literal IP
    or ``localhost`` is accepted; every other hostname is treated as
    non-local and blocked under air-gap mode. Point an air-gapped
    deployment's local endpoints (e.g. an on-LAN vLLM/Ollama host) at their
    private IP literal, not a DNS name, to reach them under ``AIRGAP_MODE=1``.
    """
    if not host:
        return False
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


def _check_airgap(request: httpx.Request) -> None:
    host = request.url.host or ""
    if not is_local_host(host):
        raise AirgapViolation(
            f"AIRGAP_MODE is enabled: outbound request to non-local host "
            f"{host!r} ({request.url}) was blocked before it was sent. Only "
            "loopback/private-network/link-local hosts are reachable in "
            "air-gap mode — point local endpoints at their private IP "
            "literal, not a DNS name."
        )


class _AirgapGuardTransport(httpx.BaseTransport):
    """Sync transport wrapper: fail-closed host check before delegating."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _check_airgap(request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


class _AsyncAirgapGuardTransport(httpx.AsyncBaseTransport):
    """Async transport wrapper: fail-closed host check before delegating."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _check_airgap(request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()


def airgap_guard_transport(
    inner: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    *,
    is_async: bool = False,
    verify: Any = True,
    limits: httpx.Limits | None = None,
) -> httpx.BaseTransport | httpx.AsyncBaseTransport | None:
    """Wrap ``inner`` in the air-gap host guard when ``AIRGAP_MODE`` is on.

    A no-op (returns ``inner`` unchanged, including ``None``) when the flag
    is off — zero behavior change for every deployment that doesn't opt in.
    When on and ``inner`` is ``None``, builds a default transport with
    ``verify``/``limits`` so callers that don't otherwise need a custom
    transport still get one to guard.
    """
    if not _airgap_enabled():
        return inner
    if inner is None:
        transport_kwargs: dict[str, Any] = {"verify": verify}
        if limits is not None:
            transport_kwargs["limits"] = limits
        inner = (
            httpx.AsyncHTTPTransport(**transport_kwargs)
            if is_async
            else httpx.HTTPTransport(**transport_kwargs)
        )
    return (
        _AsyncAirgapGuardTransport(inner) if is_async else _AirgapGuardTransport(inner)
    )


def http_retry_policy(
    *,
    max_attempts: int = 3,
    backoff_base_s: float = 0.5,
    backoff_factor: float = 2.0,
    max_backoff_s: float = 10.0,
    jitter: bool = True,
    name: str = "http",
) -> ResiliencePolicy:
    """A :class:`ResiliencePolicy` tuned for httpx transports.

    Retries httpx *transport* failures (connect errors, resets, timeouts —
    everything under :class:`httpx.TransportError`); HTTP status handling
    stays with the caller, who sees responses, not exceptions.
    """
    from agent_utilities.orchestration.resilience import ResiliencePolicy

    return ResiliencePolicy(
        max_attempts=max_attempts,
        backoff_base_s=backoff_base_s,
        backoff_factor=backoff_factor,
        max_backoff_s=max_backoff_s,
        jitter=jitter,
        retry_on=(httpx.TransportError,),
        name=name,
    )


class _ResilientTransport(httpx.BaseTransport):
    """Sync transport wrapper retrying failures under a ResiliencePolicy."""

    def __init__(self, inner: httpx.BaseTransport, policy: ResiliencePolicy):
        self._inner = inner
        self._policy = policy

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        from agent_utilities.orchestration.resilience import run_with_resilience_sync

        return run_with_resilience_sync(
            self._inner.handle_request, self._policy, request
        )

    def close(self) -> None:
        self._inner.close()


class _AsyncResilientTransport(httpx.AsyncBaseTransport):
    """Async transport wrapper retrying failures under a ResiliencePolicy."""

    def __init__(self, inner: httpx.AsyncBaseTransport, policy: ResiliencePolicy):
        self._inner = inner
        self._policy = policy

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from agent_utilities.orchestration.resilience import run_with_resilience

        return await run_with_resilience(
            self._inner.handle_async_request, self._policy, request
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def _prepare_kwargs(
    *,
    timeout: float | httpx.Timeout,
    verify: Any,
    headers: dict[str, str] | None,
    httpx_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if timeout is None:
        raise ValueError(
            "create_http_client requires a finite timeout; an infinite "
            "timeout (None) hides dead endpoints from callers."
        )
    merged_headers = {**standard_headers(), **(headers or {})}
    return {
        "timeout": timeout,
        "verify": verify,
        "headers": merged_headers,
        **httpx_kwargs,
    }


def create_http_client(
    *,
    timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT_S,
    verify: Any = True,
    headers: dict[str, str] | None = None,
    retry: ResiliencePolicy | None = None,
    **httpx_kwargs: Any,
) -> httpx.Client:
    """Build a sync :class:`httpx.Client` with the unified defaults.

    Args:
        timeout: Request timeout — seconds or an :class:`httpx.Timeout`;
            ``None`` (infinite) is rejected.
        verify: TLS verification (default ``True``). Pass ``False`` only at
            sites with an explicit, justified insecure flag.
        headers: Extra headers merged over :func:`standard_headers`.
        retry: Optional ResiliencePolicy; transport failures are retried
            under it (see :func:`http_retry_policy`).
        **httpx_kwargs: Forwarded to :class:`httpx.Client` (``transport``,
            ``limits``, ``follow_redirects``, ...).
    """
    kwargs = _prepare_kwargs(
        timeout=timeout, verify=verify, headers=headers, httpx_kwargs=httpx_kwargs
    )
    if retry is not None:
        inner = kwargs.pop("transport", None)
        if inner is None:
            transport_kwargs: dict[str, Any] = {"verify": verify}
            if "limits" in kwargs:
                transport_kwargs["limits"] = kwargs["limits"]
            inner = httpx.HTTPTransport(**transport_kwargs)
        kwargs["transport"] = _ResilientTransport(inner, retry)
    # Air-gap gate (no-op, returns kwargs.get("transport") unchanged, when
    # AIRGAP_MODE is off) — outermost so a blocked host is never handed to
    # the retry transport above.
    kwargs["transport"] = airgap_guard_transport(
        kwargs.get("transport"),
        is_async=False,
        verify=verify,
        limits=kwargs.get("limits"),
    )
    return httpx.Client(**kwargs)


def create_async_http_client(
    *,
    timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT_S,
    verify: Any = True,
    headers: dict[str, str] | None = None,
    retry: ResiliencePolicy | None = None,
    **httpx_kwargs: Any,
) -> httpx.AsyncClient:
    """Build an :class:`httpx.AsyncClient` with the unified defaults.

    Same parameters and semantics as :func:`create_http_client`.
    """
    kwargs = _prepare_kwargs(
        timeout=timeout, verify=verify, headers=headers, httpx_kwargs=httpx_kwargs
    )
    if retry is not None:
        inner = kwargs.pop("transport", None)
        if inner is None:
            transport_kwargs: dict[str, Any] = {"verify": verify}
            if "limits" in kwargs:
                transport_kwargs["limits"] = kwargs["limits"]
            inner = httpx.AsyncHTTPTransport(**transport_kwargs)
        kwargs["transport"] = _AsyncResilientTransport(inner, retry)
    kwargs["transport"] = airgap_guard_transport(
        kwargs.get("transport"),
        is_async=True,
        verify=verify,
        limits=kwargs.get("limits"),
    )
    return httpx.AsyncClient(**kwargs)
