"""Canonical outbound HTTP client factory.

Every outbound ``httpx`` client in agent-utilities is built here instead of
inline, so the safety defaults are uniform and auditable in ONE place:

* **Explicit timeout** — ``DEFAULT_HTTP_TIMEOUT_S`` (30s) unless the caller
  passes one; an infinite timeout (``None``) is rejected outright.
* **TLS verification is mandatory** — callers may supply a verified
  :class:`ssl.SSLContext`, but cannot disable certificate or peer checks.
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
import socket
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import httpx
import requests

if TYPE_CHECKING:
    from agent_utilities.orchestration.resilience import ResiliencePolicy

__all__ = [
    "DEFAULT_HTTP_TIMEOUT_S",
    "AirgapViolation",
    "PinnedEgressViolation",
    "airgap_guard_transport",
    "create_async_http_client",
    "create_http_client",
    "create_requests_session",
    "http_retry_policy",
    "is_local_host",
    "pinned_egress_transport",
    "standard_headers",
]

#: Default request timeout (seconds) — every client gets a finite timeout.
DEFAULT_HTTP_TIMEOUT_S = 30.0
_ALLOWED_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


def _is_allowed_private_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Accept only loopback, RFC1918, or IPv6 ULA for private-host egress."""

    return address.is_loopback or any(
        address in network for network in _ALLOWED_PRIVATE_NETWORKS
    )


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


class _GovernedRequestsSession(requests.Session):
    """Requests-only SDK bridge with the mandatory outer safety policy."""

    def send(self, request: Any, **kwargs: Any) -> Any:
        if self.verify is False or kwargs.get("verify", self.verify) is False:
            raise ValueError("TLS verification cannot be disabled")
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = DEFAULT_HTTP_TIMEOUT_S
        if _airgap_enabled():
            host = str(urlsplit(str(request.url)).hostname or "")
            if not is_local_host(host):
                raise AirgapViolation(
                    "outbound request blocked by air-gap mode; configure an exact "
                    "local address"
                )
        return super().send(request, **kwargs)


class PinnedEgressViolation(httpx.TransportError):
    """An outbound request failed the DNS-pinning/peer-identity boundary."""


def _normalized_private_hosts(
    hosts: tuple[str, ...] | list[str] | set[str],
) -> frozenset[str]:
    normalized: set[str] = set()
    for raw in hosts:
        host = str(raw).strip().lower().rstrip(".")
        try:
            host.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("private egress hosts must be ASCII") from exc
        if (
            not host
            or len(host) > 253
            or any(ord(character) < 33 for character in host)
            or any(character in host for character in "/@*?#[]")
        ):
            raise ValueError("private egress hosts must be exact hostnames")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            labels = host.split(".")
            if any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or not all(
                    character.isalnum() or character == "-" for character in label
                )
                for label in labels
            ):
                raise ValueError(
                    "private egress hosts must be exact hostnames"
                ) from None
        normalized.add(host)
    if len(normalized) > 256:
        raise ValueError("private egress host allowlist is too large")
    return frozenset(normalized)


def _resolve_pinned_request(
    request: httpx.Request,
    *,
    allowed_private_hosts: frozenset[str],
    allow_loopback: bool,
) -> tuple[str, str, tuple[str, int]]:
    """Resolve one request once, mutate it to the selected IP, retain Host/SNI."""

    from agent_utilities.security.egress import (
        MAX_RESOLVED_ADDRESSES,
        validate_base_url_resolved,
    )

    if len(str(request.url)) > 8_192:
        raise PinnedEgressViolation("outbound URL exceeded its security boundary")
    logical_url = request.url
    scheme = logical_url.scheme.lower()
    host = (logical_url.host or "").lower().rstrip(".")
    if scheme not in {"http", "https"} or not host:
        raise PinnedEgressViolation("outbound URL was rejected")
    parsed = urlsplit(str(logical_url))
    if parsed.username is not None or parsed.password is not None:
        raise PinnedEgressViolation("embedded outbound credentials are not permitted")

    if host in allowed_private_hosts:
        try:
            answers = socket.getaddrinfo(
                host, logical_url.port, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise PinnedEgressViolation("outbound DNS resolution failed") from exc
        ips: list[str] = []
        for answer_count, answer in enumerate(answers, start=1):
            if answer_count > MAX_RESOLVED_ADDRESSES:
                raise PinnedEgressViolation("outbound DNS result exceeded its boundary")
            try:
                rendered = ipaddress.ip_address(str(answer[4][0])).compressed
            except (IndexError, TypeError, ValueError) as exc:
                raise PinnedEgressViolation("outbound DNS result was invalid") from exc
            if rendered not in ips:
                ips.append(rendered)
        if not ips:
            raise PinnedEgressViolation("outbound DNS resolution returned no address")
        if any(not _is_allowed_private_address(ipaddress.ip_address(ip)) for ip in ips):
            raise PinnedEgressViolation(
                "configured private host resolved outside the private boundary"
            )
    else:
        decision = validate_base_url_resolved(
            str(logical_url), allow_loopback=allow_loopback
        )
        if not decision.allowed or not decision.resolved_ips:
            raise PinnedEgressViolation("outbound destination was rejected")
        ips = list(decision.resolved_ips)

    pinned_ip = ips[0]
    address = ipaddress.ip_address(pinned_ip)
    if scheme == "http" and not (address.is_loopback or host in allowed_private_hosts):
        raise PinnedEgressViolation("unencrypted public egress is not permitted")

    authority = logical_url.netloc.decode("ascii")
    request.url = logical_url.copy_with(host=pinned_ip)
    request.headers["Host"] = authority
    if scheme == "https":
        request.extensions = {**request.extensions, "sni_hostname": host}
    return (
        host,
        pinned_ip,
        (scheme, logical_url.port or (443 if scheme == "https" else 80)),
    )


def _require_pinned_peer(response: httpx.Response, pinned_ip: str) -> None:
    stream = response.extensions.get("network_stream")
    get_extra_info = getattr(stream, "get_extra_info", None)
    if not callable(get_extra_info):
        raise PinnedEgressViolation("outbound peer identity was unavailable")
    peer = get_extra_info("server_addr")
    try:
        peer_ip = ipaddress.ip_address(str(peer[0])).compressed
    except (IndexError, TypeError, ValueError) as exc:
        raise PinnedEgressViolation("outbound peer identity was invalid") from exc
    if peer_ip != pinned_ip:
        raise PinnedEgressViolation("outbound peer did not match its DNS pin")


class _PinnedEgressTransport(httpx.BaseTransport):
    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        allowed_private_hosts: frozenset[str],
        allow_loopback: bool,
        verify_peer: bool,
    ) -> None:
        self._inner = inner
        self._allowed_private_hosts = allowed_private_hosts
        self._allow_loopback = allow_loopback
        self._verify_peer = verify_peer
        self._origin_hosts: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host, pinned_ip, origin = _resolve_pinned_request(
            request,
            allowed_private_hosts=self._allowed_private_hosts,
            allow_loopback=self._allow_loopback,
        )
        key = (f"{origin[0]}:{pinned_ip}", origin[1])
        with self._lock:
            prior = self._origin_hosts.setdefault(key, host)
        if prior != host:
            raise PinnedEgressViolation("outbound connection reuse crossed origins")
        response = self._inner.handle_request(request)
        if self._verify_peer:
            try:
                _require_pinned_peer(response, pinned_ip)
            except Exception:
                response.close()
                raise
        return response

    def close(self) -> None:
        self._inner.close()


class _AsyncPinnedEgressTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        allowed_private_hosts: frozenset[str],
        allow_loopback: bool,
        verify_peer: bool,
    ) -> None:
        self._inner = inner
        self._allowed_private_hosts = allowed_private_hosts
        self._allow_loopback = allow_loopback
        self._verify_peer = verify_peer
        self._origin_hosts: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host, pinned_ip, origin = _resolve_pinned_request(
            request,
            allowed_private_hosts=self._allowed_private_hosts,
            allow_loopback=self._allow_loopback,
        )
        key = (f"{origin[0]}:{pinned_ip}", origin[1])
        with self._lock:
            prior = self._origin_hosts.setdefault(key, host)
        if prior != host:
            raise PinnedEgressViolation("outbound connection reuse crossed origins")
        response = await self._inner.handle_async_request(request)
        if self._verify_peer:
            try:
                _require_pinned_peer(response, pinned_ip)
            except Exception:
                await response.aclose()
                raise
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def pinned_egress_transport(
    inner: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    *,
    is_async: bool = False,
    verify: Any = True,
    limits: httpx.Limits | None = None,
    allowed_private_hosts: tuple[str, ...] | list[str] | set[str] = (),
    allow_loopback: bool = False,
    verify_peer: bool = True,
) -> httpx.BaseTransport | httpx.AsyncBaseTransport:
    """Build a per-request DNS-pinning transport with Host/SNI preservation.

    ``verify_peer=False`` is only for an injected in-process transport that is
    already a trusted capability (for example a deterministic unit-test
    transport). Network clients keep the default peer verification.
    """

    normalized = _normalized_private_hosts(allowed_private_hosts)
    if inner is None:
        kwargs: dict[str, Any] = {"verify": verify}
        if limits is not None:
            kwargs["limits"] = limits
        inner = (
            httpx.AsyncHTTPTransport(**kwargs)
            if is_async
            else httpx.HTTPTransport(**kwargs)
        )
    if is_async:
        return _AsyncPinnedEgressTransport(
            cast("httpx.AsyncBaseTransport", inner),
            allowed_private_hosts=normalized,
            allow_loopback=allow_loopback,
            verify_peer=verify_peer,
        )
    return _PinnedEgressTransport(
        cast("httpx.BaseTransport", inner),
        allowed_private_hosts=normalized,
        allow_loopback=allow_loopback,
        verify_peer=verify_peer,
    )


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
        _AsyncAirgapGuardTransport(cast("httpx.AsyncBaseTransport", inner))
        if is_async
        else _AirgapGuardTransport(cast("httpx.BaseTransport", inner))
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
    if verify is False:
        raise ValueError("TLS verification cannot be disabled")
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
    pin_egress: bool = False,
    allowed_private_hosts: tuple[str, ...] | list[str] | set[str] = (),
    allow_loopback: bool = False,
    verify_pinned_peer: bool = True,
    **httpx_kwargs: Any,
) -> httpx.Client:
    """Build a sync :class:`httpx.Client` with the unified defaults.

    Args:
        timeout: Request timeout — seconds or an :class:`httpx.Timeout`;
            ``None`` (infinite) is rejected.
        verify: Verified TLS context or ``True``. ``False`` is rejected.
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
    if pin_egress:
        if kwargs.get("proxy") is not None or kwargs.get("trust_env", False):
            raise ValueError(
                "DNS-pinned egress does not accept ambient or inline proxies"
            )
        kwargs["trust_env"] = False
        kwargs["follow_redirects"] = False
        kwargs["transport"] = pinned_egress_transport(
            kwargs.get("transport"),
            is_async=False,
            verify=verify,
            limits=kwargs.get("limits"),
            allowed_private_hosts=allowed_private_hosts,
            allow_loopback=allow_loopback,
            verify_peer=verify_pinned_peer,
        )
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
    pin_egress: bool = False,
    allowed_private_hosts: tuple[str, ...] | list[str] | set[str] = (),
    allow_loopback: bool = False,
    verify_pinned_peer: bool = True,
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
    if pin_egress:
        if kwargs.get("proxy") is not None or kwargs.get("trust_env", False):
            raise ValueError(
                "DNS-pinned egress does not accept ambient or inline proxies"
            )
        kwargs["trust_env"] = False
        kwargs["follow_redirects"] = False
        kwargs["transport"] = pinned_egress_transport(
            kwargs.get("transport"),
            is_async=True,
            verify=verify,
            limits=kwargs.get("limits"),
            allowed_private_hosts=allowed_private_hosts,
            allow_loopback=allow_loopback,
            verify_peer=verify_pinned_peer,
        )
    kwargs["transport"] = airgap_guard_transport(
        kwargs.get("transport"),
        is_async=True,
        verify=verify,
        limits=kwargs.get("limits"),
    )
    return httpx.AsyncClient(**kwargs)


def create_requests_session(
    *,
    transport_security: Any | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Build the governed Requests session required by Requests-only SDKs.

    New application HTTP paths use :func:`create_http_client`. This narrow bridge
    exists for SDKs such as the OTLP HTTP exporter that require an actual
    ``requests.Session``. Verification cannot be disabled, standard headers are
    always applied, and a resolved transport profile owns any custom trust, mTLS,
    proxy, and ambient-environment policy.
    """

    session = _GovernedRequestsSession()
    session.verify = True
    session.headers.update({**standard_headers(), **(headers or {})})
    if transport_security is not None:
        transport_security.configure_requests_session(session)
    if session.verify is False:
        session.close()
        raise ValueError("TLS verification cannot be disabled")
    return session
