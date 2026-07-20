from __future__ import annotations

"""Fail-closed HTTP boundary shared by native source connectors.

Every network fetch validates the destination immediately before I/O, disables
automatic redirects, re-validates each redirect target, limits redirect hops,
and bounds the decoded response body. Private/reserved destinations are denied
unless their exact hostname is present in an operator-supplied allow-list.

Injected transports/fetch callables remain available for deterministic tests;
they never weaken validation of an untrusted redirect or pagination URL.
"""

import ipaddress
import json
import math
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

from agent_utilities.security.egress import (
    MAX_RESOLVED_ADDRESSES,
    validate_base_url,
    validate_base_url_resolved,
)

DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3
HARD_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
HARD_MAX_REDIRECTS = 10
HARD_MAX_TIMEOUT_SECONDS = 300.0
HARD_MAX_URL_CHARS = 8_192
HARD_MAX_REQUEST_HEADERS = 64
HARD_MAX_REQUEST_HEADER_BYTES = 64 * 1024
HARD_MAX_QUERY_PARAMETERS = 128
_METADATA_ADDRESSES = frozenset({"169.254.169.254", "fd00:ec2::254"})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_SENSITIVE_REDIRECT_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)
_FORBIDDEN_CALLER_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class SourceEgressError(ValueError):
    """A source URL violated the connector egress policy."""


@dataclass(frozen=True, slots=True)
class ResolvedSourceURL:
    """One logical source URL and the exact addresses approved for this hop."""

    url: str
    host: str
    scheme: str
    resolved_ips: tuple[str, ...]

    @property
    def pinned_ip(self) -> str:
        """Select one resolver-ordered address; the client never resolves again."""
        return self.resolved_ips[0]


@dataclass(frozen=True, slots=True)
class _PinnedProxy:
    url: str
    scheme: str


def _normalize_hostname(value: str) -> str:
    candidate = value.strip().lower().rstrip(".")
    if not candidate or "%" in candidate:
        raise SourceEgressError("Source hostname is not permitted by egress policy")
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        pass
    try:
        rendered = candidate.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise SourceEgressError(
            "Source hostname is not permitted by egress policy"
        ) from exc
    labels = rendered.split(".")
    if (
        len(rendered) > 253
        or any(not label or len(label) > 63 for label in labels)
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-._"
            for character in rendered
        )
    ):
        raise SourceEgressError("Source hostname is not permitted by egress policy")
    return rendered


def _validate_url_shape(url: str) -> SplitResult:
    if (
        not isinstance(url, str)
        or not url
        or len(url) > HARD_MAX_URL_CHARS
        or "\\" in url
        or any(ord(character) < 32 or ord(character) == 127 for character in url)
    ):
        raise SourceEgressError("Source URL is not permitted by egress policy")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.fragment:
        raise SourceEgressError("Source URL is not permitted by egress policy")
    _normalize_hostname(parsed.hostname)
    if parsed.username is not None or parsed.password is not None:
        raise SourceEgressError("Source URL is not permitted by egress policy")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SourceEgressError("Source URL is not permitted by egress policy") from exc
    if port == 0:
        raise SourceEgressError("Source URL is not permitted by egress policy")
    return parsed


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool):
        raise ValueError("timeout must be a finite positive number")
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a finite positive number") from exc
    if not math.isfinite(value) or not 0 < value <= HARD_MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout must be greater than 0 and at most {HARD_MAX_TIMEOUT_SECONDS}"
        )
    return value


def _approved_private_ip(ip: str) -> str:
    """Validate an address reached through an exact private-host exception."""
    if "%" in ip:
        raise SourceEgressError("Approved source hostname resolved unsafely")
    try:
        address = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise SourceEgressError("Approved source hostname resolved unsafely") from exc
    rendered = address.compressed
    if (
        rendered in _METADATA_ADDRESSES
        or address.is_multicast
        or address.is_unspecified
        or not (
            address.is_global
            or address.is_private
            or address.is_loopback
            or address.is_link_local
        )
    ):
        raise SourceEgressError("Approved source hostname resolved unsafely")
    return rendered


def _resolve_addresses(host: str, *, resolver: Any = None) -> tuple[str, ...]:
    getaddrinfo = resolver or socket.getaddrinfo
    try:
        answers = getaddrinfo(host, None)
    except OSError as exc:
        raise SourceEgressError("Source hostname did not resolve") from exc
    addresses: list[str] = []
    for answer_count, answer in enumerate(answers, start=1):
        if answer_count > MAX_RESOLVED_ADDRESSES:
            raise SourceEgressError("Source hostname returned too many addresses")
        try:
            rendered = str(answer[4][0])
        except (IndexError, TypeError) as exc:
            raise SourceEgressError("Source hostname resolved unsafely") from exc
        if rendered not in addresses:
            addresses.append(rendered)
    if not addresses:
        raise SourceEgressError("Source hostname did not resolve")
    return tuple(addresses)


def resolve_safe_source_url(
    url: str,
    *,
    allowed_private_hosts: Iterable[str] | None = None,
    resolver: Any = None,
) -> ResolvedSourceURL:
    """Resolve and approve one hop, returning addresses suitable for pinning.

    DNS is called exactly once here. The fetch path connects to ``pinned_ip``
    while retaining ``host`` as its HTTP authority and TLS SNI name.
    """
    allowed = normalize_allowed_hosts(allowed_private_hosts)
    parsed = _validate_url_shape(url)
    host = _normalize_hostname(parsed.hostname or "")
    syntactic = validate_base_url(url, allow_loopback=host in allowed)
    if not syntactic.allowed and host not in allowed:
        raise SourceEgressError("Source URL is not permitted by egress policy")

    if host in allowed:
        addresses = tuple(
            _approved_private_ip(ip)
            for ip in _resolve_addresses(host, resolver=resolver)
        )
    else:
        decision = validate_base_url_resolved(
            url, allow_loopback=False, resolver=resolver
        )
        if not decision.allowed or not decision.resolved_ips:
            raise SourceEgressError("Source URL is not permitted by egress policy")
        addresses = decision.resolved_ips
    return ResolvedSourceURL(
        url=url,
        host=host,
        scheme=parsed.scheme,
        resolved_ips=addresses,
    )


def _pinned_url(logical_url: str, ip: str) -> str:
    parsed = _validate_url_shape(logical_url)
    address = ipaddress.ip_address(ip)
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, parsed.query, ""))


def _logical_authority(logical_url: str) -> str:
    parsed = _validate_url_shape(logical_url)
    host = _normalize_hostname(parsed.hostname or "")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        rendered_host = host
    else:
        rendered_host = f"[{address.compressed}]" if address.version == 6 else host
    return (
        f"{rendered_host}:{parsed.port}" if parsed.port is not None else rendered_host
    )


def _logical_origin(logical_url: str) -> tuple[str, str, int]:
    parsed = _validate_url_shape(logical_url)
    return (
        parsed.scheme,
        _normalize_hostname(parsed.hostname or ""),
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


def _validated_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if len(headers or {}) > HARD_MAX_REQUEST_HEADERS:
        raise SourceEgressError("Source request contained too many headers")
    normalized: dict[str, str] = {}
    total_bytes = 0
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name).strip()
        value = str(raw_value).strip()
        lowered = name.casefold()
        if (
            not name
            or lowered in _FORBIDDEN_CALLER_HEADERS
            or any(character in name for character in "\r\n:")
            or any(character in value for character in "\r\n")
        ):
            raise SourceEgressError("Source request contained a forbidden header")
        try:
            total_bytes += len(name.encode("ascii")) + len(value.encode("latin-1"))
        except UnicodeEncodeError as exc:
            raise SourceEgressError(
                "Source request contained an invalid header"
            ) from exc
        if total_bytes > HARD_MAX_REQUEST_HEADER_BYTES:
            raise SourceEgressError(
                "Source request headers exceeded the configured limit"
            )
        normalized[name] = value
    return normalized


def _validated_params(params: dict[str, Any] | None) -> dict[str, Any]:
    selected = dict(params or {})
    if len(selected) > HARD_MAX_QUERY_PARAMETERS:
        raise SourceEgressError("Source request contained too many query parameters")
    for raw_name, raw_value in selected.items():
        values = raw_value if isinstance(raw_value, (list, tuple)) else (raw_value,)
        if len(values) > HARD_MAX_QUERY_PARAMETERS:
            raise SourceEgressError("Source query parameter contained too many values")
        rendered_name = str(raw_name)
        if len(rendered_name) > HARD_MAX_URL_CHARS:
            raise SourceEgressError(
                "Source query parameter exceeded the configured limit"
            )
        for value in values:
            if len(str(value)) > HARD_MAX_URL_CHARS:
                raise SourceEgressError(
                    "Source query parameter exceeded the configured limit"
                )
    return selected


def _headers_for_hop(
    logical_url: str,
    headers: dict[str, str],
    *,
    initial_origin: tuple[str, str, int],
) -> dict[str, str]:
    same_origin = _logical_origin(logical_url) == initial_origin
    selected = {
        name: value
        for name, value in headers.items()
        if same_origin or name.casefold() not in _SENSITIVE_REDIRECT_HEADERS
    }
    selected["Host"] = _logical_authority(logical_url)
    return selected


def _request_extensions(resolution: ResolvedSourceURL) -> dict[str, Any]:
    return {"sni_hostname": resolution.host} if resolution.scheme == "https" else {}


def _require_direct_peer(
    response: Any,
    resolution: ResolvedSourceURL,
    proxy: _PinnedProxy | None,
) -> None:
    """Verify that a direct connection reached the address selected pre-I/O."""
    if proxy is not None:
        return
    stream = response.extensions.get("network_stream")
    get_extra_info = getattr(stream, "get_extra_info", None)
    if not callable(get_extra_info):
        raise SourceEgressError("Source connection peer could not be verified")
    peer = get_extra_info("server_addr")
    try:
        peer_ip = ipaddress.ip_address(str(peer[0])).compressed
    except (IndexError, TypeError, ValueError) as exc:
        raise SourceEgressError("Source connection peer could not be verified") from exc
    if peer_ip != resolution.pinned_ip:
        raise SourceEgressError("Source connection peer did not match its DNS pin")


def _pin_configured_proxy(proxy_url: str | None) -> _PinnedProxy | None:
    """Pin an explicit TLS-profile proxy without exposing it to ambient DNS.

    HTTPS proxies are rejected because HTTPX/httpcore cannot independently set
    the proxy TLS name after replacing its hostname with the approved address.
    Plain HTTP and SOCKS proxies remain supported; their destination is pinned
    and their credentials remain runtime-only in the resolved profile.
    """
    if not proxy_url:
        return None
    if len(proxy_url) > HARD_MAX_URL_CHARS:
        raise SourceEgressError("Configured source proxy is not safely pinnable")
    parsed = urlsplit(proxy_url)
    host = _normalize_hostname(parsed.hostname or "")
    if (
        parsed.scheme not in {"http", "socks5", "socks5h"}
        or not host
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise SourceEgressError("Configured source proxy is not safely pinnable")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SourceEgressError(
            "Configured source proxy is not safely pinnable"
        ) from exc
    if port == 0:
        raise SourceEgressError("Configured source proxy is not safely pinnable")
    pinned_ip = _approved_private_ip(_resolve_addresses(host)[0])
    rendered_ip = (
        f"[{pinned_ip}]" if ipaddress.ip_address(pinned_ip).version == 6 else pinned_ip
    )
    userinfo = parsed.netloc.rpartition("@")[0] if "@" in parsed.netloc else ""
    authority = f"{userinfo}@{rendered_ip}" if userinfo else rendered_ip
    if parsed.port is not None:
        authority = f"{authority}:{parsed.port}"
    return _PinnedProxy(
        url=urlunsplit((parsed.scheme, authority, "", "", "")),
        scheme=parsed.scheme,
    )


def _require_proxy_target_compatibility(
    proxy: _PinnedProxy | None, resolution: ResolvedSourceURL
) -> None:
    if proxy is None or proxy.scheme.startswith("socks") or resolution.scheme == "http":
        return
    try:
        ipaddress.ip_address(resolution.host)
    except ValueError as exc:
        raise SourceEgressError(
            "Configured HTTP proxy cannot safely pin this HTTPS source"
        ) from exc


def _profile_httpx_kwargs(tls: Any, proxy: _PinnedProxy | None) -> dict[str, Any]:
    """Project a TLS profile while excluding unpinnable ambient proxies."""
    kwargs = tls.httpx_kwargs()
    # Trust anchors and mTLS are already captured in the profile SSLContext.
    # Environment-discovered proxies are not reference-resolved and cannot be
    # pinned, so this egress boundary accepts only the profile's explicit proxy.
    kwargs["trust_env"] = False
    if proxy is None:
        kwargs.pop("proxy", None)
    else:
        kwargs["proxy"] = proxy.url
    return kwargs


def normalize_allowed_hosts(hosts: Iterable[str] | None) -> frozenset[str]:
    """Normalize an exact-host allow-list; wildcards and URL-shaped entries fail."""
    normalized: set[str] = set()
    for raw in hosts or ():
        host = str(raw).strip().lower().rstrip(".")
        if not host:
            continue
        if any(ch in host for ch in ("/", "@", "*", "?", "#")):
            raise SourceEgressError(
                "Private-host allow-list entries must be exact hostnames"
            )
        normalized.add(_normalize_hostname(host))
    return frozenset(normalized)


def source_url_host(url: str) -> str:
    """Return a normalized validated hostname without doing DNS I/O."""
    _validate_url_shape(url)
    decision = validate_base_url(url, allow_loopback=False)
    if not decision.allowed:
        raise SourceEgressError("Source URL is not permitted by egress policy")
    return _normalize_hostname(urlsplit(url).hostname or "")


def require_safe_source_url(
    url: str,
    *,
    allowed_private_hosts: Iterable[str] | None = None,
    resolve_dns: bool = True,
    resolver: Any = None,
) -> str:
    """Validate one source URL and return its normalized hostname.

    ``allowed_private_hosts`` is exact-match only. It is the explicit operator
    escape hatch for approved intranet sources; it does not permit arbitrary
    private IPs, wildcard domains, embedded credentials, or alternate schemes.
    """
    allowed = normalize_allowed_hosts(allowed_private_hosts)
    parsed = _validate_url_shape(url)
    host = _normalize_hostname(parsed.hostname or "")

    # Always perform syntax/IP-literal validation. An explicitly approved host
    # may be private, but it still cannot carry userinfo or an invalid scheme.
    syntactic = validate_base_url(url, allow_loopback=host in allowed)
    if not syntactic.allowed and host not in allowed:
        raise SourceEgressError("Source URL is not permitted by egress policy")
    if not resolve_dns:
        return host
    return resolve_safe_source_url(
        url,
        allowed_private_hosts=allowed,
        resolver=resolver,
    ).host


def _read_bounded(response: Any, max_bytes: int) -> bytes:
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    length = response.headers.get("content-length")
    if length:
        try:
            declared_length = int(length)
        except ValueError:
            declared_length = None
        if declared_length is not None and declared_length > max_bytes:
            raise SourceEgressError("Source response exceeded the configured limit")
    body = bytearray()
    for chunk in response.iter_bytes():
        if len(chunk) > max_bytes - len(body):
            raise SourceEgressError("Source response exceeded the configured limit")
        body.extend(chunk)
    return bytes(body)


async def _read_bounded_async(response: Any, max_bytes: int) -> bytes:
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    length = response.headers.get("content-length")
    if length:
        try:
            declared_length = int(length)
        except ValueError:
            declared_length = None
        if declared_length is not None and declared_length > max_bytes:
            raise SourceEgressError("Source response exceeded the configured limit")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(chunk) > max_bytes - len(body):
            raise SourceEgressError("Source response exceeded the configured limit")
        body.extend(chunk)
    return bytes(body)


def safe_get_bytes(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allowed_private_hosts: Iterable[str] | None = None,
    allowed_redirect_hosts: Iterable[str] | None = None,
    transport: Any = None,
    tls_service: str = "source-http",
) -> tuple[bytes, str | None]:
    """Fetch a bounded body with a DNS-pinned, hop-by-hop egress policy."""
    if not 0 <= max_redirects <= HARD_MAX_REDIRECTS:
        raise ValueError(f"max_redirects must be between 0 and {HARD_MAX_REDIRECTS}")
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Native HTTP source connectors require 'httpx'") from exc

    timeout = _validate_timeout(timeout)
    request_headers = _validated_headers(headers)
    private_hosts = normalize_allowed_hosts(allowed_private_hosts)
    redirect_hosts = normalize_allowed_hosts(allowed_redirect_hosts)
    initial_host = require_safe_source_url(
        url,
        allowed_private_hosts=private_hosts,
        resolve_dns=False,
    )
    initial_origin = _logical_origin(url)
    permitted_redirect_hosts = {initial_host, *redirect_hosts}
    current = url
    current_params = _validated_params(params)
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": False,
    }
    tls = None
    proxy = None
    try:
        if transport is not None:
            client_kwargs["transport"] = transport
        else:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            tls = resolve_configured_tls_profile(tls_service)
            proxy = _pin_configured_proxy(tls.proxy_url)
            client_kwargs.update(_profile_httpx_kwargs(tls, proxy))

        from agent_utilities.core.http_client import create_http_client
    except BaseException:
        if tls is not None:
            tls.cleanup()
        raise

    try:
        with create_http_client(**client_kwargs) as client:
            for hop in range(max_redirects + 1):
                if transport is None:
                    resolution = resolve_safe_source_url(
                        current,
                        allowed_private_hosts=private_hosts,
                    )
                    _require_proxy_target_compatibility(proxy, resolution)
                    request_url = _pinned_url(current, resolution.pinned_ip)
                    extensions = _request_extensions(resolution)
                else:
                    parsed = _validate_url_shape(current)
                    resolution = ResolvedSourceURL(
                        url=current,
                        host=_normalize_hostname(parsed.hostname or ""),
                        scheme=parsed.scheme,
                        resolved_ips=((parsed.hostname or ""),),
                    )
                    request_url = current
                    extensions = {}
                if resolution.host not in permitted_redirect_hosts:
                    raise SourceEgressError(
                        "Cross-host source redirect is not permitted"
                    )
                logical_request_url = str(httpx.URL(current, params=current_params))
                _validate_url_shape(logical_request_url)
                hop_headers = _headers_for_hop(
                    current,
                    request_headers,
                    initial_origin=initial_origin,
                )
                with client.stream(
                    "GET",
                    request_url,
                    params=current_params,
                    headers=hop_headers,
                    extensions=extensions,
                ) as response:
                    if transport is None:
                        _require_direct_peer(response, resolution, proxy)
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        if hop >= max_redirects:
                            raise SourceEgressError("Source redirect limit exceeded")
                        location = response.headers.get("location")
                        if not location:
                            raise SourceEgressError(
                                "Source redirect omitted its destination"
                            )
                        destination = urljoin(logical_request_url, location)
                        next_url = _validate_url_shape(destination)
                        next_host = require_safe_source_url(
                            destination,
                            allowed_private_hosts=private_hosts,
                            resolve_dns=False,
                        )
                        if next_host not in permitted_redirect_hosts:
                            raise SourceEgressError(
                                "Cross-host source redirect is not permitted"
                            )
                        if resolution.scheme == "https" and next_url.scheme == "http":
                            raise SourceEgressError(
                                "HTTPS source redirect downgrade is forbidden"
                            )
                        current = destination
                        current_params = {}
                        continue
                    response.raise_for_status()
                    return _read_bounded(response, max_bytes), response.encoding
    except httpx.HTTPError:
        raise SourceEgressError("Source fetch failed") from None
    finally:
        if tls is not None:
            tls.cleanup()
    raise SourceEgressError("Source fetch did not produce a response")


async def safe_get_bytes_async(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allowed_private_hosts: Iterable[str] | None = None,
    allowed_redirect_hosts: Iterable[str] | None = None,
    transport: Any = None,
    tls_service: str = "source-http",
) -> tuple[bytes, str | None]:
    """Async equivalent of :func:`safe_get_bytes` with identical policy."""
    if not 0 <= max_redirects <= HARD_MAX_REDIRECTS:
        raise ValueError(f"max_redirects must be between 0 and {HARD_MAX_REDIRECTS}")
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Native HTTP source connectors require 'httpx'") from exc

    timeout = _validate_timeout(timeout)
    request_headers = _validated_headers(headers)
    private_hosts = normalize_allowed_hosts(allowed_private_hosts)
    redirect_hosts = normalize_allowed_hosts(allowed_redirect_hosts)
    initial_host = require_safe_source_url(
        url,
        allowed_private_hosts=private_hosts,
        resolve_dns=False,
    )
    initial_origin = _logical_origin(url)
    permitted_redirect_hosts = {initial_host, *redirect_hosts}
    current = url
    current_params = _validated_params(params)
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": False,
    }
    tls = None
    proxy = None
    try:
        if transport is not None:
            client_kwargs["transport"] = transport
        else:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            tls = resolve_configured_tls_profile(tls_service)
            proxy = _pin_configured_proxy(tls.proxy_url)
            client_kwargs.update(_profile_httpx_kwargs(tls, proxy))

        from agent_utilities.core.http_client import create_async_http_client
    except BaseException:
        if tls is not None:
            tls.cleanup()
        raise

    try:
        async with create_async_http_client(**client_kwargs) as client:
            for hop in range(max_redirects + 1):
                if transport is None:
                    resolution = resolve_safe_source_url(
                        current,
                        allowed_private_hosts=private_hosts,
                    )
                    _require_proxy_target_compatibility(proxy, resolution)
                    request_url = _pinned_url(current, resolution.pinned_ip)
                    extensions = _request_extensions(resolution)
                else:
                    parsed = _validate_url_shape(current)
                    resolution = ResolvedSourceURL(
                        url=current,
                        host=_normalize_hostname(parsed.hostname or ""),
                        scheme=parsed.scheme,
                        resolved_ips=((parsed.hostname or ""),),
                    )
                    request_url = current
                    extensions = {}
                if resolution.host not in permitted_redirect_hosts:
                    raise SourceEgressError(
                        "Cross-host source redirect is not permitted"
                    )
                logical_request_url = str(httpx.URL(current, params=current_params))
                _validate_url_shape(logical_request_url)
                hop_headers = _headers_for_hop(
                    current,
                    request_headers,
                    initial_origin=initial_origin,
                )
                async with client.stream(
                    "GET",
                    request_url,
                    params=current_params,
                    headers=hop_headers,
                    extensions=extensions,
                ) as response:
                    if transport is None:
                        _require_direct_peer(response, resolution, proxy)
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        if hop >= max_redirects:
                            raise SourceEgressError("Source redirect limit exceeded")
                        location = response.headers.get("location")
                        if not location:
                            raise SourceEgressError(
                                "Source redirect omitted its destination"
                            )
                        destination = urljoin(logical_request_url, location)
                        next_url = _validate_url_shape(destination)
                        next_host = require_safe_source_url(
                            destination,
                            allowed_private_hosts=private_hosts,
                            resolve_dns=False,
                        )
                        if next_host not in permitted_redirect_hosts:
                            raise SourceEgressError(
                                "Cross-host source redirect is not permitted"
                            )
                        if resolution.scheme == "https" and next_url.scheme == "http":
                            raise SourceEgressError(
                                "HTTPS source redirect downgrade is forbidden"
                            )
                        current = destination
                        current_params = {}
                        continue
                    response.raise_for_status()
                    return await _read_bounded_async(
                        response, max_bytes
                    ), response.encoding
    except httpx.HTTPError:
        raise SourceEgressError("Source fetch failed") from None
    finally:
        if tls is not None:
            tls.cleanup()
    raise SourceEgressError("Source fetch did not produce a response")


def safe_get_text(url: str, **kwargs: Any) -> str:
    body, encoding = safe_get_bytes(url, **kwargs)
    return body.decode(encoding or "utf-8", errors="replace")


def safe_get_json(url: str, **kwargs: Any) -> Any:
    body, encoding = safe_get_bytes(url, **kwargs)
    try:
        return json.loads(body.decode(encoding or "utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceEgressError("Source response was not valid JSON") from exc


async def safe_get_text_async(url: str, **kwargs: Any) -> str:
    body, encoding = await safe_get_bytes_async(url, **kwargs)
    return body.decode(encoding or "utf-8", errors="replace")


async def safe_get_json_async(url: str, **kwargs: Any) -> Any:
    body, encoding = await safe_get_bytes_async(url, **kwargs)
    try:
        return json.loads(body.decode(encoding or "utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceEgressError("Source response was not valid JSON") from exc


def _safe_write_json(
    method: str,
    url: str,
    payload: Any,
    *,
    send_body: bool,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    allowed_private_hosts: Iterable[str] | None = None,
    transport: Any = None,
    tls_service: str = "source-http",
) -> Any:
    """Issue one bounded, non-redirecting JSON state-change request."""
    if method not in {"POST", "DELETE"}:
        raise ValueError("Only POST and DELETE JSON requests are supported")
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    if not 1 <= max_request_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(
            f"max_request_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}"
        )
    request_body: bytes | None = None
    if send_body:
        try:
            request_body = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise SourceEgressError("Source request was not valid JSON") from exc
        if len(request_body) > max_request_bytes:
            raise SourceEgressError("Source request exceeded the configured limit")

    timeout = _validate_timeout(timeout)
    request_headers = _validated_headers(headers)
    private_hosts = normalize_allowed_hosts(allowed_private_hosts)
    require_safe_source_url(
        url,
        allowed_private_hosts=private_hosts,
        resolve_dns=False,
    )
    initial_origin = _logical_origin(url)
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Native HTTP source connectors require 'httpx'") from exc

    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": False,
    }
    tls = None
    proxy = None
    try:
        if transport is not None:
            client_kwargs["transport"] = transport
        else:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            tls = resolve_configured_tls_profile(tls_service)
            proxy = _pin_configured_proxy(tls.proxy_url)
            client_kwargs.update(_profile_httpx_kwargs(tls, proxy))

        from agent_utilities.core.http_client import create_http_client
    except BaseException:
        if tls is not None:
            tls.cleanup()
        raise

    try:
        with create_http_client(**client_kwargs) as client:
            if transport is None:
                resolution = resolve_safe_source_url(
                    url,
                    allowed_private_hosts=private_hosts,
                )
                _require_proxy_target_compatibility(proxy, resolution)
                request_url = _pinned_url(url, resolution.pinned_ip)
                extensions = _request_extensions(resolution)
            else:
                parsed = _validate_url_shape(url)
                resolution = ResolvedSourceURL(
                    url=url,
                    host=_normalize_hostname(parsed.hostname or ""),
                    scheme=parsed.scheme,
                    resolved_ips=((parsed.hostname or ""),),
                )
                request_url = url
                extensions = {}
            hop_headers = _headers_for_hop(
                url,
                request_headers,
                initial_origin=initial_origin,
            )
            if send_body:
                hop_headers.setdefault("Content-Type", "application/json")
            with client.stream(
                method,
                request_url,
                content=request_body,
                headers=hop_headers,
                extensions=extensions,
            ) as response:
                if transport is None:
                    _require_direct_peer(response, resolution, proxy)
                if response.status_code in _REDIRECT_STATUS_CODES:
                    raise SourceEgressError(
                        "State-changing source redirect is forbidden"
                    )
                response.raise_for_status()
                raw = _read_bounded(response, max_bytes)
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode(response.encoding or "utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SourceEgressError(
                        "Source response was not valid JSON"
                    ) from exc
    except httpx.HTTPError:
        raise SourceEgressError("Source request failed") from None
    finally:
        if tls is not None:
            tls.cleanup()


def safe_post_json(
    url: str,
    payload: Any,
    **kwargs: Any,
) -> Any:
    """POST JSON through the canonical bounded, DNS-pinned HTTP boundary."""
    return _safe_write_json("POST", url, payload, send_body=True, **kwargs)


def safe_delete_json(url: str, **kwargs: Any) -> Any:
    """DELETE through the canonical bounded, DNS-pinned HTTP boundary."""
    return _safe_write_json("DELETE", url, None, send_body=False, **kwargs)


async def safe_post_json_async(
    url: str,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    allowed_private_hosts: Iterable[str] | None = None,
    transport: Any = None,
    tls_service: str = "source-http",
) -> Any:
    """POST JSON to one validated endpoint and read one bounded JSON response.

    Redirects are rejected for state-changing requests. The destination is
    resolved immediately before I/O and uses the shared runtime TLS profile.
    """
    if not 1 <= max_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}")
    if not 1 <= max_request_bytes <= HARD_MAX_RESPONSE_BYTES:
        raise ValueError(
            f"max_request_bytes must be between 1 and {HARD_MAX_RESPONSE_BYTES}"
        )
    try:
        request_body = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise SourceEgressError("Source request was not valid JSON") from exc
    if len(request_body) > max_request_bytes:
        raise SourceEgressError("Source request exceeded the configured limit")
    timeout = _validate_timeout(timeout)
    request_headers = _validated_headers(headers)
    private_hosts = normalize_allowed_hosts(allowed_private_hosts)
    _ = require_safe_source_url(
        url,
        allowed_private_hosts=private_hosts,
        resolve_dns=False,
    )
    initial_origin = _logical_origin(url)
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Native HTTP source connectors require 'httpx'") from exc
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": False,
    }
    tls = None
    proxy = None
    try:
        if transport is not None:
            client_kwargs["transport"] = transport
        else:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            tls = resolve_configured_tls_profile(tls_service)
            proxy = _pin_configured_proxy(tls.proxy_url)
            client_kwargs.update(_profile_httpx_kwargs(tls, proxy))

        from agent_utilities.core.http_client import create_async_http_client
    except BaseException:
        if tls is not None:
            tls.cleanup()
        raise

    try:
        async with create_async_http_client(**client_kwargs) as client:
            if transport is None:
                resolution = resolve_safe_source_url(
                    url,
                    allowed_private_hosts=private_hosts,
                )
                _require_proxy_target_compatibility(proxy, resolution)
                request_url = _pinned_url(url, resolution.pinned_ip)
                extensions = _request_extensions(resolution)
            else:
                parsed = _validate_url_shape(url)
                resolution = ResolvedSourceURL(
                    url=url,
                    host=_normalize_hostname(parsed.hostname or ""),
                    scheme=parsed.scheme,
                    resolved_ips=((parsed.hostname or ""),),
                )
                request_url = url
                extensions = {}
            hop_headers = _headers_for_hop(
                url,
                request_headers,
                initial_origin=initial_origin,
            )
            hop_headers.setdefault("Content-Type", "application/json")
            async with client.stream(
                "POST",
                request_url,
                content=request_body,
                headers=hop_headers,
                extensions=extensions,
            ) as response:
                if transport is None:
                    _require_direct_peer(response, resolution, proxy)
                if response.status_code in _REDIRECT_STATUS_CODES:
                    raise SourceEgressError(
                        "State-changing source redirect is forbidden"
                    )
                response.raise_for_status()
                raw = await _read_bounded_async(response, max_bytes)
                if not raw:
                    return {}
                try:
                    value = json.loads(raw.decode(response.encoding or "utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SourceEgressError(
                        "Source response was not valid JSON"
                    ) from exc
                return value
    except httpx.HTTPError:
        raise SourceEgressError("Source request failed") from None
    finally:
        if tls is not None:
            tls.cleanup()


def configured_source_http_policy() -> dict[str, Any]:
    """Return the bounded, operator-configured policy for outbound source reads.

    Importing ``AgentConfig`` lazily keeps this low-level module usable by
    connector unit tests with injected transports while ensuring older fetch
    paths share the same deny-by-default private-host and redirect policy.
    """
    from agent_utilities.core.config import config

    return {
        "max_bytes": int(config.source_http_max_response_bytes),
        "max_redirects": int(config.source_http_max_redirects),
        "allowed_private_hosts": tuple(config.source_http_allowed_private_hosts),
        "allowed_redirect_hosts": tuple(config.source_http_allowed_redirect_hosts),
    }


__all__ = [
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "HARD_MAX_REDIRECTS",
    "HARD_MAX_RESPONSE_BYTES",
    "ResolvedSourceURL",
    "SourceEgressError",
    "configured_source_http_policy",
    "normalize_allowed_hosts",
    "require_safe_source_url",
    "resolve_safe_source_url",
    "safe_get_bytes",
    "safe_get_bytes_async",
    "safe_get_json",
    "safe_get_json_async",
    "safe_get_text",
    "safe_get_text_async",
    "safe_delete_json",
    "safe_post_json",
    "safe_post_json_async",
    "source_url_host",
]
