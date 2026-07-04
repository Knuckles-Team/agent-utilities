"""CONCEPT:AU-ORCH.adapter.byok-provider-proxy / OS-5.3 — Egress (SSRF) guard for operator-supplied endpoints.

Assimilated from open-design's ``validateBaseUrl`` / ``validateBaseUrlResolved`` (connectionTest.ts):
when a BYOK/custom ``base_url`` is accepted, the host is **DNS-resolved** and every resolved IP is
checked against a private/reserved blocklist *before any outbound request* — closing the common
public-DNS→private-IP SSRF vector that hostname-literal checks miss. Loopback is allowed (configurable)
so local LLMs (Ollama/vLLM/LM Studio) keep working.

Pure stdlib; no new deps. Used by the provider proxy router and reusable anywhere a custom URL is
accepted (webhooks, connectors).
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(slots=True)
class EgressDecision:
    """Outcome of an egress check."""

    allowed: bool
    reason: str = ""
    resolved_ips: tuple[str, ...] = ()


def _ip_is_blocked(ip: str, *, allow_loopback: bool) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → block
    if addr.is_loopback:
        return not allow_loopback
    # Block private, link-local, reserved, multicast, unspecified, and the cloud metadata IP.
    if (
        addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    ):
        return True
    if str(addr) in {"169.254.169.254", "fd00:ec2::254"}:  # cloud metadata service
        return True
    return False


def validate_base_url(url: str, *, allow_loopback: bool = True) -> EgressDecision:
    """Syntactic check: scheme + host present, and an IP *literal* host is not in a blocked range.

    Does not perform DNS resolution — use :func:`validate_base_url_resolved` for that.
    """
    if not url or not isinstance(url, str):
        return EgressDecision(False, "empty url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return EgressDecision(False, f"unsupported scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        return EgressDecision(False, "missing host")
    # If the host is an IP literal, check it directly.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return EgressDecision(True, "hostname (resolve to verify)")
    if _ip_is_blocked(host, allow_loopback=allow_loopback):
        return EgressDecision(False, f"blocked IP literal: {host}", (host,))
    return EgressDecision(True, "allowed IP literal", (host,))


def validate_base_url_resolved(
    url: str, *, allow_loopback: bool = True, resolver=None
) -> EgressDecision:
    """Full check: DNS-resolve the host and reject if **any** resolved IP is blocked.

    ``resolver`` is injectable for tests: a callable ``host -> list[(family, ..., sockaddr)]`` matching
    :func:`socket.getaddrinfo`'s return shape (defaults to ``socket.getaddrinfo``).
    """
    syntactic = validate_base_url(url, allow_loopback=allow_loopback)
    if not syntactic.allowed:
        return syntactic
    if syntactic.resolved_ips:  # was an IP literal already validated above
        return syntactic

    host = urlparse(url).hostname or ""
    getaddrinfo = resolver or socket.getaddrinfo
    try:
        infos = getaddrinfo(host, None)
    except OSError as exc:
        return EgressDecision(False, f"DNS resolution failed: {exc}")
    ips: list[str] = []
    for info in infos:
        sockaddr = info[4]
        ips.append(str(sockaddr[0]))
    if not ips:
        return EgressDecision(False, "no addresses resolved")
    for ip in ips:
        if _ip_is_blocked(ip, allow_loopback=allow_loopback):
            return EgressDecision(False, f"resolves to blocked IP: {ip}", tuple(ips))
    return EgressDecision(True, "all resolved IPs allowed", tuple(ips))
