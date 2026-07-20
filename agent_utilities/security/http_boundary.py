"""Reusable ASGI trust-boundary middleware for served HTTP transports.

The application routers and mounted protocol applications are implementation
details behind one network boundary.  These small, dependency-light
middlewares therefore run at ASGI scope level, before route matching, so a
``Mount`` cannot accidentally bypass authentication, origin checks, request
bounds, or a trusted-ingress peer restriction.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

import anyio

_MAX_HEADER_VALUE_BYTES = 16_384


def _header_values(scope: Mapping[str, Any], name: bytes) -> list[bytes]:
    target = name.lower()
    return [
        bytes(value)
        for key, value in scope.get("headers", ())
        if bytes(key).lower() == target
    ]


async def _json_response(
    send: Any,
    status: int,
    error: str,
    *,
    headers: Iterable[tuple[bytes, bytes]] = (),
) -> None:
    body = json.dumps({"error": error}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                *headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def normalize_host_authorities(authorities: Iterable[str]) -> frozenset[str]:
    """Validate exact HTTP Host authorities, including an optional port.

    Starlette's generic trusted-host middleware discards the port before it
    compares an allowlist.  GraphOS configuration promises exact authorities,
    so host-header validation must retain that boundary instead of silently
    widening ``service.example:8443`` to every port on ``service.example``.
    """

    normalized: set[str] = set()
    for raw in authorities:
        value = str(raw or "").strip()
        if (
            not value
            or len(value.encode("utf-8")) > _MAX_HEADER_VALUE_BYTES
            or any(character in value for character in "/\\@?#\r\n\t ")
            or (value.count(":") > 1 and not value.startswith("["))
        ):
            raise ValueError("host allowlist must contain exact authorities")
        try:
            parsed = urlsplit(f"//{value}")
            host = str(parsed.hostname or "").rstrip(".").casefold()
            port = parsed.port
        except ValueError:
            raise ValueError("host allowlist must contain exact authorities") from None
        if (
            not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("host allowlist must contain exact authorities")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                rendered_host = host.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("host allowlist contains an invalid name") from exc
            labels = rendered_host.split(".")
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
                raise ValueError("host allowlist contains an invalid name") from None
        else:
            rendered_host = (
                f"[{address.compressed}]"
                if isinstance(address, ipaddress.IPv6Address)
                else address.compressed
            )
        normalized.add(f"{rendered_host}:{port}" if port is not None else rendered_host)
    if not normalized or len(normalized) > 256:
        raise ValueError("host allowlist must contain 1..256 exact authorities")
    return frozenset(normalized)


class ExactHostAuthorityMiddleware:
    """Reject missing, duplicate, malformed, or non-allowlisted Host headers."""

    def __init__(self, app: Any, allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self.allowed_hosts = normalize_host_authorities(allowed_hosts)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        values = _header_values(scope, b"host")
        candidate: str | None = None
        if len(values) == 1 and len(values[0]) <= _MAX_HEADER_VALUE_BYTES:
            try:
                rendered = values[0].decode("ascii")
                parsed = normalize_host_authorities([rendered])
                candidate = next(iter(parsed))
            except (UnicodeDecodeError, ValueError, StopIteration):
                candidate = None
        if candidate in self.allowed_hosts:
            await self.app(scope, receive, send)
        elif scope.get("type") == "websocket":
            await send(
                {"type": "websocket.close", "code": 4403, "reason": "host rejected"}
            )
        else:
            await _json_response(send, 400, "host rejected")


def normalize_origins(origins: Iterable[str]) -> frozenset[str]:
    """Validate and normalize exact browser origins.

    Wildcards, user-info, path prefixes and opaque ``null`` origins are not
    accepted.  An origin is a scheme/authority tuple, not a URL prefix.
    """
    normalized: set[str] = set()
    for raw in origins:
        value = str(raw or "").strip()
        if not value:
            continue
        if value == "*" or len(value) > 2_048:
            raise ValueError("origins must be exact HTTP(S) authorities")
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("origins must be exact HTTP(S) authorities")
        host = parsed.hostname.lower().rstrip(".")
        try:
            rendered_host = f"[{ipaddress.ip_address(host).compressed}]"
        except ValueError:
            try:
                rendered_host = host.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("origin hostname is invalid") from exc
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        port = parsed.port
        authority = (
            rendered_host if port in {None, default_port} else f"{rendered_host}:{port}"
        )
        normalized.add(f"{parsed.scheme.lower()}://{authority}")
    if len(normalized) > 256:
        raise ValueError("origin allowlist is too large")
    return frozenset(normalized)


class OriginPolicyMiddleware:
    """Reject browser/WebSocket origins not explicitly allowlisted.

    CORS response headers alone do not stop a non-browser client and do not
    protect WebSockets.  This middleware enforces the inbound decision.  A
    request without ``Origin`` remains valid for CLI/service clients.
    """

    def __init__(self, app: Any, allowed_origins: Iterable[str] = ()) -> None:
        self.app = app
        self.allowed_origins = normalize_origins(allowed_origins)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        values = _header_values(scope, b"origin")
        if not values:
            await self.app(scope, receive, send)
            return
        if len(values) != 1 or len(values[0]) > _MAX_HEADER_VALUE_BYTES:
            allowed = False
        else:
            try:
                candidate = next(iter(normalize_origins([values[0].decode("ascii")])))
            except (UnicodeDecodeError, ValueError, StopIteration):
                allowed = False
            else:
                allowed = candidate in self.allowed_origins
        if allowed:
            await self.app(scope, receive, send)
        elif scope.get("type") == "websocket":
            await send(
                {"type": "websocket.close", "code": 4403, "reason": "origin rejected"}
            )
        else:
            await _json_response(send, 403, "origin rejected")


class BoundedRequestBodyMiddleware:
    """Buffer and bound request bodies before any application code sees them."""

    def __init__(
        self,
        app: Any,
        max_bytes: int,
        *,
        read_timeout_seconds: float = 30.0,
    ) -> None:
        if not 1_024 <= int(max_bytes) <= 256 * 1024 * 1024:
            raise ValueError("request body limit is outside the safe range")
        if not 1.0 <= float(read_timeout_seconds) <= 300.0:
            raise ValueError("request body timeout is outside the safe range")
        self.app = app
        self.max_bytes = int(max_bytes)
        self.read_timeout_seconds = float(read_timeout_seconds)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        lengths = _header_values(scope, b"content-length")
        if len(lengths) > 1:
            await _json_response(send, 400, "invalid content length")
            return
        if lengths:
            try:
                declared = int(lengths[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                await _json_response(send, 400, "invalid content length")
                return
            if declared < 0 or declared > self.max_bytes:
                await _json_response(send, 413, "request body too large")
                return

        chunks: list[bytes] = []
        total = 0
        disconnected = False
        try:
            with anyio.fail_after(self.read_timeout_seconds):
                while True:
                    message = await receive()
                    if message.get("type") == "http.disconnect":
                        disconnected = True
                        break
                    body = bytes(message.get("body", b""))
                    total += len(body)
                    if total > self.max_bytes:
                        await _json_response(send, 413, "request body too large")
                        return
                    if body:
                        chunks.append(body)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await _json_response(send, 408, "request body timeout")
            return

        buffered = b"".join(chunks)
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                if disconnected:
                    return {"type": "http.disconnect"}
                return {"type": "http.request", "body": buffered, "more_body": False}
            # Streaming ASGI applications keep a concurrent receive waiter so
            # they can cancel work if the real client disconnects.  Fabricating
            # a disconnect immediately after replaying the buffered body aborts
            # a healthy Streamable HTTP response before its final body frame.
            # Delegate subsequent reads to the original channel instead.
            return await receive()

        await self.app(scope, replay, send)


def parse_cidrs(
    values: Iterable[str],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        try:
            network = ipaddress.ip_network(value, strict=True)
        except ValueError as exc:
            raise ValueError("trusted proxy CIDR is invalid") from exc
        networks.append(network)
    if not networks or len(networks) > 64:
        raise ValueError("trusted proxy CIDRs must contain 1..64 exact networks")
    return tuple(networks)


class TrustedProxyPeerMiddleware:
    """Accept requests only from a configured TLS-terminating ingress peer."""

    def __init__(self, app: Any, trusted_cidrs: Iterable[str]) -> None:
        self.app = app
        self.networks = parse_cidrs(trusted_cidrs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        try:
            peer = ipaddress.ip_address(str(client[0]))
        except (IndexError, TypeError, ValueError):
            allowed = False
        else:
            allowed = any(peer in network for network in self.networks)
        if allowed:
            await self.app(scope, receive, send)
        elif scope.get("type") == "websocket":
            await send(
                {"type": "websocket.close", "code": 4403, "reason": "peer rejected"}
            )
        else:
            await _json_response(send, 403, "peer rejected")


class AuthenticationBoundaryMiddleware:
    """Authenticate every route and mounted ASGI app at the outer boundary."""

    def __init__(self, app: Any, exempt_paths: Iterable[str] = ()) -> None:
        self.app = app
        self.exempt_paths = frozenset(str(path) for path in exempt_paths)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if (
            scope.get("type") == "http"
            and scope.get("method") == "OPTIONS"
            and len(_header_values(scope, b"origin")) == 1
            and len(_header_values(scope, b"access-control-request-method")) == 1
        ):
            # Browser preflight carries no bearer credential. Origin
            # and Host policy still run outside this middleware, while CORS
            # decides whether the requested method/headers are acceptable.
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        from agent_utilities.security.auth import authenticate_header_values

        try:
            identity = await authenticate_header_values(
                authorization=_header_values(scope, b"authorization"),
            )
        except PermissionError:
            if scope.get("type") == "websocket":
                await send(
                    {
                        "type": "websocket.close",
                        "code": 4401,
                        "reason": "authentication required",
                    }
                )
            else:
                await _json_response(
                    send,
                    401,
                    "authentication required",
                    headers=((b"www-authenticate", b"Bearer"),),
                )
            return

        state = scope.setdefault("state", {})
        if identity is not None:
            state["user_claims"] = identity
        await self.app(scope, receive, send)


__all__ = [
    "AuthenticationBoundaryMiddleware",
    "BoundedRequestBodyMiddleware",
    "ExactHostAuthorityMiddleware",
    "OriginPolicyMiddleware",
    "TrustedProxyPeerMiddleware",
    "normalize_host_authorities",
    "normalize_origins",
    "parse_cidrs",
]
