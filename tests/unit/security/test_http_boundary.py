from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_utilities.security.http_boundary import (
    AuthenticationBoundaryMiddleware,
    BoundedRequestBodyMiddleware,
    ExactHostAuthorityMiddleware,
    OriginPolicyMiddleware,
    TrustedProxyPeerMiddleware,
    normalize_host_authorities,
    normalize_origins,
)


async def _capture_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    del receive
    await send(
        {
            "type": "http.response.start",
            "status": 204,
            "headers": [(b"x-identity", str(scope.get("state", {})).encode())],
        }
    )
    await send({"type": "http.response.body", "body": b""})


async def _messages(*messages: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for message in messages:
        yield message


async def _invoke(
    app: Any,
    scope: dict[str, Any],
    *incoming: dict[str, Any],
) -> list[dict[str, Any]]:
    iterator = _messages(*incoming)
    output: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return await anext(iterator)

    async def send(message: dict[str, Any]) -> None:
        output.append(message)

    await app(scope, receive, send)
    return output


def _http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    return {
        "type": "http",
        "method": "POST",
        "path": "/mounted/tool",
        "headers": headers or [],
        "client": ("127.0.0.1", 30000),
    }


def test_normalize_origins_rejects_wildcards_and_paths() -> None:
    with pytest.raises(ValueError):
        normalize_origins(["*"])
    with pytest.raises(ValueError):
        normalize_origins(["https://service.invalid/path"])


def test_normalize_host_authorities_retains_port() -> None:
    assert normalize_host_authorities(["Service.Invalid:8443", "[::1]:8100"]) == {
        "service.invalid:8443",
        "[::1]:8100",
    }
    with pytest.raises(ValueError):
        normalize_host_authorities(["*.invalid"])


@pytest.mark.anyio
async def test_host_authority_requires_exact_port_and_single_header() -> None:
    app = ExactHostAuthorityMiddleware(_capture_app, ["127.0.0.1:8100"])
    accepted = await _invoke(
        app,
        _http_scope([(b"host", b"127.0.0.1:8100")]),
    )
    assert accepted[0]["status"] == 204

    wrong_port = await _invoke(
        app,
        _http_scope([(b"host", b"127.0.0.1:8101")]),
    )
    assert wrong_port[0]["status"] == 400

    duplicate = await _invoke(
        app,
        _http_scope([(b"host", b"127.0.0.1:8100"), (b"host", b"127.0.0.1:8100")]),
    )
    assert duplicate[0]["status"] == 400


@pytest.mark.anyio
async def test_origin_policy_rejects_websocket_before_accept() -> None:
    async def websocket_app(scope: Any, receive: Any, send: Any) -> None:
        del scope, receive
        await send({"type": "websocket.accept"})

    app = OriginPolicyMiddleware(
        websocket_app, allowed_origins=["https://allowed.invalid"]
    )
    output = await _invoke(
        app,
        {
            "type": "websocket",
            "path": "/events",
            "headers": [(b"origin", b"https://rejected.invalid")],
            "client": ("127.0.0.1", 30000),
        },
    )
    assert output == [
        {"type": "websocket.close", "code": 4403, "reason": "origin rejected"}
    ]


@pytest.mark.anyio
async def test_body_boundary_rejects_duplicate_length() -> None:
    app = BoundedRequestBodyMiddleware(_capture_app, max_bytes=1024)
    output = await _invoke(
        app,
        _http_scope([(b"content-length", b"0"), (b"content-length", b"0")]),
    )
    assert output[0]["status"] == 400
    assert (b"cache-control", b"no-store") in output[0]["headers"]


@pytest.mark.anyio
async def test_body_boundary_rejects_stream_over_limit() -> None:
    app = BoundedRequestBodyMiddleware(_capture_app, max_bytes=1024)
    output = await _invoke(
        app,
        _http_scope(),
        {"type": "http.request", "body": b"x" * 1025, "more_body": False},
    )
    assert output[0]["status"] == 413


@pytest.mark.anyio
async def test_body_boundary_waits_for_real_stream_disconnect() -> None:
    request_messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await request_messages.put(
        {"type": "http.request", "body": b"{}", "more_body": False}
    )
    downstream_waiting = asyncio.Event()
    output: list[dict[str, Any]] = []

    async def streaming_app(scope: Any, receive: Any, send: Any) -> None:
        del scope
        request = await receive()
        assert request == {
            "type": "http.request",
            "body": b"{}",
            "more_body": False,
        }
        downstream_waiting.set()
        assert await receive() == {"type": "http.disconnect"}
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def original_receive() -> dict[str, Any]:
        return await request_messages.get()

    async def send(message: dict[str, Any]) -> None:
        output.append(message)

    app = BoundedRequestBodyMiddleware(streaming_app, max_bytes=1024)
    task = asyncio.create_task(app(_http_scope(), original_receive, send))
    await downstream_waiting.wait()
    await asyncio.sleep(0)
    assert not task.done()
    await request_messages.put({"type": "http.disconnect"})
    await task
    assert output[0]["status"] == 204


@pytest.mark.anyio
async def test_trusted_ingress_checks_immediate_peer() -> None:
    app = TrustedProxyPeerMiddleware(_capture_app, ["192.0.2.0/24"])
    scope = _http_scope()
    scope["client"] = ("198.51.100.10", 30000)
    output = await _invoke(app, scope)
    assert output[0]["status"] == 403


@pytest.mark.anyio
async def test_auth_boundary_rejects_duplicate_credentials(monkeypatch: Any) -> None:
    async def fake_authenticate_header_values(**kwargs: Any) -> None:
        assert len(kwargs["authorization"]) == 2
        raise PermissionError("invalid credentials")

    monkeypatch.setattr(
        "agent_utilities.security.auth.authenticate_header_values",
        fake_authenticate_header_values,
    )
    app = AuthenticationBoundaryMiddleware(_capture_app)
    output = await _invoke(
        app,
        _http_scope([(b"authorization", b"Bearer a"), (b"authorization", b"Bearer b")]),
    )
    assert output[0]["status"] == 401
    assert (b"cache-control", b"no-store") in output[0]["headers"]


@pytest.mark.anyio
async def test_auth_boundary_propagates_validated_identity(monkeypatch: Any) -> None:
    async def fake_authenticate_header_values(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"auth_type": "jwt", "subject_ref": "pref_subject_" + "0" * 64}

    monkeypatch.setattr(
        "agent_utilities.security.auth.authenticate_header_values",
        fake_authenticate_header_values,
    )
    app = AuthenticationBoundaryMiddleware(_capture_app)
    output = await _invoke(
        app,
        _http_scope([(b"authorization", b"Bearer opaque-token")]),
    )
    assert output[0]["status"] == 204
    assert b"pref_subject_" in dict(output[0]["headers"])[b"x-identity"]
