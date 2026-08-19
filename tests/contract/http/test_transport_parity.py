"""NE-015 parity contract for the staged HTTP adapters.

The contract is intentionally expressed once and run against both concrete
adapters.  It proves that a call family can move between the ``httpx`` and
``httpx2`` implementations without changing the application-visible response,
request kwargs, timeout/TLS guard, or transport-error taxonomy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import httpx2
import pytest

from agent_utilities.httpsupport.client_protocol import (
    HttpConnectError,
    HttpResponse,
)
from agent_utilities.httpsupport.httpx2_adapter import (
    AsyncHttpx2Adapter,
    Httpx2Adapter,
)
from agent_utilities.httpsupport.httpx_adapter import AsyncHttpxAdapter, HttpxAdapter


_KINDS = ("httpx", "httpx2")


def _sync_adapter(
    kind: str, handler: Callable[[Any], Any], **kwargs: Any
) -> HttpxAdapter | Httpx2Adapter:
    if kind == "httpx":
        return HttpxAdapter(transport=httpx.MockTransport(handler), **kwargs)
    return Httpx2Adapter(transport=httpx2.MockTransport(handler), **kwargs)


def _async_adapter(
    kind: str, handler: Callable[[Any], Any], **kwargs: Any
) -> AsyncHttpxAdapter | AsyncHttpx2Adapter:
    if kind == "httpx":
        return AsyncHttpxAdapter(transport=httpx.MockTransport(handler), **kwargs)
    return AsyncHttpx2Adapter(transport=httpx2.MockTransport(handler), **kwargs)


@pytest.mark.parametrize("kind", _KINDS)
def test_sync_adapters_share_request_response_contract(kind: str) -> None:
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        response_type = httpx.Response if kind == "httpx" else httpx2.Response
        return response_type(
            207,
            headers={"X-Transport-Contract": "v1"},
            json={"ok": True, "kind": kind},
        )

    client = _sync_adapter(
        kind,
        handler,
        timeout=7.0,
        headers={"X-Contract": "request"},
    )
    try:
        response = client.request(
            "POST",
            "https://example.test/contract?fixed=1",
            params={"page": 2},
            json={"value": 1},
            headers={"X-Call": "request"},
        )
    finally:
        client.close()

    assert isinstance(response, HttpResponse)
    assert response.status_code == 207
    assert response.headers["x-transport-contract"] == "v1"
    assert response.json() == {"ok": True, "kind": kind}
    assert seen["method"] == "POST"
    assert "fixed=1" in seen["url"] and "page=2" in seen["url"]
    assert seen["headers"]["x-contract"] == "request"
    assert seen["headers"]["x-call"] == "request"
    assert b'"value":1' in seen["body"]


@pytest.mark.parametrize("kind", _KINDS)
def test_sync_adapters_share_transport_error_contract(kind: str) -> None:
    def handler(request: Any) -> Any:
        error_type = httpx.ConnectError if kind == "httpx" else httpx2.ConnectError
        raise error_type("connection refused", request=request)

    client = _sync_adapter(kind, handler, timeout=7.0)
    try:
        with pytest.raises(HttpConnectError):
            client.request("GET", "https://example.test/down")
    finally:
        client.close()


@pytest.mark.parametrize("kind", _KINDS)
def test_sync_adapters_share_timeout_and_tls_guards(kind: str) -> None:
    adapter_type = HttpxAdapter if kind == "httpx" else Httpx2Adapter
    with pytest.raises(ValueError, match="finite timeout"):
        adapter_type(timeout=None)
    with pytest.raises(ValueError, match="TLS verification"):
        adapter_type(timeout=7.0, verify=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _KINDS)
async def test_async_adapters_share_request_response_contract(kind: str) -> None:
    def handler(request: Any) -> Any:
        response_type = httpx.Response if kind == "httpx" else httpx2.Response
        return response_type(200, text="ok")

    client = _async_adapter(kind, handler, timeout=7.0)
    try:
        response = await client.request("GET", "https://example.test/health")
    finally:
        await client.aclose()

    assert isinstance(response, HttpResponse)
    assert response.status_code == 200
    assert response.text == "ok"
