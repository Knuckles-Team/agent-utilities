"""GOC-87 staged httpx -> httpx2 migration: neutral protocol + adapters + factory.

Drives the REAL adapters against a fake local transport for each backing
package (``httpx.MockTransport`` / ``httpx2.MockTransport``) — never mocks
the adapter/protocol seam itself — so these tests prove the boundary
actually round-trips a request/response and maps a real transport failure
onto the AU taxonomy, for both packages.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import httpx2
import pytest

from agent_utilities.httpsupport.client_protocol import (
    HttpConnectError,
    HttpResponse,
    HttpTimeoutError,
    HttpTransportError,
    map_transport_error,
    normalize_response,
)
from agent_utilities.httpsupport.httpx2_adapter import (
    AsyncHttpx2Adapter,
    Httpx2Adapter,
)
from agent_utilities.httpsupport.httpx_adapter import AsyncHttpxAdapter, HttpxAdapter
from agent_utilities.httpsupport.transport_factory import (
    MIGRATED_HTTPX2_FAMILIES,
    create_async_http_client,
    create_http_client,
)

# --------------------------------------------------------------------------
# Static boundary check — neither adapter module even IMPORTS the other
# package, proving one package's concrete client/transport object cannot be
# constructed by, or passed into, the other's adapter (GOC-87 authority #4).
# --------------------------------------------------------------------------

_ADAPTERS_DIR = Path(__file__).resolve().parents[3] / "agent_utilities" / "httpsupport"


def _top_level_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_httpx_adapter_module_never_imports_httpx2():
    imports = _top_level_imports(_ADAPTERS_DIR / "httpx_adapter.py")
    assert "httpx2" not in imports


def test_httpx2_adapter_module_never_imports_httpx():
    imports = _top_level_imports(_ADAPTERS_DIR / "httpx2_adapter.py")
    assert "httpx" not in imports


def test_client_protocol_module_imports_neither_concrete_package():
    """The neutral protocol/taxonomy module has zero dependency on either
    package — a third adapter (a third HTTP client) could implement it
    without ever importing httpx or httpx2."""
    imports = _top_level_imports(_ADAPTERS_DIR / "client_protocol.py")
    assert "httpx" not in imports
    assert "httpx2" not in imports


# --------------------------------------------------------------------------
# normalize_response / HttpResponse — no concrete transport object leaks
# --------------------------------------------------------------------------


def test_normalize_response_from_httpx_leaks_no_concrete_type():
    raw = httpx.Response(200, json={"ok": 1}, headers={"X-Test": "1"})
    normalized = normalize_response(raw)

    assert isinstance(normalized, HttpResponse)
    assert not isinstance(normalized, httpx.Response)
    assert normalized.status_code == 200
    assert normalized.headers["x-test"] == "1"
    assert normalized.json() == {"ok": 1}


def test_normalize_response_from_httpx2_leaks_no_concrete_type():
    raw = httpx2.Response(201, json={"family": "httpx2"})
    normalized = normalize_response(raw)

    assert isinstance(normalized, HttpResponse)
    assert not isinstance(normalized, httpx2.Response)
    assert normalized.status_code == 201
    assert normalized.json() == {"family": "httpx2"}


# --------------------------------------------------------------------------
# map_transport_error — both packages' exceptions land on the same taxonomy
# --------------------------------------------------------------------------


def test_map_transport_error_unifies_httpx_and_httpx2_timeout():
    httpx_exc = httpx.ConnectTimeout("boom")
    httpx2_exc = httpx2.ConnectTimeout("boom")

    mapped_a = map_transport_error(httpx_exc)
    mapped_b = map_transport_error(httpx2_exc)

    assert type(mapped_a) is type(mapped_b) is HttpTimeoutError
    assert isinstance(mapped_a, HttpTransportError)


def test_map_transport_error_unifies_httpx_and_httpx2_connect_error():
    mapped_a = map_transport_error(httpx.ConnectError("refused"))
    mapped_b = map_transport_error(httpx2.ConnectError("refused"))

    assert type(mapped_a) is type(mapped_b) is HttpConnectError


def test_map_transport_error_unknown_exception_falls_back_to_base_taxonomy():
    class SomeOtherTransportFailure(Exception):
        pass

    mapped = map_transport_error(SomeOtherTransportFailure("weird"))
    assert type(mapped) is HttpTransportError


# --------------------------------------------------------------------------
# HttpxAdapter — wraps core.http_client unchanged
# --------------------------------------------------------------------------


def test_httpx_adapter_round_trips_request():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"seen": request.url.path})
    )
    adapter = HttpxAdapter(timeout=5.0, transport=transport)
    try:
        resp = adapter.request("GET", "https://example.test/api/tags")
        assert resp.status_code == 200
        assert resp.json() == {"seen": "/api/tags"}
    finally:
        adapter.close()


def test_httpx_adapter_maps_transport_error_to_taxonomy():
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(_raise)
    adapter = HttpxAdapter(timeout=5.0, transport=transport)
    try:
        with pytest.raises(HttpConnectError):
            adapter.request("GET", "https://example.test/down")
    finally:
        adapter.close()


async def test_async_httpx_adapter_round_trips_request():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    adapter = AsyncHttpxAdapter(timeout=5.0, transport=transport)
    try:
        resp = await adapter.request("GET", "https://example.test/health")
        assert resp.status_code == 200
        assert resp.text == "ok"
    finally:
        await adapter.aclose()


# --------------------------------------------------------------------------
# Httpx2Adapter — parallel behavior on the httpx2 package
# --------------------------------------------------------------------------


def test_httpx2_adapter_round_trips_request():
    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(200, json={"seen": request.url.path})
    )
    adapter = Httpx2Adapter(timeout=5.0, transport=transport)
    try:
        resp = adapter.request("GET", "https://example.test/api/ps")
        assert resp.status_code == 200
        assert resp.json() == {"seen": "/api/ps"}
    finally:
        adapter.close()


def test_httpx2_adapter_maps_transport_error_to_taxonomy():
    def _raise(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    transport = httpx2.MockTransport(_raise)
    adapter = Httpx2Adapter(timeout=5.0, transport=transport)
    try:
        with pytest.raises(HttpConnectError):
            adapter.request("GET", "https://example.test/down")
    finally:
        adapter.close()


def test_httpx2_adapter_rejects_infinite_timeout():
    with pytest.raises(ValueError):
        Httpx2Adapter(timeout=None)


def test_httpx2_adapter_rejects_disabled_tls_verification():
    with pytest.raises(ValueError):
        Httpx2Adapter(timeout=5.0, verify=False)


async def test_async_httpx2_adapter_round_trips_request():
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, text="ok"))
    adapter = AsyncHttpx2Adapter(timeout=5.0, transport=transport)
    try:
        resp = await adapter.request("GET", "https://example.test/health")
        assert resp.status_code == 200
        assert resp.text == "ok"
    finally:
        await adapter.aclose()


# --------------------------------------------------------------------------
# transport_factory — per-family selection; unmigrated families are unaffected
# --------------------------------------------------------------------------


def test_unmigrated_family_gets_httpx_adapter():
    client = create_http_client(family="some-unmigrated-family", timeout=5.0)
    try:
        assert isinstance(client, HttpxAdapter)
    finally:
        client.close()


def test_migrated_family_gets_httpx2_adapter():
    (family,) = MIGRATED_HTTPX2_FAMILIES
    client = create_http_client(family=family, timeout=5.0)
    try:
        assert isinstance(client, Httpx2Adapter)
    finally:
        client.close()


async def test_async_migrated_family_gets_httpx2_adapter():
    (family,) = MIGRATED_HTTPX2_FAMILIES
    client = create_async_http_client(family=family, timeout=5.0)
    try:
        assert isinstance(client, AsyncHttpx2Adapter)
    finally:
        await client.aclose()
