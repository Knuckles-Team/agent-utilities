"""U-74: the generic ``/engine/<domain>`` REST twin forwards the ENTIRE JSON
request body as kwargs into the target MCP tool with no schema validation
(``kg_server._make_tool_endpoint`` -> ``_execute_tool(tool_name, **body)`` ->
``tool_func(**kwargs)``). ``engine_tenants``'s generated tool
(``engine_tools._make_domain_tool``) only accepts ``action``/``params_json``/
``graph`` -- it has no ``connection`` selector, unlike the query/write tools.

Live symptom: ``POST /engine/tenants`` with
``{"action": "list", "connection": "default"}`` raised a raw
``TypeError: _engine_domain_tool() got an unexpected keyword argument
'connection'`` that surfaced as an opaque HTTP 500 -- indistinguishable from a
genuine server fault, even though this is a deterministic client-side schema
mismatch every time.

The fix validates ``kwargs`` against the target tool's signature INSIDE
``_execute_tool`` (the one chokepoint both the REST and MCP surfaces dispatch
through) before the call is ever made, raising ``UnsupportedToolFieldError``;
the generic REST endpoint factory maps that specific exception to a
deterministic HTTP 400 instead of the default 500. Action-only calls, and any
call that supplies only fields the tool actually declares, are unaffected.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("epistemic_graph.client")

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools


class _FakeRequest:
    """The generic handler's only interaction with its ``Request`` argument is
    ``await request.json()`` -- stand in for a real Starlette ``Request``
    rather than constructing a full ASGI scope."""

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


class _RecordingTenants:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def list(self):
        self.calls.append(("list", (), {}))
        return [{"name": "__commons__"}, {"name": "kf-pilot:dlx"}]


class _RecordingClient:
    def __init__(self, tenants: _RecordingTenants) -> None:
        self.tenants = tenants


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


@pytest.fixture
def _fake_client(monkeypatch):
    tenants = _RecordingTenants()
    client = _RecordingClient(tenants)
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    return tenants


def test_unknown_field_is_rejected_before_any_dispatch(_fake_client):
    """The exact live reproduction: connection= alongside action=list."""
    kg_server.ensure_tools_registered()

    with pytest.raises(kg_server.UnsupportedToolFieldError, match="connection"):
        asyncio.run(
            kg_server._execute_tool(
                "engine_tenants", action="list", connection="default"
            )
        )

    # The deterministic point of the fix: rejection happens BEFORE the
    # underlying engine client is ever touched -- no partial/leaked call.
    assert _fake_client.calls == []


def test_action_only_call_still_succeeds(_fake_client):
    """The documented safe-mitigation shape must keep working unchanged."""
    kg_server.ensure_tools_registered()

    out = json.loads(
        asyncio.run(kg_server._execute_tool("engine_tenants", action="list"))
    )
    assert out == [{"name": "__commons__"}, {"name": "kf-pilot:dlx"}]
    assert _fake_client.calls == [("list", (), {})]


def test_rest_endpoint_maps_unknown_field_to_deterministic_400(_fake_client):
    """End-to-end through the actual generic REST factory: no more opaque 500
    for a deterministic client-side mismatch."""
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(
        handler(_FakeRequest({"action": "list", "connection": "default"}))
    )

    assert response.status_code == 400
    assert _fake_client.calls == []


def test_rest_endpoint_action_only_still_returns_200(_fake_client):
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(handler(_FakeRequest({"action": "list"})))

    assert response.status_code == 200
    assert _fake_client.calls == [("list", (), {})]
