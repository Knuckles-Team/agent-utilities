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

GOC-83-W05 extends this to the remaining three hostile-input classes the
``engine_<domain>`` dispatcher (``engine_tools._dispatch``) accepts as DATA
rather than raising: an unknown/non-existent action name, a parameter
``TypeError`` (missing-required / wrong-type / duplicate-conflicting) when
calling the resolved EG method, and an undecodable ``params_json``.
``_dispatch`` deliberately never raises for these (every MCP tool caller must
keep getting a JSON string back, not an exception to unwrap) -- but the
generic REST twin used to wrap ANY such string in ``{"status": "success"}``
at HTTP 200, indistinguishable from a real result over REST.
``kg_server._is_engine_dispatch_client_error`` recognizes both shapes
``_dispatch`` emits for a caller mistake (a bare ``error: str`` for an
action-name rejection, or ``error.code == "invalid_request"`` for a
parameter/params_json rejection) and ``_make_tool_endpoint`` maps them to 400,
scoped to ``engine_*`` tool names only so no other tool's REST status-code
contract changes.
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

    def create(self, name):
        # A REQUIRED parameter (no default) — omitting it from params_json
        # makes the resolved EG method's own call raise TypeError, exercising
        # the "missing parameter" hostile-input case.
        self.calls.append(("create", (name,), {}))
        return {"name": name}


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


# ── GOC-83-W05: the remaining three hostile-input classes ──────────────────
# Each of these previously returned HTTP 200 with an error-shaped body hidden
# under `{"status": "success"}` -- never a 500, but never a structured 4xx
# either. Each test proves BOTH halves: `status_code != 500` (the ORIGINAL
# U-74 symptom never regresses) AND `status_code == 400` (the actual fix).


def test_rest_endpoint_maps_unknown_action_name_to_400(_fake_client):
    """Unknown/non-existent action name (not a field mismatch: a valid field,
    an invalid value) — `engine_tools._dispatch` returns `{"error": "unknown
    action ..."}` as data; the REST twin must not present that as success."""
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(handler(_FakeRequest({"action": "smuggle"})))

    assert response.status_code != 500
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["status"] == "failed"
    assert "smuggle" in body["result"]["error"]
    assert _fake_client.calls == []


def test_rest_endpoint_maps_missing_required_engine_parameter_to_400(_fake_client):
    """`action="create"` needs `name`; an empty `params_json` omits it, so the
    resolved EG method call raises `TypeError` for a missing required arg —
    caught by `_dispatch`, never reaches the engine (pre-transport)."""
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(
        handler(_FakeRequest({"action": "create", "params_json": "{}"}))
    )

    assert response.status_code != 500
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["status"] == "failed"
    assert body["result"]["error"]["code"] == "invalid_request"
    assert _fake_client.calls == []


def test_rest_endpoint_maps_wrong_type_params_json_to_400(_fake_client):
    """`params_json` must be a JSON-encoded object; a non-string value fails
    `json.loads` with a `TypeError` before any parameter is ever inspected."""
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(
        handler(_FakeRequest({"action": "create", "params_json": 42}))
    )

    assert response.status_code != 500
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["status"] == "failed"
    assert body["result"]["error"]["code"] == "invalid_request"
    assert _fake_client.calls == []


def test_rest_endpoint_maps_duplicate_graph_selector_conflict_to_400(_fake_client):
    """The wrapper's top-level `graph` selector (connection/session binding)
    and a `graph` key inside `params_json` (forwarded straight to the
    resolved EG method) are two different channels for a same-named field.
    Supplying BOTH — `graph="graph-A"` at the wrapper level and
    `{"graph": "graph-B"}` inside `params_json` — skips the wrapper's own
    graph-injection (params already has the key) and forwards `graph` to
    `tenants.list()`, which doesn't accept it: a `TypeError` for a
    caller-duplicated parameter, caught by `_dispatch` the same way."""
    kg_server.ensure_tools_registered()
    handler = kg_server._make_tool_endpoint("engine_tenants")

    response = asyncio.run(
        handler(
            _FakeRequest(
                {
                    "action": "list",
                    "graph": "graph-A",
                    "params_json": json.dumps({"graph": "graph-B"}),
                }
            )
        )
    )

    assert response.status_code != 500
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["status"] == "failed"
    assert body["result"]["error"]["code"] == "invalid_request"
    assert _fake_client.calls == []


def test_engine_dispatch_client_error_scoping_is_narrow(_fake_client):
    """The 400-mapping must be scoped to `engine_*` tool names only — proves
    the recognizer function directly rather than relying on there being no
    non-`engine_*` tool in the suite that happens to return the same shape."""
    assert kg_server._is_engine_dispatch_client_error(
        {"error": "unknown action 'x' for engine_tenants", "actions": []}
    )
    assert kg_server._is_engine_dispatch_client_error(
        {"error": {"code": "invalid_request"}}
    )
    # A real result, a listing response, and a non-client-error payload must
    # NOT be reclassified as a 4xx.
    assert not kg_server._is_engine_dispatch_client_error(
        [{"name": "__commons__"}]
    )
    assert not kg_server._is_engine_dispatch_client_error(
        {"domain": "tenants", "actions": ["list"], "admin_domain": False}
    )
    assert not kg_server._is_engine_dispatch_client_error(
        {"error": {"code": "dependency_unavailable"}}
    )
