"""REST twin of the MCP fleet catalog meta-tools (GOC-60-W03).

CONCEPT:AU-ECO.mcp.catalog-rest-surface

``agent_utilities/mcp/multiplexer.py`` computes the fleet's dispatchable truth
(``list_catalog``, ``multiplexer_status``) for the ``find_tools``/
``load_tools``/``multiplexer_status`` MCP meta-tools but had NO REST route at
all — a violation of this repository's own "Two surfaces by default" rule
(GOC-60 lane, evidence E5). These tests cover the REST twin added by this
lane: authorized, unauthorized, and degraded-multiplexer cases, plus a parity
test proving the REST payload equals the direct multiplexer (MCP-tool-side)
payload for the same shared instance.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.mcp import shared_multiplexer as shared_mux_mod
from agent_utilities.server.routers import mcp_catalog


class _RecordingClaimsMiddleware:
    """Injects ``request.state.user_claims`` the way the real authentication
    boundary does after verifying a credential, mirroring the pattern used by
    ``agent-webui``'s ``__tests__/test_mcp_delegation_routes.py``."""

    def __init__(self, app, claims: dict[str, Any] | None):
        self._app = app
        self._claims = claims

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and self._claims is not None:
            scope = dict(scope)
            state = dict(scope.get("state") or {})
            state["user_claims"] = self._claims
            scope["state"] = state
        await self._app(scope, receive, send)


def _client(claims: dict[str, Any] | None) -> TestClient:
    app = FastAPI()
    app.include_router(mcp_catalog.router)
    wrapped = _RecordingClaimsMiddleware(app, claims)
    return TestClient(wrapped, raise_server_exceptions=False)


_DISCOVER_CLAIMS = {
    "auth_type": "jwt",
    "sub": "mcp-catalog-test",
    "scope": "mcp:discover",
}
_NO_SCOPE_CLAIMS = {
    "auth_type": "jwt",
    "sub": "mcp-catalog-test-unauth",
    "scope": "chat:read",
}


class _StubMultiplexer:
    """Deterministic stand-in with the same public surface the REST route
    calls (``list_catalog``, ``status_snapshot``)."""

    def __init__(self, *, fail_catalog: bool = False, fail_status: bool = False):
        self._fail_catalog = fail_catalog
        self._fail_status = fail_status

    async def list_catalog(self, server: str = "", include_tools: bool = True) -> dict:
        if self._fail_catalog:
            raise RuntimeError("catalog probe exploded")
        if server:
            if server != "github-api":
                return {"error": f"'{server}' is not in the catalog"}
            return {
                "server": server,
                "prefix": "gh",
                "process_running": False,
                "probed": True,
                "available": True,
                "error": None,
                "age_s": 0.1,
                "tools": [
                    {
                        "prefixed_name": "gh_create_issue",
                        "tool": "create_issue",
                        "description": "Open an issue",
                        "enabled": True,
                        "mounted": False,
                    }
                ],
            }
        return {
            "total_servers": 1,
            "total_tools": 1,
            "servers_running": [],
            "unavailable": [],
            "servers": [
                {
                    "server": "github-api",
                    "prefix": "gh",
                    "tool_count": 1,
                    "enabled_count": 1,
                    "process_running": False,
                    "probed": True,
                    "available": True,
                }
            ],
        }

    def status_snapshot(self) -> dict:
        if self._fail_status:
            raise RuntimeError("status snapshot exploded")
        return {"children": {}, "catalog_size": 1}


@pytest.fixture(autouse=True)
def _reset_shared_multiplexer():
    shared_mux_mod._reset_shared_multiplexer_for_tests()
    yield
    shared_mux_mod._reset_shared_multiplexer_for_tests()


def _install_stub(monkeypatch, stub: _StubMultiplexer) -> None:
    async def _get_stub() -> Any:
        return stub

    monkeypatch.setattr(shared_mux_mod, "get_shared_multiplexer", _get_stub)


# ── authorized ──────────────────────────────────────────────────────────────


def test_catalog_route_returns_the_multiplexer_payload_when_authorized(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/catalog")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_servers"] == 1
    assert body["servers"][0]["server"] == "github-api"


def test_catalog_route_drill_down_returns_tools_with_dispatchable_truth(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/catalog", params={"server": "github-api"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tools"][0]["tool"] == "create_issue"
    assert body["tools"][0]["mounted"] is False


def test_catalog_route_unknown_server_is_a_typed_404_not_a_200_error_body(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/catalog", params={"server": "does-not-exist"})

    assert response.status_code == 404


def test_status_route_returns_the_multiplexer_snapshot_when_authorized(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/status")

    assert response.status_code == 200, response.text
    assert response.json() == {"children": {}, "catalog_size": 1}


# ── unauthorized ────────────────────────────────────────────────────────────


def test_catalog_route_refuses_a_caller_with_no_discover_scope(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_NO_SCOPE_CLAIMS)

    response = client.get("/api/mcp/catalog")

    assert response.status_code == 403


def test_status_route_refuses_a_caller_with_no_discover_scope(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_NO_SCOPE_CLAIMS)

    response = client.get("/api/mcp/status")

    assert response.status_code == 403


# ── degraded ────────────────────────────────────────────────────────────────


def test_catalog_route_surfaces_a_typed_degraded_state_never_a_silent_empty_list(
    monkeypatch,
):
    _install_stub(monkeypatch, _StubMultiplexer(fail_catalog=True))
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/catalog")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "DEGRADED"
    assert detail["reason"] == "list_catalog_failed"
    assert "RuntimeError" in detail["detail"]


def test_status_route_surfaces_a_typed_degraded_state_on_snapshot_failure(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer(fail_status=True))
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/status")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "DEGRADED"
    assert detail["reason"] == "status_snapshot_failed"


def test_catalog_route_surfaces_degraded_when_the_shared_multiplexer_cannot_construct(
    monkeypatch,
):
    async def _boom() -> Any:
        raise OSError("mcp_config.json unreadable")

    monkeypatch.setattr(shared_mux_mod, "get_shared_multiplexer", _boom)
    client = _client(_DISCOVER_CLAIMS)

    response = client.get("/api/mcp/catalog")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "DEGRADED"
    assert detail["reason"] == "mcp_multiplexer_unavailable"


# ── REST/MCP parity ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rest_catalog_payload_matches_the_shared_multiplexer_payload_directly(
    monkeypatch, tmp_path
):
    """Same session, same underlying multiplexer: the REST route must return
    EXACTLY what a direct ``mux.list_catalog()`` call (the same call the
    ``list_catalog`` MCP tool makes) returns — no reshaping, no drift.
    """
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(shared_mux_mod, "_default_config_path", lambda: config_path)

    # First call constructs and caches the real (empty-catalog) shared
    # multiplexer; the REST route below reuses that SAME instance.
    direct_mux = await shared_mux_mod.get_shared_multiplexer()
    direct_payload = await direct_mux.list_catalog(server="", include_tools=True)
    direct_status = direct_mux.status_snapshot()

    client = _client(_DISCOVER_CLAIMS)
    rest_catalog = client.get("/api/mcp/catalog").json()
    rest_status = client.get("/api/mcp/status").json()

    assert rest_catalog == direct_payload
    assert rest_status == direct_status
