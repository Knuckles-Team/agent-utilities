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
from agent_utilities.mcp.catalog_reconciliation import (
    CatalogContractError,
    CatalogIdentity,
    CatalogRefreshResult,
    CatalogSessionResumeResult,
    CatalogSnapshot,
)
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
_ADMIN_CLAIMS = {
    "auth_type": "jwt",
    "sub": "mcp-catalog-test-admin",
    "scope": "mcp:admin",
}
_DELEGATE_CLAIMS = {
    "auth_type": "jwt",
    "sub": "mcp-catalog-test-delegate",
    "scope": "mcp:delegate",
}
_REFRESH_REQUEST = {
    "request_id": "refresh-1",
    "expected_config_revision": "config-1",
    "expected_catalog_generation": 0,
    "expected_snapshot_digest": "a" * 64,
    "deadline_ms": 1000,
}


class _StubMultiplexer:
    """Deterministic stand-in with the same public surface the REST route
    calls (``list_catalog``, ``status_snapshot``)."""

    def __init__(
        self,
        *,
        fail_catalog: bool = False,
        fail_status: bool = False,
        fail_refresh: bool = False,
        fail_dispatch: Exception | None = None,
        fail_resume: Exception | None = None,
    ):
        self._fail_catalog = fail_catalog
        self._fail_status = fail_status
        self._fail_refresh = fail_refresh
        self._fail_dispatch = fail_dispatch
        self._fail_resume = fail_resume
        self.refreshed: list[str] = []
        self.dispatched: list[str] = []

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

    async def refresh_catalog(self, request) -> CatalogRefreshResult:
        if self._fail_refresh:
            raise RuntimeError("refresh exploded")
        self.refreshed.append(request.request_id)
        return CatalogRefreshResult(
            request_id=request.request_id,
            served_instance_id="graph-os:test",
            release_id="test",
            config_revision=request.expected_config_revision,
            catalog_generation=request.expected_catalog_generation,
            snapshot_digest=request.expected_snapshot_digest,
            changed=False,
            reingestion_state="reconciled",
        )

    def catalog_snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            identity=CatalogIdentity(
                served_instance_id="graph-os:test",
                release_id="test",
                config_revision="config-1",
                catalog_generation=0,
                snapshot_digest="a" * 64,
                child_connection_generation=0,
                authorization_scope_digest="b" * 64,
            ),
            children=(),
        )

    async def dispatch_catalog_tool(
        self,
        *,
        tool_name: str,
        arguments: dict,
        expected_catalog_generation: int,
        expected_snapshot_digest: str,
    ):
        from mcp import types as mcp_types

        if self._fail_dispatch is not None:
            raise self._fail_dispatch
        self.dispatched.append(tool_name)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=f"dispatched:{tool_name}")]
        )

    def resume_catalog_session(self, request) -> CatalogSessionResumeResult:
        if self._fail_resume is not None:
            raise self._fail_resume
        return CatalogSessionResumeResult(
            session_id=request.session_id,
            served_instance_id="graph-os:test",
            release_id=request.release_id,
            config_revision=request.config_revision,
            catalog_generation=request.catalog_generation,
            snapshot_digest=request.snapshot_digest,
            child_connection_generation=request.child_connection_generation,
            authorization_scope_digest=request.authorization_scope_digest,
            resume_state="resumed",
        )


@pytest.fixture(autouse=True)
def _reset_shared_multiplexer():
    shared_mux_mod._reset_served_multiplexer_for_tests()
    yield
    shared_mux_mod._reset_served_multiplexer_for_tests()


def _install_stub(monkeypatch, stub: _StubMultiplexer) -> None:
    async def _get_stub() -> Any:
        return stub

    monkeypatch.setattr(shared_mux_mod, "get_served_multiplexer", _get_stub)


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


def test_refresh_route_returns_exact_mux_payload_for_admin(monkeypatch):
    stub = _StubMultiplexer()
    _install_stub(monkeypatch, stub)
    client = _client(_ADMIN_CLAIMS)

    response = client.post("/api/mcp/catalog/refresh", json=_REFRESH_REQUEST)

    assert response.status_code == 200, response.text
    assert response.json()["request_id"] == "refresh-1"
    assert response.json()["snapshot_digest"] == "a" * 64
    assert stub.refreshed == ["refresh-1"]


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


def test_refresh_route_refuses_discover_without_admin_scope(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.post("/api/mcp/catalog/refresh", json=_REFRESH_REQUEST)

    assert response.status_code == 403


def test_catalog_route_fails_closed_when_no_claims_were_ever_set(monkeypatch):
    """Absent ``request.state.user_claims`` must NOT be treated as the
    trusted static-API-key bypass — a request that never went through
    identity verification is denied (403), not silently waved through."""
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(None)

    response = client.get("/api/mcp/catalog")

    assert response.status_code == 403


def test_status_route_fails_closed_when_no_claims_were_ever_set(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(None)

    response = client.get("/api/mcp/status")

    assert response.status_code == 403


def test_refresh_route_fails_closed_when_no_claims_were_ever_set(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(None)

    response = client.post("/api/mcp/catalog/refresh", json=_REFRESH_REQUEST)

    assert response.status_code == 403


# ── dispatch/session-resume parity (blocker: REST lacked these entirely) ────


_DISPATCH_REQUEST = {
    "tool_name": "gh__create_issue",
    "arguments": {"title": "hi"},
    "expected_catalog_generation": 0,
    "expected_snapshot_digest": "a" * 64,
}
_SESSION_RESUME_REQUEST = {
    "session_id": "sess-1",
    "previous_served_instance_id": "graph-os:test",
    "release_id": "test",
    "config_revision": "config-1",
    "catalog_generation": 0,
    "snapshot_digest": "a" * 64,
    "child_connection_generation": 0,
    "authorization_scope_digest": "b" * 64,
    "resume_token_digest": "c" * 64,
    "deadline_ms": 1000,
}


def test_dispatch_route_invokes_the_same_multiplexer_dispatch_method(monkeypatch):
    stub = _StubMultiplexer()
    _install_stub(monkeypatch, stub)
    client = _client(_DELEGATE_CLAIMS)

    response = client.post("/api/mcp/dispatch", json=_DISPATCH_REQUEST)

    assert response.status_code == 200, response.text
    assert stub.dispatched == ["gh__create_issue"]
    body = response.json()
    assert body["content"][0]["text"] == "dispatched:gh__create_issue"


def test_dispatch_route_refuses_a_caller_with_no_delegate_scope(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)  # discover only, no mcp:delegate

    response = client.post("/api/mcp/dispatch", json=_DISPATCH_REQUEST)

    assert response.status_code == 403


def test_dispatch_route_fails_closed_when_no_claims_were_ever_set(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(None)

    response = client.post("/api/mcp/dispatch", json=_DISPATCH_REQUEST)

    assert response.status_code == 403


def test_dispatch_route_surfaces_a_stale_generation_as_a_deterministic_409(
    monkeypatch,
):
    stub = _StubMultiplexer(
        fail_dispatch=CatalogContractError(
            "catalog-generation-stale", "stale generation", retryable=True
        )
    )
    _install_stub(monkeypatch, stub)
    client = _client(_DELEGATE_CLAIMS)

    response = client.post("/api/mcp/dispatch", json=_DISPATCH_REQUEST)

    assert response.status_code == 409
    assert response.json()["code"] == "catalog-generation-stale"


def test_session_resume_route_invokes_the_same_multiplexer_resume_method(monkeypatch):
    stub = _StubMultiplexer()
    _install_stub(monkeypatch, stub)
    client = _client(_DELEGATE_CLAIMS)

    response = client.post("/api/mcp/session_resume", json=_SESSION_RESUME_REQUEST)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == "sess-1"
    assert body["resume_state"] == "resumed"


def test_session_resume_route_refuses_a_caller_with_no_delegate_scope(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer())
    client = _client(_DISCOVER_CLAIMS)

    response = client.post("/api/mcp/session_resume", json=_SESSION_RESUME_REQUEST)

    assert response.status_code == 403


def test_session_resume_route_surfaces_a_contract_error_as_a_typed_409(monkeypatch):
    stub = _StubMultiplexer(
        fail_resume=CatalogContractError(
            "replica-generation-divergent", "cohort diverged", retryable=True
        )
    )
    _install_stub(monkeypatch, stub)
    client = _client(_DELEGATE_CLAIMS)

    response = client.post("/api/mcp/session_resume", json=_SESSION_RESUME_REQUEST)

    assert response.status_code == 409
    assert response.json()["code"] == "replica-generation-divergent"


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


def test_refresh_route_surfaces_runtime_failure(monkeypatch):
    _install_stub(monkeypatch, _StubMultiplexer(fail_refresh=True))
    client = _client(_ADMIN_CLAIMS)
    degraded = client.post("/api/mcp/catalog/refresh", json=_REFRESH_REQUEST)
    assert degraded.status_code == 503
    assert degraded.json()["detail"]["reason"] == "mcp_catalog_refresh_failed"


def test_catalog_route_surfaces_degraded_when_the_shared_multiplexer_cannot_construct(
    monkeypatch,
):
    async def _boom() -> Any:
        raise OSError("mcp_config.json unreadable")

    monkeypatch.setattr(shared_mux_mod, "get_served_multiplexer", _boom)
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
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    direct_mux = MCPMultiplexer(config_path)
    shared_mux_mod.bind_served_multiplexer(direct_mux)
    direct_payload = await direct_mux.list_catalog(server="", include_tools=True)
    direct_status = direct_mux.status_snapshot()

    client = _client(_DISCOVER_CLAIMS)
    rest_catalog = client.get("/api/mcp/catalog").json()
    rest_status = client.get("/api/mcp/status").json()

    assert rest_catalog == direct_payload
    assert rest_status == direct_status
