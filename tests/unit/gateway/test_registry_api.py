"""Registry API contracts post-EH-345: parsing, redaction, row-shape
conversion from EG's typed fleet-catalog DTOs, cursor round-trip, and the
dispatch layer over fake ``server_registry``/``fleet_catalog`` clients.

EH-345 (2026-09-22) deleted the AU SQL fleet-catalog tier this route used to
read and, with it, the WHERE-clause-building/keyset-predicate/row-scope-
validation machinery the previous version of this test file exercised (a
hand-written SQL interpreter over a fake engine). That machinery no longer
exists in the production code -- tenant/principal/grant visibility,
filtering, ordering and the total are now computed by EG itself, one round
trip, never locally. This file tests what the rewritten route actually does:
parse/validate caller input, convert EG's row shapes into the public
response models, mint/verify the opaque pagination cursor, and dispatch to
whichever typed client (``server_registry`` for ``servers``, ``fleet_catalog``
for every other kind) a request's ``kind`` names -- using fake clients that
implement the same synchronous method surface the real
``SyncEpistemicGraphClient`` wrappers expose (see AU-CUTOVER.md §0), since
the real generated ``epistemic_graph.generated.fleet_catalog``/
``server_registry`` modules do not exist in this dev environment until the
``eg-fleet-catalog`` lane's work compiles and lands (this route's own EG
calls therefore also fail closed here, exactly as they will in any
deployment before that lands -- see WRAPUP.md for the full evidence trail).

Full tenant/principal/grant visibility-enforcement testing against the real
mechanism is EG's own test suite's job now, not AU's -- this file cannot and
does not attempt to re-prove that boundary locally, unlike its predecessor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from agent_utilities.gateway import registry_api
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    use_session,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

# ---------------------------------------------------------------------------
# Fake EG row/view/cursor objects -- plain SimpleNamespace, matching only the
# attributes the production `_row_from_server_view`/`_row_from_fleet_row`
# actually read (via `getattr`/`hasattr`), never the real generated pydantic
# classes (which do not exist in this environment -- see module docstring).
# ---------------------------------------------------------------------------


def _server_view(
    name: str,
    *,
    url: str = "http://svc:1",
    transport: str = "streamable_http",
    desired: str = "enabled",
) -> SimpleNamespace:
    return SimpleNamespace(name=name, url=url, transport=transport, desired=desired)


def _acl(tenant_id: str = "tenant-a") -> SimpleNamespace:
    return SimpleNamespace(tenant_id=tenant_id)


def _component(
    *,
    id: str,
    name: str,
    server_name: str = "svc-a",
    connector: str = "svc-a",
    description: str = "a tool",
    enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        name=name,
        server_name=server_name,
        connector=connector,
        description=description,
        enabled=enabled,
        acl=_acl(),
    )


def _tool_row(
    component: SimpleNamespace, *, digest: str = "sha256:" + "a" * 64
) -> SimpleNamespace:
    return SimpleNamespace(
        component=component, input_schema_digest=digest, tool_mode="condensed"
    )


def _resource_row(component: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        component=component,
        uri="skill://x",
        media_type="text/markdown",
        resource_kind="skill",
    )


def _skill_row(component: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        component=component,
        uri="skill://x",
        skill_type="skill",
        classification="Atomic Skill",
    )


def _prompt_row(component: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(component=component, uri="prompt://x")


def _discovery_row(
    *,
    id: str = "srvobs:svc-a:tenant_local",
    server_name: str = "svc-a",
    reachable: bool = True,
    error: str = "",
) -> SimpleNamespace:
    outcome = (
        SimpleNamespace(status="reachable")
        if reachable
        else SimpleNamespace(status="unreachable", error=error)
    )
    return SimpleNamespace(
        id=id,
        server_name=server_name,
        outcome=outcome,
        counts=SimpleNamespace(tools=3, skills=1, prompts=0, resources=0),
        observed_at_ms=1234567,
        acl=_acl(),
    )


def _row_entry(row: SimpleNamespace) -> SimpleNamespace:
    """One ``FleetCatalogRow`` tagged-union entry: ``entry.row`` is the body."""
    return SimpleNamespace(row=row)


# ---------------------------------------------------------------------------
# Row-shape conversion (the new logic EH-345 actually added)
# ---------------------------------------------------------------------------


def test_row_from_server_view_maps_desired_to_enabled():
    row = registry_api._row_from_server_view(_server_view("svc-a", desired="enabled"))
    assert row == {
        "id": "mcp_server_svc-a",
        "name": "svc-a",
        "transport": "streamable_http",
        "url": "http://svc:1",
        "enabled": True,
    }


def test_row_from_server_view_disabled():
    row = registry_api._row_from_server_view(_server_view("svc-b", desired="disabled"))
    assert row["enabled"] is False


def test_row_from_server_view_tolerates_missing_transport_desired():
    """Fields not yet generated on RegisteredServerView (pre-eg-fleet-catalog-
    landing) degrade to safe defaults rather than raising."""
    bare = SimpleNamespace(name="svc-c", url="http://svc:3")
    row = registry_api._row_from_server_view(bare)
    assert row["transport"] == ""
    assert row["enabled"] is True  # no "desired" field -> not "disabled" -> True


def test_discovery_outcome_fields_reachable():
    reachable, error = registry_api._discovery_outcome_fields(
        SimpleNamespace(status="reachable")
    )
    assert (reachable, error) == (True, "")


def test_discovery_outcome_fields_unreachable():
    reachable, error = registry_api._discovery_outcome_fields(
        SimpleNamespace(status="unreachable", error="connection refused")
    )
    assert (reachable, error) == (False, "connection refused")


def test_row_from_fleet_row_discovery():
    row = registry_api._row_from_fleet_row(_row_entry(_discovery_row()))
    assert row["id"] == "srvobs:svc-a:tenant_local"
    assert row["server_id"] == row["server_name"] == "svc-a"
    assert row["reachable"] is True
    assert row["tool_count"] == 3
    assert row["tenant_id"] == "tenant-a"


def test_row_from_fleet_row_discovery_unreachable_classifies_error():
    row = registry_api._row_from_fleet_row(
        _row_entry(_discovery_row(reachable=False, error="dns failure"))
    )
    assert row["reachable"] is False
    assert row["last_error"] == "dns failure"


def test_row_from_fleet_row_tool():
    component = _component(id="mcp:svc-a/tool/t1", name="t1")
    row = registry_api._row_from_fleet_row(_row_entry(_tool_row(component)))
    assert row["id"] == "mcp:svc-a/tool/t1"
    assert row["tool_mode"] == "condensed"
    assert row["schema_digest"].startswith("sha256:")
    assert row["provider"] == "svc-a"


def test_row_from_fleet_row_resource():
    component = _component(id="mcp:svc-a/resource/r1", name="r1")
    row = registry_api._row_from_fleet_row(_row_entry(_resource_row(component)))
    assert row["resource_kind"] == "skill"
    assert row["mime_type"] == "text/markdown"


def test_row_from_fleet_row_skill():
    component = _component(id="mcp:skills/skill/s1", name="s1")
    row = registry_api._row_from_fleet_row(_row_entry(_skill_row(component)))
    assert row["skill_type"] == "skill"
    assert row["classification"] == "Atomic Skill"


def test_row_from_fleet_row_prompt():
    component = _component(id="mcp:svc-a/prompt/p1", name="p1")
    row = registry_api._row_from_fleet_row(_row_entry(_prompt_row(component)))
    assert row["uri"] == "prompt://x"
    assert "skill_type" not in row and "tool_mode" not in row


# ---------------------------------------------------------------------------
# Request parsing (unchanged behaviour)
# ---------------------------------------------------------------------------


def _request(query_string: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/registry",
        "query_string": query_string.encode(),
        "headers": [],
    }
    return Request(scope)


def test_parse_request_defaults():
    limit, query, cursor = registry_api._parse_request(_request())
    assert limit == registry_api._DEFAULT_LIMIT
    assert query == ""
    assert cursor is None


def test_parse_request_rejects_out_of_bounds_limit():
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        registry_api._parse_request(_request("limit=0"))
    with pytest.raises(HTTPException):
        registry_api._parse_request(_request(f"limit={registry_api._MAX_LIMIT + 1}"))


def test_parse_request_rejects_oversized_query():
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        registry_api._parse_request(
            _request("q=" + "x" * (registry_api._MAX_QUERY_BYTES + 1))
        )


def test_parse_registry_kinds_validates_against_kind_models():
    from fastapi import HTTPException

    assert registry_api._parse_registry_kinds("tools,skills,tools") == [
        "tools",
        "skills",
    ]
    with pytest.raises(HTTPException):
        registry_api._parse_registry_kinds("not-a-kind")
    with pytest.raises(HTTPException):
        registry_api._parse_registry_kinds("")


# ---------------------------------------------------------------------------
# Redaction/safety helpers (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_safe_url_keeps_bare_host_only():
    assert (
        registry_api._safe_url("https://example.com:8443/some/path")
        == "https://example.com:8443"
    )


def test_safe_url_drops_credentialed_or_query_bearing_urls():
    assert registry_api._safe_url("https://user:pw@example.com/x") == ""
    assert registry_api._safe_url("https://example.com/x?token=secret") == ""


def test_safe_url_rejects_non_http_scheme():
    assert registry_api._safe_url("file:///etc/passwd") == ""


def test_sanitize_discovery_row_classifies_error_and_drops_principal():
    result = {"last_error": "some raw engine text", "discovery_principal": "p1"}
    registry_api._sanitize_discovery_row(result)
    assert result["last_error"] == "unavailable"
    assert "discovery_principal" not in result


# ---------------------------------------------------------------------------
# Cursor mint/decode round trip (new dict-based `after`)
# ---------------------------------------------------------------------------


def test_cursor_round_trips_the_eg_cursor_payload():
    after = {
        "after_name": "svc-a",
        "after_id": "mcp:svc-a/tool/t1",
        "snapshot_digest": "d" * 64,
    }
    token = registry_api._cursor_token(
        kind="tools", query="", after=after, tenant="tenant-a", principal="actor-a"
    )
    decoded = registry_api._decode_cursor(
        token,
        kind="tools",
        query="",
        tenant="tenant-a",
        principal="actor-a",
        grant_digests=(),
    )
    assert decoded == after


def test_cursor_rejects_wrong_kind_or_tenant():
    from fastapi import HTTPException

    after = {"after_name": "svc-a", "after_id": "x"}
    token = registry_api._cursor_token(
        kind="tools", query="", after=after, tenant="tenant-a", principal="actor-a"
    )
    with pytest.raises(HTTPException):
        registry_api._decode_cursor(
            token,
            kind="skills",
            query="",
            tenant="tenant-a",
            principal="actor-a",
            grant_digests=(),
        )
    with pytest.raises(HTTPException):
        registry_api._decode_cursor(
            token,
            kind="tools",
            query="",
            tenant="tenant-b",
            principal="actor-a",
            grant_digests=(),
        )


# ---------------------------------------------------------------------------
# Dispatch layer: fake server_registry / fleet_catalog clients
# ---------------------------------------------------------------------------


class _FakeServerRegistry:
    def __init__(self, views: list[SimpleNamespace]) -> None:
        self.views = views

    def page(self, *, limit: int | None = None, cursor: Any = None) -> SimpleNamespace:
        return SimpleNamespace(
            entries=self.views[: limit or len(self.views)],
            next_cursor=None,
            total_live=len(self.views),
        )

    def list_all(self) -> tuple[SimpleNamespace, ...]:
        return tuple(self.views)


class _FakeFleetCatalog:
    def __init__(self, rows: dict[str, list[SimpleNamespace]]) -> None:
        self.rows = rows
        self.last_request: Any = None

    def page(self, request: Any) -> SimpleNamespace:
        self.last_request = request
        kind_rows = self.rows.get(request.kind, [])
        limit = request.limit or len(kind_rows)
        return SimpleNamespace(
            rows=kind_rows[:limit], next_cursor=None, total=len(kind_rows)
        )

    def lookup(self, ids: list[str], grant_digests: Any = ()) -> SimpleNamespace:
        wanted = set(ids)
        all_rows = [row for kind_rows in self.rows.values() for row in kind_rows]
        matched = [
            entry
            for entry in all_rows
            if registry_api._row_from_fleet_row(entry).get("id") in wanted
        ]
        return SimpleNamespace(rows=matched)


class _FakeGraphCompute:
    def __init__(self, server_registry: Any, fleet_catalog: Any) -> None:
        self.client = SimpleNamespace(
            server_registry=server_registry, fleet_catalog=fleet_catalog
        )


class _FakeEngine:
    def __init__(self, server_registry: Any = None, fleet_catalog: Any = None) -> None:
        self.graph_compute = _FakeGraphCompute(server_registry, fleet_catalog)
        self.preferences: dict[str, str] = {}
        self.query_cypher_calls: list[tuple[str, dict[str, Any]]] = []

    def query_cypher(self, query: str, params: dict[str, Any] | None = None):
        self.query_cypher_calls.append((query, dict(params or {})))
        pref_ids = (params or {}).get("pref_ids", [])
        return [
            {"id": pref_id, "value": self.preferences[pref_id]}
            for pref_id in pref_ids
            if pref_id in self.preferences
        ]


def test_authorized_page_servers_reads_server_registry():
    engine = _FakeEngine(
        server_registry=_FakeServerRegistry([_server_view("a"), _server_view("b")])
    )
    rows, next_cursor, total = registry_api._authorized_page(
        "servers", grant_digests=(), query="", after=None, limit=10, engine=engine
    )
    assert [row["name"] for row in rows] == ["a", "b"]
    assert total == 2
    assert next_cursor is None


def test_authorized_page_servers_unavailable_without_client():
    engine = _FakeEngine()
    with pytest.raises(registry_api.CatalogUnavailable):
        registry_api._authorized_page(
            "servers", grant_digests=(), query="", after=None, limit=10, engine=engine
        )


def test_authorized_page_tools_reads_fleet_catalog():
    component = _component(id="mcp:svc-a/tool/t1", name="t1")
    engine = _FakeEngine(
        fleet_catalog=_FakeFleetCatalog({"tools": [_row_entry(_tool_row(component))]})
    )
    rows, next_cursor, total = registry_api._authorized_page(
        "tools", grant_digests=("a" * 64,), query="", after=None, limit=10, engine=engine
    )
    assert rows[0]["id"] == "mcp:svc-a/tool/t1"
    assert total == 1


def test_authorized_item_servers_finds_by_name():
    engine = _FakeEngine(server_registry=_FakeServerRegistry([_server_view("target")]))
    row = registry_api._authorized_item(
        "servers", grant_digests=(), item_id="mcp_server_target", engine=engine
    )
    assert row is not None
    assert row["name"] == "target"


def test_authorized_item_missing_id_is_none():
    engine = _FakeEngine(server_registry=_FakeServerRegistry([]))
    assert (
        registry_api._authorized_item(
            "servers", grant_digests=(), item_id="mcp_server_nope", engine=engine
        )
        is None
    )


def test_authorized_item_tools_uses_lookup():
    component = _component(id="mcp:svc-a/tool/t1", name="t1")
    engine = _FakeEngine(
        fleet_catalog=_FakeFleetCatalog({"tools": [_row_entry(_tool_row(component))]})
    )
    row = registry_api._authorized_item(
        "tools", grant_digests=(), item_id="mcp:svc-a/tool/t1", engine=engine
    )
    assert row is not None
    assert row["name"] == "t1"


# ---------------------------------------------------------------------------
# Route-level smoke tests (auth gate, 422s, catalog-unavailable degrade)
# ---------------------------------------------------------------------------


class _AuthorityMiddleware:
    def __init__(self, app: Any, actor: ActorContext, session: GraphSession) -> None:
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send) -> None:
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)


def _authority_app(
    monkeypatch, *, engine: _FakeEngine, tenant_id: str = "tenant-a"
) -> TestClient:
    actor = ActorContext(
        actor_id="actor-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("registry:read",),
        tenant_id=tenant_id,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=tenant_id,
        policy_version="test",
        audience="test",
    )
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: engine)
    monkeypatch.setattr(
        registry_api, "_resolve_current_discovery_grants", lambda actor: ()
    )
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")
    return TestClient(_AuthorityMiddleware(app, actor, session))


def test_route_servers_list_returns_registered_servers(monkeypatch):
    # `_authorized_page`'s "servers" branch imports
    # `epistemic_graph.generated.server_registry.RegisteredServerCursor` only
    # when a cursor is actually supplied (`after` is truthy) -- there is none
    # on a first page, so this route path itself needs no cursor type. But
    # `page.next_cursor.model_dump(...)`/`RegisteredServerView` attribute
    # access still goes through whatever `epistemic_graph` this venv has
    # installed; skip cleanly (rather than asserting a specific degrade) when
    # that installed copy predates the ``generated`` subpackage entirely --
    # this dev host's `.venv` currently ships `epistemic_graph==2.27.0` built
    # before ``generated/`` existed (confirmed: no `generated/` directory in
    # the installed wheel), the same pre-existing gap WRAPUP.md documents for
    # `agent_utilities.api`/`control_plane`.
    pytest.importorskip(
        "epistemic_graph.generated.server_registry",
        reason="this venv's installed epistemic_graph predates the generated/ subpackage",
    )
    engine = _FakeEngine(server_registry=_FakeServerRegistry([_server_view("a")]))
    client = _authority_app(monkeypatch, engine=engine)
    resp = client.get("/api/registry/servers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["items"][0]["name"] == "a"


def test_route_unknown_kind_multi_is_422(monkeypatch):
    engine = _FakeEngine()
    client = _authority_app(monkeypatch, engine=engine)
    resp = client.get("/api/registry?kinds=not-a-kind")
    assert resp.status_code == 422


def test_route_missing_kinds_param_is_422(monkeypatch):
    engine = _FakeEngine()
    client = _authority_app(monkeypatch, engine=engine)
    resp = client.get("/api/registry")
    assert resp.status_code == 422


def test_route_catalog_unavailable_degrades_to_503(monkeypatch):
    engine = _FakeEngine()  # no server_registry client configured
    client = _authority_app(monkeypatch, engine=engine)
    resp = client.get("/api/registry/servers")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"


def test_route_missing_item_is_404(monkeypatch):
    engine = _FakeEngine(server_registry=_FakeServerRegistry([]))
    client = _authority_app(monkeypatch, engine=engine)
    resp = client.get("/api/registry/servers/mcp_server_nope")
    assert resp.status_code == 404


def test_route_no_session_is_403():
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")
    client = TestClient(app)
    resp = client.get("/api/registry/servers")
    assert resp.status_code == 403


def test_route_only_get_verbs_are_mounted(monkeypatch):
    engine = _FakeEngine(server_registry=_FakeServerRegistry([]))
    client = _authority_app(monkeypatch, engine=engine)
    for path in ("/api/registry/servers", "/api/registry/servers/x"):
        resp = client.post(path, json={})
        assert resp.status_code in (404, 405)
