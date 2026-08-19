"""Focused tenant, cursor, bound, and degraded-read registry contracts."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.gateway import registry_api
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    suspend_session,
    use_session,
)
from agent_utilities.mcp.remote_oauth_broker import OAuthGrantBinding
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


class _FakeGraphCompute:
    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None):
        self.rows = rows or {}
        self.statements: list[str] = []
        self.fail = False
        self.return_raw = False

    def sql_exec(self, statement: str):
        self.statements.append(statement)
        if self.fail:
            raise OSError("catalog backend unavailable")
        table = statement.split(" FROM ", 1)[1].split(" WHERE ", 1)[0]
        rows = self.rows.get(table, [])
        if self.return_raw:
            return rows
        tenant_marker = "tenant_id = '"
        requested_tenant = statement.split(tenant_marker, 1)[1].split("'", 1)[0]
        principal = None
        principal_marker = "discovery_principal = '"
        if principal_marker in statement:
            principals = [
                value
                for value in re.findall(r"discovery_principal = '([^']*)'", statement)
                if value
            ]
            principal = principals[-1] if principals else ""
        grants: tuple[str, ...] = ()
        marker = "discovery_grant_digest IN ("
        if marker in statement:
            raw_grants = statement.split(marker, 1)[1].split(")", 1)[0]
            grants = tuple(
                part.strip().strip("'").replace("''", "'")
                for part in raw_grants.split(",")
                if part.strip()
            )
        local_scope = "discovery_authority_kind = 'tenant_local'" in statement
        oauth_scope = "discovery_authority_kind = 'oauth_grant'" in statement
        # The fake models the engine's row-level tenant/principal predicate so
        # tests exercise the same boundary the real SQL authority enforces.
        scoped = []
        for row in rows:
            if row.get("tenant_id") != requested_tenant:
                continue
            if principal is None:
                scoped.append(row)
                continue
            if local_scope and row.get("discovery_authority_kind") == "tenant_local":
                if (
                    row.get("discovery_principal") == ""
                    and row.get("discovery_grant_digest") == ""
                ):
                    scoped.append(row)
                continue
            if (
                oauth_scope
                and row.get("discovery_authority_kind") == "oauth_grant"
                and row.get("discovery_principal") == principal
                and row.get("discovery_grant_digest") in grants
            ):
                scoped.append(row)
        return scoped


class _FakeEngine:
    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None):
        self.graph_compute = _FakeGraphCompute(rows)


class _AuthorityMiddleware:
    def __init__(self, app, actor: ActorContext, session: GraphSession):
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send):
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)


def _authority_app(
    monkeypatch,
    *,
    engine: _FakeEngine,
    actor_id: str = "actor-a",
    tenant_id: str = "tenant-a",
):
    actor = ActorContext(
        actor_id=actor_id,
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
        registry_api,
        "_resolve_current_discovery_grants",
        lambda actor: (_grant_digest(actor.actor_id, actor.tenant_id),),
    )
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")
    return TestClient(_AuthorityMiddleware(app, actor, session))


def _grant_digest(actor_id: str, tenant_id: str = "tenant-a") -> str:
    return OAuthGrantBinding(
        tenant_id=tenant_id,
        principal_id=actor_id,
        provider_id="acme",
        resource_url="https://protected-mcp.example.com/mcp",
        audience="https://protected-mcp.example.com/mcp",
        granted_scopes=("mcp:read",),
        key_version=1,
        grant_revision=f"registry-test-{tenant_id}-{actor_id}",
    ).fingerprint


def _rows() -> dict[str, list[dict[str, Any]]]:
    grant_a = _grant_digest("actor-a")
    grant_b = _grant_digest("actor-b")
    return {
        "mcp_servers": [
            {
                "id": "mcp_server_alpha",
                "tenant_id": "tenant-a",
                "name": "alpha",
                "transport": "http",
                "url": "https://user:password@example.test/api?token=secret",
                "enabled": True,
            },
            {
                "id": "mcp_server_beta",
                "tenant_id": "tenant-a",
                "name": "beta",
                "transport": "stdio",
                "url": "https://example.test/mcp/secret-token",
                "enabled": False,
            },
            {
                "id": "mcp_server_alpha",
                "tenant_id": "tenant-b",
                "name": "foreign",
                "transport": "http",
                "url": "https://foreign.test",
                "enabled": True,
            },
        ],
        "mcp_server_discovery": [
            {
                "id": "disc_alpha",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "reachable": False,
                "last_error": "/private/path/token=secret",
                "tool_count": 1,
                "skill_count": 0,
                "prompt_count": 0,
                "resource_count": 0,
                "observed_at": "2026-08-18T00:00:00Z",
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-a",
                "discovery_grant_digest": grant_a,
            },
            {
                "id": "disc_other",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "reachable": True,
                "last_error": "",
                "tool_count": 2,
                "skill_count": 0,
                "prompt_count": 0,
                "resource_count": 0,
                "observed_at": "2026-08-18T00:00:00Z",
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-b",
                "discovery_grant_digest": grant_b,
            },
        ],
        "mcp_tools": [
            {
                "id": "tool_alpha_a",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "only-a",
                "description": "",
                "schema_digest": "a",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-a",
                "discovery_grant_digest": grant_a,
            },
            {
                "id": "tool_alpha_b",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "only-b",
                "description": "",
                "schema_digest": "b",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-b",
                "discovery_grant_digest": grant_b,
            },
        ],
    }


def _rows_with_local() -> dict[str, list[dict[str, Any]]]:
    rows = _rows()
    rows["mcp_server_discovery"].append(
        {
            **rows["mcp_server_discovery"][0],
            "id": "disc_local",
            "server_name": "local-alpha",
            "server_id": "mcp_server_local-alpha",
            "discovery_authority_kind": "tenant_local",
            "discovery_principal": "",
            "discovery_grant_digest": "",
        }
    )
    rows["mcp_tools"].append(
        {
            **rows["mcp_tools"][0],
            "id": "tool_local",
            "server_name": "local-alpha",
            "server_id": "mcp_server_local-alpha",
            "name": "local-tool",
            "discovery_authority_kind": "tenant_local",
            "discovery_principal": "",
            "discovery_grant_digest": "",
        }
    )
    return rows


def _registry_routes(app: Any) -> list[tuple[str, set[str]]]:
    """Flatten FastAPI/Starlette route wrappers for a stable contract check."""

    found: list[tuple[str, set[str]]] = []
    pending = list(getattr(app, "routes", ()))
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if isinstance(path, str) and "registry" in path:
            found.append((path, set(methods or ())))
        original = getattr(route, "original_router", None)
        if original is not None:
            pending.extend(getattr(original, "routes", ()))
        pending.extend(getattr(route, "routes", ()) or ())
    return found


def test_registry_reads_native_catalog_with_tenant_and_principal_predicate(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers", params={"q": "alpha"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["name"] == "alpha"
    assert (
        body["items"][0]["url"] == ""
    )  # userinfo/query credentials are never returned
    statement = engine.graph_compute.statements[-1]
    assert "tenant_id = 'tenant-a'" in statement
    assert "alpha" not in statement  # caller filter is applied after ACL read
    assert "tenant-b" not in response.text


def test_discovery_is_principal_scoped_and_error_is_classified(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/discoveries")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["last_error"] == "unavailable"
    assert "actor-b" not in response.text
    assert "discovery_principal = 'actor-a'" in engine.graph_compute.statements[-1]


def test_derived_catalog_is_disjoint_by_principal_and_grant(monkeypatch):
    engine = _FakeEngine(_rows())
    actor_a = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")
    response_a = actor_a.get("/api/registry/tools")
    assert response_a.status_code == 200, response_a.text
    assert [item["name"] for item in response_a.json()["items"]] == ["only-a"]
    assert "discovery_grant_digest IN (" in engine.graph_compute.statements[-1]

    actor_b = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    response_b = actor_b.get("/api/registry/tools")
    assert response_b.status_code == 200, response_b.text
    assert [item["name"] for item in response_b.json()["items"]] == ["only-b"]
    assert "only-a" not in response_b.text


def test_local_discovery_is_tenant_readable_without_oauth_grant(monkeypatch):
    engine = _FakeEngine(_rows_with_local())
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(
        registry_api, "_resolve_current_discovery_grants", lambda _actor: ()
    )

    response = client.get("/api/registry/tools")

    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["items"]] == ["local-tool"]
    statement = engine.graph_compute.statements[-1]
    assert "discovery_authority_kind = 'tenant_local'" in statement
    assert "discovery_grant_digest IN (" not in statement


def test_local_visibility_does_not_broaden_oauth_principal_isolation(monkeypatch):
    rows = _rows_with_local()
    engine = _FakeEngine(rows)
    client_a = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")
    response_a = client_a.get("/api/registry/tools")
    assert response_a.status_code == 200, response_a.text
    assert {item["name"] for item in response_a.json()["items"]} == {
        "local-tool",
        "only-a",
    }

    client_b = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    response_b = client_b.get("/api/registry/tools")
    assert response_b.status_code == 200, response_b.text
    assert {item["name"] for item in response_b.json()["items"]} == {
        "local-tool",
        "only-b",
    }
    assert "only-a" not in response_b.text


def test_registry_predicate_accepts_only_current_grants_for_one_principal(monkeypatch):
    rows = _rows()
    grant_a = _grant_digest("actor-a")
    grant_refresh = _grant_digest("actor-a-refresh")
    rows["mcp_tools"].append(
        {
            **rows["mcp_tools"][0],
            "id": "tool_alpha_refresh",
            "name": "only-a-refresh",
            "discovery_grant_digest": grant_refresh,
        }
    )
    engine = _FakeEngine(rows)
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(
        registry_api,
        "_resolve_current_discovery_grants",
        lambda _actor: (grant_a, grant_refresh),
    )

    response = client.get("/api/registry/tools")

    assert response.status_code == 200, response.text
    assert {item["name"] for item in response.json()["items"]} == {
        "only-a",
        "only-a-refresh",
    }
    statement = engine.graph_compute.statements[-1]
    assert "discovery_grant_digest IN (" in statement


def test_cursor_is_stable_scope_bound_and_tamper_evident(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    first = client.get("/api/registry/servers", params={"limit": 1})
    assert first.status_code == 200
    cursor = first.json()["next_cursor"]
    assert cursor

    second = client.get("/api/registry/servers", params={"limit": 1, "cursor": cursor})
    assert second.status_code == 200
    assert [item["name"] for item in second.json()["items"]] == ["beta"]
    assert second.json()["items"][0]["url"] == "https://example.test"
    assert "secret-token" not in second.text

    tampered = client.get(
        "/api/registry/servers", params={"limit": 1, "cursor": cursor + "x"}
    )
    assert tampered.status_code == 400
    assert tampered.json()["detail"] == "invalid registry cursor"


def test_cursor_is_bound_to_actor_tenant_filter_and_kind(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)
    first = client.get("/api/registry/servers", params={"limit": 1})
    cursor = first.json()["next_cursor"]
    assert cursor

    actor_client = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    assert (
        actor_client.get(
            "/api/registry/servers", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )
    tenant_client = _authority_app(
        monkeypatch, engine=engine, actor_id="actor-a", tenant_id="tenant-b"
    )
    assert (
        tenant_client.get(
            "/api/registry/servers", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/registry/servers",
            params={"limit": 1, "q": "alpha", "cursor": cursor},
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/registry/tools", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )


def test_expired_cursor_is_rejected(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(registry_api, "_CURSOR_TTL_SECONDS", -1.0)
    expired = registry_api._cursor_token(
        kind="servers",
        query="",
        after=("alpha", "mcp_server_alpha"),
        tenant="tenant-a",
        principal="actor-a",
        grant_digest="",
    )

    response = client.get(
        "/api/registry/servers", params={"limit": 1, "cursor": expired}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid registry cursor"


def test_filter_is_literal_and_bounds_are_enforced(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    injection_like = client.get("/api/registry/servers", params={"q": "' OR 1=1 --"})
    assert injection_like.status_code == 200
    assert injection_like.json()["items"] == []
    assert client.get("/api/registry/servers", params={"limit": 0}).status_code == 422
    assert client.get("/api/registry/servers", params={"limit": 101}).status_code == 422


def test_catalog_failure_is_explicit_unavailable(monkeypatch):
    engine = _FakeEngine(_rows())
    engine.graph_compute.fail = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


@pytest.mark.parametrize("malformation", ["missing_tenant", "wrong_tenant"])
def test_malformed_catalog_scope_is_explicitly_unavailable(monkeypatch, malformation):
    engine = _FakeEngine(_rows())
    engine.graph_compute.return_raw = True
    row = engine.graph_compute.rows["mcp_servers"][0]
    if malformation == "missing_tenant":
        row.pop("tenant_id")
    else:
        row["tenant_id"] = "tenant-b"
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_legacy_unbound_derived_row_is_not_relabelled_public(monkeypatch):
    rows = {
        "mcp_tools": [
            {
                "id": "legacy-tool",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "legacy",
                "description": "",
                "schema_digest": "legacy",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_principal": "",
                # The legacy row intentionally has no grant digest.
            }
        ]
    }
    engine = _FakeEngine(rows)
    engine.graph_compute.return_raw = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/tools")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


@pytest.mark.parametrize(
    ("authority_kind", "principal", "grant"),
    [
        ("unknown", "", ""),
        ("tenant_local", "spoofed", ""),
        ("tenant_local", "", "spoofed-grant"),
    ],
)
def test_malformed_local_authority_is_unavailable(
    monkeypatch, authority_kind, principal, grant
):
    rows = _rows_with_local()
    rows["mcp_tools"] = [
        {
            **rows["mcp_tools"][1],
            "discovery_authority_kind": authority_kind,
            "discovery_principal": principal,
            "discovery_grant_digest": grant,
        }
    ]
    engine = _FakeEngine(rows)
    engine.graph_compute.return_raw = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/tools")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_invalid_catalog_model_field_is_explicitly_unavailable(monkeypatch):
    engine = _FakeEngine(_rows())
    engine.graph_compute.rows["mcp_servers"][0]["enabled"] = "not-a-boolean"
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_catalog_row_bound_is_explicitly_unavailable(monkeypatch):
    source = _rows()["mcp_servers"][0]
    rows = [{**source, "id": f"mcp_server_{index}"} for index in range(10_001)]
    engine = _FakeEngine({"mcp_servers": rows})
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_missing_item_is_privacy_safe_and_no_write_verbs_are_mounted(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "registry item not found"

    routes = _registry_routes(client.app.app)
    assert routes
    assert all(methods == {"GET"} for _, methods in routes)


def test_missing_graph_session_is_denied(monkeypatch):
    monkeypatch.setattr(
        registry_api, "_get_catalog_engine", lambda: _FakeEngine(_rows())
    )
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")

    with suspend_session():
        response = TestClient(app).get("/api/registry/servers")

    assert response.status_code == 403
    assert response.json()["detail"] == "registry access denied"
