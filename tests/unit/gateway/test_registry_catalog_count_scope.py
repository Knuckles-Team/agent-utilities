"""BUG-CX-118 — the registry ``count`` must never carry another tenant's rows.

Found by probing (not reading) the externally-reachable ``/api/registry/*``
route with an engine whose ``graph_compute.sql_exec`` applies no row-level
filtering of its own — which is what that method actually does.

``_authorized_page`` re-validates every returned row against the caller's
tenant/principal contract (``_validate_scope``) and denies the WHOLE read when
one is out of scope. ``_authorized_count`` had no equivalent and could not have
one: an aggregate has no rows to judge. So an engine that honored the WHERE
clause on the page but not on ``SELECT COUNT(*)`` served a tenant-a caller
``items`` containing only tenant-a's two rows and ``count: 3`` — the third
being tenant-b's. A count is a small leak, but it is a real cross-tenant one,
and it was the only value on this route no row-level check could reach.

The fix derives the total from the already-scope-validated page whenever the
page IS the whole result set, so the unvalidatable aggregate is not issued at
all; the residual multi-page case is at least reconciled against the page.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.gateway import registry_api
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

_TENANT_TERM = re.compile(r"tenant_id = '([^']*)'")
_TABLE = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)")
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$")


def _catalog_rows() -> dict[str, list[dict[str, Any]]]:
    """Two tenant-a servers and one tenant-b server in the SAME table."""

    return {
        "mcp_servers": [
            {
                "id": "srv-a1",
                "tenant_id": "tenant-a",
                "name": "alpha",
                "transport": "http",
                "url": "https://a1.test",
                "enabled": True,
            },
            {
                "id": "srv-a2",
                "tenant_id": "tenant-a",
                "name": "beta",
                "transport": "stdio",
                "url": "https://a2.test",
                "enabled": True,
            },
            {
                "id": "srv-b1",
                "tenant_id": "tenant-b",
                "name": "foreign",
                "transport": "http",
                "url": "https://b1.test",
                "enabled": True,
            },
        ]
    }


class _AggregateBlindGraphCompute:
    """The known-bad engine: correct row filtering, WRONG aggregate.

    Models an engine whose predicate pushdown is lost on the aggregate plan —
    exactly the "misconfigured engine projection" threat ``_validate_scope``'s
    own docstring names, but on the one value ``_validate_scope`` cannot see.
    """

    def __init__(self, rows: dict[str, list[dict[str, Any]]]):
        self.rows = rows
        self.statements: list[str] = []

    def sql_exec(self, statement: str) -> list[dict[str, Any]]:
        self.statements.append(statement)
        table = _TABLE.search(statement).group(1)
        table_rows = self.rows.get(table, [])
        if statement.lstrip().upper().startswith("SELECT COUNT("):
            # The defect: the WHERE clause never reaches the aggregate.
            return [{"row_count": len(table_rows)}]
        tenant = _TENANT_TERM.search(statement)
        matched = [
            row
            for row in table_rows
            if tenant is None or row.get("tenant_id") == tenant.group(1)
        ]
        matched.sort(key=lambda row: (str(row["name"]).lower(), str(row["id"])))
        bound = _LIMIT.search(statement)
        return matched[: int(bound.group(1))] if bound else matched


class _Engine:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]):
        self.graph_compute = _AggregateBlindGraphCompute(rows)


class _BindAuthority:
    def __init__(self, app, actor: ActorContext, session: GraphSession):
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send):
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)


def _client(monkeypatch, engine: _Engine, *, tenant: str = "tenant-a") -> TestClient:
    actor = ActorContext(
        actor_id="actor-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("registry:read",),
        tenant_id=tenant,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read"}),
        graph=tenant,
        policy_version="test",
        audience="test",
    )
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: engine)
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")
    return TestClient(_BindAuthority(app, actor, session))


def test_count_never_reports_another_tenants_rows(monkeypatch):
    """The probe, as a regression: ``count`` must equal what the caller may see."""

    engine = _Engine(_catalog_rows())
    client = _client(monkeypatch, engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["name"] for item in body["items"]] == ["alpha", "beta"]
    assert body["count"] == 2, (
        "the registry served a count that includes another tenant's rows: "
        f"{body['count']} reported for {len(body['items'])} authorized items"
    )
    assert "tenant-b" not in response.text and "foreign" not in response.text


def test_complete_page_does_not_ask_the_engine_to_count_at_all(monkeypatch):
    """When the page IS the whole result set, no unvalidatable aggregate is run."""

    engine = _Engine(_catalog_rows())
    client = _client(monkeypatch, engine)

    assert client.get("/api/registry/servers").status_code == 200
    assert not [
        statement
        for statement in engine.graph_compute.statements
        if statement.lstrip().upper().startswith("SELECT COUNT(")
    ], engine.graph_compute.statements


def test_multi_kind_route_shares_the_same_count_contract(monkeypatch):
    engine = _Engine(_catalog_rows())
    client = _client(monkeypatch, engine)

    response = client.get("/api/registry", params={"kinds": "servers"})

    assert response.status_code == 200, response.text
    servers = response.json()["kinds"]["servers"]
    assert servers["count"] == len(servers["items"]) == 2


def test_a_count_smaller_than_the_authorized_page_is_refused():
    """The residual multi-page path still holds the aggregate to one invariant."""

    with pytest.raises(registry_api.CatalogUnavailable):
        registry_api._reconciled_total(1, [{"id": "a"}, {"id": "b"}])
    assert registry_api._reconciled_total(9, [{"id": "a"}]) == 9


@pytest.mark.parametrize(
    ("has_more", "after", "expected"),
    [
        (False, None, True),
        (True, None, False),
        (False, ("beta", "srv-a2"), False),
        (True, ("beta", "srv-a2"), False),
    ],
)
def test_only_an_uncursored_complete_page_is_its_own_total(has_more, after, expected):
    assert registry_api._page_is_the_whole_result(has_more, after) is expected
