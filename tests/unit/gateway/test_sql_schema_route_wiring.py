"""Wire-First proof for ``POST /graph/sql-schema`` (WD1-EB-05).

The route exists so ``agent-webui``'s catalog browser can render
``catalogs -> tables/views -> columns``; the design decision it encodes is that
it must **not** be a raw SQL passthrough
(``plans/semantic-indexing/DESIGN-embedding-bindings.md`` §4).

These are wiring tests, not unit tests: each drives the real mounted endpoint
and asserts a specific downstream seam was reached, with ``observe`` recording
the REAL ``kg_server._execute_tool`` rather than replacing it. The only test
double is the *engine-backed tool* at the far end of that core (a
``graph_table`` stand-in in ``REGISTERED_TOOLS``), because the seam under proof
is route -> shared action core, not the engine's SQL planner.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.gateway import graph_api
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    suspend_session,
    use_session,
)
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import graph_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from tests.wiring import observe

ROUTE = "/api/graph/sql-schema"

_TABLE_ROWS = [
    {
        "table_catalog": "epistemic-graph",
        "table_schema": "public",
        "table_name": "nodes",
        "table_type": "BASE TABLE",
    },
    {
        "table_catalog": "epistemic-graph",
        "table_schema": "public",
        "table_name": "hot_incidents",
        "table_type": "VIEW",
    },
]

_COLUMN_ROWS = [
    {
        "table_catalog": "epistemic-graph",
        "table_schema": "public",
        "table_name": "nodes",
        "column_name": "id",
        "ordinal_position": 1,
        "is_nullable": "YES",
        "data_type": "text",
        "udt_name": "text",
    },
    {
        "table_catalog": "epistemic-graph",
        "table_schema": "public",
        "table_name": "hot_incidents",
        "column_name": "severity",
        "ordinal_position": 1,
        "is_nullable": "YES",
        "data_type": "bigint",
        "udt_name": "int8",
    },
]


class _AuthorityMiddleware:
    def __init__(self, app: Any, actor: ActorContext, session: GraphSession) -> None:
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)


def _authority() -> tuple[ActorContext, GraphSession]:
    actor = ActorContext(
        actor_id="actor-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:read"}),
        graph="tenant-a",
        policy_version="test",
        audience="test",
    )
    return actor, session


def _fake_graph_table(rows_by_statement: dict[str, Any]):
    """A ``graph_table`` stand-in with the real tool's signature + JSON contract."""

    def graph_table(
        action: str = "list",
        source: str = "",
        table: str = "",
        config_json: str = "{}",
        columns_json: str = "[]",
        rows_json: str = "[]",
        sql: str = "",
        limit: int = 1000,
        replace: bool = False,
    ) -> str:
        assert action == "query", (
            f"sql-schema must only use the read action, got {action!r}"
        )
        return json.dumps(rows_by_statement.get(sql, []))

    return graph_table


@pytest.fixture
def client(monkeypatch):
    def _build(rows_by_statement: dict[str, Any]) -> TestClient:
        monkeypatch.setitem(
            kg_server.REGISTERED_TOOLS,
            "graph_table",
            _fake_graph_table(rows_by_statement),
        )
        app = FastAPI()
        graph_api._mount_sql_schema_route(app, prefix="/api")
        actor, session = _authority()
        return TestClient(_AuthorityMiddleware(app, actor, session))

    return _build


def _full_catalog() -> dict[str, Any]:
    return {
        graph_tools.CATALOG_STATEMENTS["tables"]: _TABLE_ROWS,
        graph_tools.CATALOG_STATEMENTS["columns"]: _COLUMN_ROWS,
        graph_tools.CATALOG_STATEMENTS["primary_keys"]: [],
    }


def test_route_reaches_the_shared_execute_tool_action_core(client):
    """The seam: HTTP -> ``kg_server._execute_tool('graph_table', action='query')``."""
    api = client(_full_catalog())

    with observe(kg_server, "_execute_tool") as core:
        response = api.post(ROUTE, json={})

    assert response.status_code == 200, response.text
    core.assert_called(why="both surfaces must dispatch through one action core")
    assert [call.args[0] for call in core.calls] == ["graph_table"] * 3
    assert {call.kwargs["action"] for call in core.calls} == {"query"}
    assert [call.kwargs["sql"] for call in core.calls] == list(
        graph_tools.CATALOG_STATEMENTS.values()
    )


def test_response_is_the_catalogs_tables_columns_projection(client):
    api = client(_full_catalog())
    body = api.post(ROUTE, json={}).json()

    assert body["status"] == "success"
    assert body["counts"] == {
        "catalogs": 1,
        "schemas": 1,
        "tables": 2,
        "columns": 2,
    }
    catalog = body["catalogs"][0]
    assert catalog["catalog"] == "epistemic-graph"
    schema = catalog["schemas"][0]
    assert schema["schema"] == "public"
    assert [(t["name"], t["kind"]) for t in schema["tables"]] == [
        ("nodes", "table"),
        ("hot_incidents", "view"),
    ]
    column = schema["tables"][0]["columns"][0]
    assert column == {
        "name": "id",
        "position": 1,
        "data_type": "text",
        "udt_name": "text",
        "nullable": True,
        "primary_key": None,
    }
    # Honest capability reporting: the engine's key_column_usage /
    # table_constraints relations are shaped-but-empty and is_nullable is
    # hardcoded 'YES', so neither facet may be presented as authoritative.
    assert body["capabilities"] == {"primary_keys": False, "nullability": False}


def test_no_caller_supplied_sql_can_reach_the_engine(client):
    """A ``sql`` field in the body is inert — the statements stay constant."""
    api = client(_full_catalog())

    with observe(kg_server, "_execute_tool") as core:
        response = api.post(
            ROUTE, json={"sql": "SELECT * FROM nodes", "query": "DROP TABLE nodes"}
        )

    assert response.status_code == 200, response.text
    executed = [call.kwargs["sql"] for call in core.calls]
    assert executed == list(graph_tools.CATALOG_STATEMENTS.values())
    assert all(statement.startswith("SELECT ") for statement in executed)
    assert not any("nodes" in statement for statement in executed)


def test_injection_shaped_schema_filter_is_rejected_before_dispatch(client):
    api = client(_full_catalog())

    with observe(kg_server, "_execute_tool") as core:
        response = api.post(ROUTE, json={"schema": "public'; DROP TABLE nodes --"})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_schema_filter"
    core.assert_not_called(why="a malformed filter must never reach the engine")


def test_schema_filter_is_applied_in_python_not_in_sql(client):
    api = client(_full_catalog())

    with observe(kg_server, "_execute_tool") as core:
        response = api.post(ROUTE, json={"schema": "public"})

    assert response.status_code == 200, response.text
    assert response.json()["counts"]["tables"] == 2
    assert [call.kwargs["sql"] for call in core.calls] == list(
        graph_tools.CATALOG_STATEMENTS.values()
    ), "the filter must not change the statements sent to the engine"


def test_unknown_schema_is_404_not_an_empty_success(client):
    api = client(_full_catalog())
    response = api.post(ROUTE, json={"schema": "no_such_schema"})

    assert response.status_code == 404
    assert response.json()["code"] == "schema_not_found"


def test_unreadable_catalog_fails_closed_instead_of_reporting_no_tables(client):
    """An engine error must NOT be served as a healthy, empty schema tree."""
    api = client({})  # every statement returns []

    response = api.post(ROUTE, json={})

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["code"] == "catalog_unavailable"
    assert "catalogs" not in payload


def test_engine_error_object_is_not_decoded_as_rows(client):
    api = client(
        {
            graph_tools.CATALOG_STATEMENTS["tables"]: {"error": "engine down"},
            graph_tools.CATALOG_STATEMENTS["columns"]: _COLUMN_ROWS,
            graph_tools.CATALOG_STATEMENTS["primary_keys"]: [],
        }
    )
    response = api.post(ROUTE, json={})
    assert response.status_code == 503


def test_unauthenticated_caller_is_denied():
    """No verified GraphSession -> 403, and the engine is never reached."""
    app = FastAPI()
    graph_api._mount_sql_schema_route(app, prefix="/api")

    with observe(kg_server, "_execute_tool") as core, suspend_session():
        response = TestClient(app).post(ROUTE, json={})

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    core.assert_not_called(why="an unauthorized caller must not reach the catalog")


def test_route_is_mounted_as_a_documented_post_only_operation():
    app = FastAPI()
    graph_api._mount_sql_schema_route(app, prefix="/api")

    matches = [r for r in app.routes if getattr(r, "path", None) == ROUTE]
    assert len(matches) == 1
    assert set(matches[0].methods) == {"POST"}
    # OpenAPI-visible with a summary AND description: check_openapi_coverage
    # ratchets on undocumented raw-Starlette routes and on documented
    # operations missing either field.
    operation = app.openapi()["paths"][ROUTE]["post"]
    assert operation["summary"]
    assert operation["description"]


def test_gateway_entrypoint_mounts_the_route():
    """The production entrypoint must actually call the mounter (not just
    define it) — the ``reachable != invoked`` failure this repo keeps hitting."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(graph_api.register_graph_routes))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_mount_sql_schema_route" in called
